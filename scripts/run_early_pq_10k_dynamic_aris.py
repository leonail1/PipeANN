#!/usr/bin/env python3
"""ARIS runner for early-PQ-from-10k dynamic update experiments.

The experiment starts from an empty dynamic index, trains PQ pivots on the
first N seed vectors, materializes from flat mode when online inserts cross
that threshold, then continues to 1M before running 60% delete/insert cycles.
It reuses the established PipeANN ARIS helpers for truth generation,
route/L calibration, and matched-reference drift accounting.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dynamic_delete_pq_drift_aris as aris  # noqa: E402
import run_pq_drift_1m_aris as pq1m  # noqa: E402


DEFAULT_BUCKETS = pq1m.DEFAULT_BUCKETS
DEFAULT_L_SWEEP = [50, 75, 100, 150, 200, 250, 300, 400, 450, 470, 500]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return aris.read_jsonl(path)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(value) for k, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    return obj


def parse_l_sweep(value: str) -> list[int]:
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError("--l-sweep must contain at least one integer")
    if parsed[0] <= 0:
        raise argparse.ArgumentTypeError("--l-sweep values must be positive")
    return parsed


def build_paths(repo: Path, out_dir: Path | None) -> aris.Paths:
    out = out_dir or repo / "experiments" / f"early_pq_10k_dynamic_aris_{now_stamp()}"
    paths = aris.Paths(
        repo=repo,
        out=out,
        raw=out / "raw",
        logs=out / "logs",
        evidence=out / "evidence",
        data=out / "data",
        labels=out / "labels",
        truth=out / "truth",
        indexes=out / "indexes",
    )
    aris.mkdirs(paths)
    return paths


def write_claim_registry(paths: aris.Paths) -> None:
    claims = [
        {
            "id": "E1_EARLY_PQ_ONLINE_10K",
            "claim": "PQ pivots are trained at the configured early threshold and reused when online inserts materialize the 1M index.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "E2_NO_RETRAIN_DYNAMIC_5CYCLE",
            "claim": "After at least five 60% mark-delete/new-vector insert cycles, no-retrain search reaches recall@10 >= 98 with avg and p95 latency below the configured limit.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "E3_MATCHED_REFERENCE_DRIFT",
            "claim": "No-retrain candidates match the per-cycle retrain reference recall target on every selected filter bucket.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "E4_DELETE_INSERT_MAINTENANCE",
            "claim": "Each cycle uses mark-delete/tombstone semantics, fast per-vector delete, and merge/insert maintenance evidence is recorded separately from foreground search.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "E5_PQ_MAINTENANCE_STRATEGY",
            "claim": "If early no-retrain PQ drift fails latency or recall, a triggered/background maintenance strategy is selected and compared against no-retrain and full-retrain references.",
            "status": "PENDING",
            "evidence": [],
        },
    ]
    write_json(paths.out / "claim_registry.json", {"created_utc": now_stamp(), "claims": claims})


def update_claim(paths: aris.Paths, claim_id: str, status: str, evidence: list[str], note: str) -> None:
    registry_path = paths.out / "claim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for claim in registry["claims"]:
        if claim["id"] == claim_id:
            claim.update({"status": status, "evidence": evidence, "note": note})
            break
    else:
        raise KeyError(f"unknown claim id: {claim_id}")
    write_json(registry_path, registry)


def scale_status(args: argparse.Namespace, *, needs_cycles: bool = False, passed: bool = True) -> str:
    if not passed:
        return "FAIL"
    if args.npoints < 1_000_000:
        return "EVIDENCE_SMOKE"
    if needs_cycles and args.cycles < 5:
        return "EVIDENCE_SMOKE"
    return "EVIDENCE_MAIN"


def scale_gate_status(args: argparse.Namespace, *, needs_cycles: bool, gate_passed: bool,
                      full_scale_complete: bool) -> str:
    full_scale_expected = args.npoints >= 1_000_000 and (not needs_cycles or args.cycles >= 5)
    if gate_passed and full_scale_complete:
        return scale_status(args, needs_cycles=needs_cycles, passed=True)
    if gate_passed and not full_scale_expected:
        return scale_status(args, needs_cycles=needs_cycles, passed=True)
    if not full_scale_complete and full_scale_expected:
        return "INCOMPLETE"
    if not full_scale_complete:
        return "EVIDENCE_SMOKE_WITH_GAP"
    return "FAIL"


def write_env(paths: aris.Paths, args: argparse.Namespace) -> None:
    aris.phase0_inventory(paths, args)
    write_json(paths.evidence / "runner_config.json", jsonable(vars(args) | {
        "script": str(Path(__file__).resolve()),
        "script_sha256": aris.sha256_file(Path(__file__).resolve()),
        "l_sweep": args.l_sweep,
        "dataset_scope": "BigANN/SIFT 6M prefix split into a 1M live corpus plus replacement segments; not full SIFT100M.",
        "early_pq_semantics": "Seed PQ pivots are trained on the first early_pq_train_points vectors; zero-insert uses --online-flat-materialize so crossing the threshold materializes during insert.",
    }))


def segment_bin(paths: aris.Paths, source: Path, start: int, count: int, name: str) -> Path:
    return pq1m.segment_bin(paths, source, start, count, name)


def prepare_common(paths: aris.Paths, args: argparse.Namespace) -> dict[str, Path]:
    return pq1m.prepare_common(paths, args)


def build_index(paths: aris.Paths, args: argparse.Namespace, prefix: Path, data_bin: Path, labels: Path,
                cpu_cap: int | None = None) -> Path:
    return pq1m.build_index(paths, args, prefix, data_bin, labels, cpu_cap=cpu_cap)


def calibrate_variant(paths: aris.Paths, args: argparse.Namespace, phase: str, variant: str, cycle_idx: int,
                      prefix: Path, data_bin: Path, labels: Path, query: Path, tags: Path,
                      buckets: list[str]) -> list[dict[str, Any]]:
    return pq1m.calibrate_variant(paths, args, phase, variant, cycle_idx, prefix, data_bin, labels, query, tags, buckets)


def compare_penalty(paths: aris.Paths, phase: str, cycle_idx: int, retrain_rows: list[dict[str, Any]],
                    no_retrain_rows: list[dict[str, Any]]) -> None:
    pq1m.compare_penalty(paths, phase, cycle_idx, retrain_rows, no_retrain_rows)


def require_file_hash(record: dict[str, Any] | None, role: str) -> str:
    return pq1m.require_file_hash(record, role)


def absolute_path(paths: aris.Paths, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.repo / path


def phase_b_early(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path],
                  buckets: list[str]) -> None:
    points = args.npoints
    seed_points = args.early_pq_train_points
    if seed_points <= 0 or seed_points >= points:
        raise ValueError("--early-pq-train-points must be positive and smaller than --npoints")

    seed_data = segment_bin(paths, common["source"], 0, seed_points, f"phaseB_seed_{seed_points}.bin")
    seed_labels = paths.labels / f"phaseB_seed_{seed_points}.spmat"
    aris.write_spmat_prefix(common["labels"], seed_labels, seed_points)
    seed_prefix = paths.indexes / f"phaseB_seed_pq{args.pq_bytes}_{seed_points}"
    build_index(paths, args, seed_prefix, seed_data, seed_labels)
    seed_pivots = Path(str(seed_prefix) + "_pq_pivots.bin")

    direct_prefix = paths.indexes / f"phaseB_direct_retrain_1m_pq{args.pq_bytes}"
    build_index(paths, args, direct_prefix, common["data0"], common["labels"])
    direct_log = paths.logs / f"build_{direct_prefix.name}.log"

    zero_prefix = paths.indexes / f"phaseB_online_earlypq_no_retrain_1m_seed{seed_points}"
    zero_jsonl = paths.raw / "phaseB_zero_insert.jsonl"
    before = len(read_jsonl(zero_jsonl))
    zero_cmd = aris.driver_base_cmd(paths, args, "zero-insert-only", zero_prefix, zero_jsonl)
    zero_cmd += [
        "--data-bin", str(common["data0"]),
        "--insert-start", "0",
        "--insert-count", str(points),
        "--flat-threshold", str(seed_points),
        "--flat-build-memory-gb", str(args.flat_build_memory_gb),
        "--pq-bytes", str(args.pq_bytes),
        "--flat-pq-pivots", str(seed_pivots),
        "--base-label-file", str(common["labels"]),
        "--online-flat-materialize",
    ]
    aris.run_command(
        zero_cmd,
        cwd=paths.repo,
        log_path=paths.logs / "phaseB_online_early_pq_zero_insert.log",
        cpu_cap=args.cpu_cap,
        env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"},
    )
    zero_row = aris.latest_driver_row(zero_jsonl, before, [
        "mode", "status", "final_index_prefix", "insert_count", "insert_wall_s", "merge_wall_s",
        "live_point_count", "flat_build_memory_gb", "flat_pq_pivots", "main_index_label_size",
        "label_sidecar_loadable", "online_flat_materialize", "constructor_flat_threshold",
        "materialized_during_insert", "materialize_trigger", "raw_command",
    ])
    zero_final = absolute_path(paths, zero_row["final_index_prefix"])
    actual_zero_live = int(zero_row["live_point_count"])
    if actual_zero_live != points:
        update_claim(paths, "E1_EARLY_PQ_ONLINE_10K", "FAIL", ["raw/phaseB_zero_insert.jsonl"],
                     f"Zero-insert live count {actual_zero_live} did not match requested {points}.")
        raise RuntimeError(f"phaseB zero-insert live count mismatch: {actual_zero_live} != {points}")

    direct_record = pq1m.pq_records_for(paths, args, "direct_retrain_1m", direct_prefix, points, points, True,
                                        direct_log, direct_log)
    pq1m.validate_pq_record(direct_record, points)
    append_jsonl(paths.raw / "phaseB_pq_drift.jsonl", direct_record)

    zero_record = pq1m.pq_records_for(
        paths,
        args,
        "zero_insert_no_retrain_1m",
        zero_final,
        actual_zero_live,
        seed_points,
        False,
        paths.logs / f"build_{seed_prefix.name}.log",
        paths.logs / "phaseB_online_early_pq_zero_insert.log",
        seed_pivots=seed_pivots,
    )
    zero_record.update({
        "early_pq_train_points": seed_points,
        "insert_count": zero_row.get("insert_count"),
        "insert_wall_s": zero_row.get("insert_wall_s"),
        "merge_wall_s": zero_row.get("merge_wall_s"),
        "materialize_wall_s": zero_row.get("materialize_wall_s"),
        "flat_threshold": seed_points,
        "flat_build_memory_gb": zero_row.get("flat_build_memory_gb"),
        "zero_insert_path": "online_flat_materialize_at_early_threshold",
        "online_flat_materialize": zero_row.get("online_flat_materialize"),
        "constructor_flat_threshold": zero_row.get("constructor_flat_threshold"),
        "materialized_during_insert": zero_row.get("materialized_during_insert"),
        "materialize_trigger": zero_row.get("materialize_trigger"),
        "materialize_wall_accounting": zero_row.get("materialize_wall_accounting"),
        "seed_points": seed_points,
        "seed_pivot_hash_matches_final": require_file_hash(zero_record["seed_pq_pivots_hash"], "phaseB_seed_pivots")
        == require_file_hash(zero_record["pq_codebook_hash"], "phaseB_zero_final_pivots"),
    })
    pq1m.validate_pq_record(zero_record, points, require_seed_hash=True)
    append_jsonl(paths.raw / "phaseB_pq_drift.jsonl", zero_record)

    online_ok = (
        zero_record["seed_pivot_hash_matches_final"] is True
        and zero_row.get("online_flat_materialize") is True
        and int(zero_row.get("constructor_flat_threshold", -1)) == seed_points
        and zero_row.get("materialized_during_insert") is True
    )
    write_json(paths.out / "early_pq_train_10k_summary.json", {
        "status": "PASS" if online_ok else "FAIL",
        "npoints": points,
        "early_pq_train_points": seed_points,
        "zero_final_prefix": str(zero_final),
        "seed_prefix": str(seed_prefix),
        "direct_prefix": str(direct_prefix),
        "zero_insert_driver_row": zero_row,
        "direct_pq_record": direct_record,
        "zero_pq_record": zero_record,
        "online_materialize_evidence": {
            "constructor_flat_threshold": zero_row.get("constructor_flat_threshold"),
            "online_flat_materialize": zero_row.get("online_flat_materialize"),
            "materialized_during_insert": zero_row.get("materialized_during_insert"),
            "materialize_wall_accounting": zero_row.get("materialize_wall_accounting"),
        },
    })
    if not online_ok:
        update_claim(paths, "E1_EARLY_PQ_ONLINE_10K", "FAIL",
                     ["early_pq_train_10k_summary.json", "raw/phaseB_zero_insert.jsonl", "raw/phaseB_pq_drift.jsonl"],
                     "Early-PQ online materialization evidence failed one or more checks.")
        raise RuntimeError("early-PQ online materialization checks failed")
    update_claim(paths, "E1_EARLY_PQ_ONLINE_10K", scale_status(args),
                 ["early_pq_train_10k_summary.json", "raw/phaseB_zero_insert.jsonl", "raw/phaseB_pq_drift.jsonl"],
                 f"PQ pivots trained on {seed_points} vectors; final 1M no-retrain codebook hash matches seed pivots.")

    direct_selected = calibrate_variant(paths, args, "phaseB", "direct_retrain_1m", 0, direct_prefix,
                                        common["data0"], common["labels"], common["query"], common["tags"], buckets)
    zero_selected = calibrate_variant(paths, args, "phaseB", "zero_insert_no_retrain_1m", 0, zero_final,
                                      common["data0"], common["labels"], common["query"], common["tags"], buckets)
    compare_penalty(paths, "phaseB", 0, direct_selected, zero_selected)


def phase_c_from_early(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path],
                       buckets: list[str]) -> None:
    points = args.npoints
    zero_rows = read_jsonl(paths.raw / "phaseB_zero_insert.jsonl")
    if not zero_rows:
        raise RuntimeError("phaseC requires phaseB_zero_insert.jsonl from an early-PQ zero-insert run")
    current_prefix = absolute_path(paths, zero_rows[-1]["final_index_prefix"])
    initial_pivots = Path(str(current_prefix) + "_pq_pivots.bin")
    initial_pivot_hash = aris.file_record(initial_pivots, "phaseC_initial_early_pq_pivots")
    replacements: dict[int, bytes] = {}

    append_jsonl(paths.raw / "phaseC_cycle_inventory.jsonl", {
        "cycle_idx": 0,
        "no_retrain_prefix": str(current_prefix),
        "initial_pq_pivots": initial_pivot_hash,
        "early_pq_train_points": args.early_pq_train_points,
        "source": "phaseB_online_early_pq_zero_insert_final_prefix",
    })

    for cycle_idx in range(1, args.cycles + 1):
        delete_ids = paths.data / f"phaseC_cycle{cycle_idx:02d}_delete_ids_60pct.txt"
        delete_count = aris.make_delete_ids(delete_ids, points, args.delete_fraction, args.seed + cycle_idx)
        after_delete = paths.indexes / f"phaseC_cycle{cycle_idx:02d}_after_delete_merge"
        delete_jsonl = paths.raw / "phaseC_delete_steps.jsonl"
        before = len(read_jsonl(delete_jsonl))
        delete_cmd = aris.driver_base_cmd(paths, args, "delete-batch", current_prefix, delete_jsonl)
        delete_cmd += [
            "--dest-prefix", str(after_delete),
            "--delete-id-file", str(delete_ids),
            "--delete-count", str(delete_count),
            "--data-bin", str(common["data0"]),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(
            delete_cmd,
            cwd=paths.repo,
            log_path=paths.logs / f"phaseC_cycle{cycle_idx:02d}_delete_merge.log",
            cpu_cap=args.cpu_cap,
            env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"},
        )
        delete_row = aris.latest_driver_row(delete_jsonl, before, [
            "mode", "status", "source_prefix", "dest_prefix", "delete_count", "deleted_tag_hash",
            "delete_scope", "delete_elapsed_s", "merge_elapsed_s", "live_point_count", "raw_command",
        ])
        delete_row["cycle_idx"] = cycle_idx

        insert_segment = segment_bin(paths, common["source"], cycle_idx * points, delete_count,
                                     f"phaseC_cycle{cycle_idx:02d}_insert_vectors.bin")
        dest = paths.indexes / f"phaseC_cycle{cycle_idx:02d}_no_retrain_after_insert"
        insert_jsonl = paths.raw / "phaseC_no_retrain_cycles.jsonl"
        before = len(read_jsonl(insert_jsonl))
        insert_cmd = aris.driver_base_cmd(paths, args, "insert-only", after_delete, insert_jsonl)
        insert_cmd += [
            "--dest-prefix", str(dest),
            "--data-bin", str(insert_segment),
            "--insert-start", "0",
            "--insert-count", str(delete_count),
            "--insert-tag-file", str(delete_ids),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(
            insert_cmd,
            cwd=paths.repo,
            log_path=paths.logs / f"phaseC_cycle{cycle_idx:02d}_insert_merge.log",
            cpu_cap=args.cpu_cap,
            env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"},
        )
        insert_row = aris.latest_driver_row(insert_jsonl, before, [
            "mode", "status", "source_prefix", "dest_prefix", "insert_count", "insert_scope",
            "inserted_tag_hash", "insert_elapsed_s", "merge_elapsed_s", "live_point_count", "raw_command",
        ])
        insert_row["cycle_idx"] = cycle_idx
        current_prefix = dest

        deleted_tags = [int(line) for line in delete_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        replacements.update(aris.load_segment_replacements(insert_segment, deleted_tags))
        live_data = paths.data / f"phaseC_cycle{cycle_idx:02d}_live_data_by_tag.bin"
        aris.materialize_live_data(common["data0"], replacements, live_data, points)
        retrain_prefix = paths.indexes / f"phaseC_cycle{cycle_idx:02d}_retrain_each_cycle"
        build_index(paths, args, retrain_prefix, live_data, common["labels"])

        retrain_pivots = Path(str(retrain_prefix) + "_pq_pivots.bin")
        no_retrain_pivots = Path(str(current_prefix) + "_pq_pivots.bin")
        no_retrain_pivot_record = aris.file_record(no_retrain_pivots, f"phaseC_cycle{cycle_idx:02d}_no_retrain_pivots")
        pivot_unchanged = (
            require_file_hash(no_retrain_pivot_record, f"phaseC_cycle{cycle_idx:02d}_no_retrain_pivots")
            == require_file_hash(initial_pivot_hash, "phaseC_initial_early_pq_pivots")
        )
        append_jsonl(paths.raw / "phaseC_cycle_inventory.jsonl", {
            "cycle_idx": cycle_idx,
            "delete_step": delete_row,
            "insert_step": insert_row,
            "insert_segment": aris.file_record(insert_segment, f"phaseC_cycle{cycle_idx:02d}_insert_segment"),
            "live_data": aris.file_record(live_data, f"phaseC_cycle{cycle_idx:02d}_live_data"),
            "retrain_prefix": str(retrain_prefix),
            "no_retrain_prefix": str(current_prefix),
            "initial_pq_pivots": initial_pivot_hash,
            "no_retrain_pq_pivots": no_retrain_pivot_record,
            "retrain_pq_pivots": aris.file_record(retrain_pivots, f"phaseC_cycle{cycle_idx:02d}_retrain_pivots"),
            "no_retrain_pivot_hash_unchanged": pivot_unchanged,
            "replacement_policy": "delete current live tags then insert new BigANN/SIFT segment vectors reusing deleted tags",
            "early_pq_train_points": args.early_pq_train_points,
        })
        if not pivot_unchanged:
            update_claim(paths, "E2_NO_RETRAIN_DYNAMIC_5CYCLE", "FAIL",
                         ["raw/phaseC_cycle_inventory.jsonl"],
                         f"No-retrain PQ pivots changed in cycle {cycle_idx}.")
            raise RuntimeError(f"phaseC no-retrain pivot hash changed in cycle {cycle_idx}")

        retrain_selected = calibrate_variant(paths, args, "phaseC", "retrain_each_cycle", cycle_idx, retrain_prefix,
                                             live_data, common["labels"], common["query"], common["tags"], buckets)
        no_retrain_selected = calibrate_variant(paths, args, "phaseC", "no_retrain_across_cycles", cycle_idx,
                                                current_prefix, live_data, common["labels"], common["query"],
                                                common["tags"], buckets)
        compare_penalty(paths, "phaseC", cycle_idx, retrain_selected, no_retrain_selected)


def phase_d_core_sweep(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path]) -> None:
    data_bin = common["data0"]
    inventory = read_jsonl(paths.raw / "phaseC_cycle_inventory.jsonl")
    live_records = [row for row in inventory if int(row.get("cycle_idx", -1)) > 0 and row.get("live_data")]
    if live_records:
        last_live = Path(live_records[-1]["live_data"]["path"])
        if last_live.exists():
            data_bin = last_live
    for core in args.pq_core_sweep:
        local_args = copy.copy(args)
        local_args.cpu_cap = core
        prefix = paths.indexes / f"phaseD_retrain_core{core}"
        started = time.time()
        build_index(paths, local_args, prefix, data_bin, common["labels"], cpu_cap=core)
        elapsed = time.time() - started
        log_path = paths.logs / f"build_{prefix.name}.log"
        append_jsonl(paths.raw / "phaseD_pq_core_sweep.jsonl", {
            "phase": "phaseD",
            "core_count": core,
            "prefix": str(prefix),
            "data_bin": str(data_bin),
            "build_wall_s": elapsed,
            "pq_train_wall_s": aris.extract_log_seconds(log_path, r"Pivots generated in ([0-9.]+)s"),
            "pq_recode_wall_s": aris.extract_log_seconds(log_path, r"Compressed data written in: ([0-9.]+)s"),
            "pq_training_points": aris.extract_log_int(log_path, r"Generating PQ pivots with training data of size: ([0-9]+)"),
            "pq_codebook_hash": aris.file_record(Path(str(prefix) + "_pq_pivots.bin"), f"phaseD_core{core}_pq_pivots"),
            "pq_code_hash": aris.file_record(Path(str(prefix) + "_pq_compressed.bin"), f"phaseD_core{core}_pq_codes"),
            "log_path": str(log_path),
        })


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_rows(paths: aris.Paths, stem: str, rows: list[dict[str, Any]]) -> None:
    jsonl_path = paths.out / f"{stem}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = paths.out / f"{stem}.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def flatten_phase_selected(paths: aris.Paths, phase: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wrapper in read_jsonl(paths.raw / f"{phase}_selected_route_l.jsonl"):
        selected = dict(wrapper.get("selected", {}))
        selected.update({
            "phase": wrapper.get("phase", phase),
            "variant": wrapper.get("variant"),
            "cycle_idx": wrapper.get("cycle_idx"),
            "selector_type": wrapper.get("selector_type"),
            "bucket": wrapper.get("bucket"),
        })
        avg_ms = float(selected.get("avg_latency_us", 0.0)) / 1000.0
        p95_ms = float(selected.get("p95_latency_us", 0.0)) / 1000.0
        recall = float(selected.get("recall@10", selected.get("recall", 0.0)))
        selected.update({
            "avg_latency_ms": avg_ms,
            "p95_latency_ms": p95_ms,
            "recall_pass_ge_98": recall >= 98.0,
            "avg_latency_pass": avg_ms < args.latency_ms,
            "p95_latency_pass": p95_ms < args.latency_ms,
            "avg_p95_latency_pass": avg_ms < args.latency_ms and p95_ms < args.latency_ms,
            "goal_pass_recall_avg_p95": recall >= 98.0 and avg_ms < args.latency_ms and p95_ms < args.latency_ms,
        })
        rows.append(selected)
    return rows


def delete_insert_rows(paths: aris.Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(paths.raw / "phaseC_delete_steps.jsonl"):
        out = dict(row)
        delete_count = float(out.get("delete_count", 0) or 0)
        elapsed = float(out.get("delete_elapsed_s", 0) or 0)
        out.update({
            "operation": "mark_delete_then_merge",
            "delete_ms_per_vector": (elapsed * 1000.0 / delete_count) if delete_count else None,
        })
        rows.append(out)
    for row in read_jsonl(paths.raw / "phaseC_no_retrain_cycles.jsonl"):
        out = dict(row)
        insert_count = float(out.get("insert_count", 0) or 0)
        elapsed = float(out.get("insert_elapsed_s", 0) or 0)
        out.update({
            "operation": "insert_replacement_then_merge",
            "insert_ms_per_vector": (elapsed * 1000.0 / insert_count) if insert_count else None,
        })
        rows.append(out)
    return rows


def strategy_rows(selected_rows: list[dict[str, Any]], phase_d_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        if row.get("phase") != "phaseC":
            continue
        variant = row.get("variant")
        strategy = {
            "no_retrain_across_cycles": "no_retrain_early10k_codebook",
            "retrain_each_cycle": "full_retrain_each_cycle_reference",
        }.get(str(variant), str(variant))
        rows.append({
            "strategy": strategy,
            "phase": row.get("phase"),
            "cycle_idx": row.get("cycle_idx"),
            "selector_type": row.get("selector_type"),
            "bucket": row.get("bucket"),
            "route": row.get("route"),
            "search_l": row.get("search_l"),
            "recall@10": row.get("recall@10"),
            "avg_latency_ms": row.get("avg_latency_ms"),
            "p95_latency_ms": row.get("p95_latency_ms"),
            "goal_pass_recall_avg_p95": row.get("goal_pass_recall_avg_p95"),
            "foreground_search_strategy": strategy == "no_retrain_early10k_codebook",
        })
    for row in phase_d_rows:
        rows.append({
            "strategy": "background_full_retrain_cost_probe",
            "phase": "phaseD",
            "core_count": row.get("core_count"),
            "build_wall_s": row.get("build_wall_s"),
            "pq_train_wall_s": row.get("pq_train_wall_s"),
            "pq_recode_wall_s": row.get("pq_recode_wall_s"),
            "pq_training_points": row.get("pq_training_points"),
            "foreground_search_strategy": False,
        })
    return rows


def summarize_selected(rows: list[dict[str, Any]], *, variant: str | None = None) -> dict[str, Any]:
    filtered = [row for row in rows if variant is None or row.get("variant") == variant]
    if not filtered:
        return {"count": 0}
    return {
        "count": len(filtered),
        "goal_pass_count": sum(1 for row in filtered if row.get("goal_pass_recall_avg_p95") is True),
        "recall_pass_count": sum(1 for row in filtered if row.get("recall_pass_ge_98") is True),
        "avg_latency_pass_count": sum(1 for row in filtered if row.get("avg_latency_pass") is True),
        "p95_latency_pass_count": sum(1 for row in filtered if row.get("p95_latency_pass") is True),
        "min_recall": min(float(row.get("recall@10", 0.0)) for row in filtered),
        "max_avg_latency_ms": max(float(row.get("avg_latency_ms", 0.0)) for row in filtered),
        "max_p95_latency_ms": max(float(row.get("p95_latency_ms", 0.0)) for row in filtered),
        "median_avg_latency_ms": statistics.median(float(row.get("avg_latency_ms", 0.0)) for row in filtered),
    }


def matched_reference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    matched = [row for row in rows if row.get("matched_reference_status") == "matched"]
    unmatched = [row for row in rows if row.get("matched_reference_status") != "matched"]
    deltas = [
        float(row.get("matched_reference_recall_delta_ms"))
        for row in rows
        if row.get("matched_reference_recall_delta_ms") is not None
    ]
    return {
        "count": len(rows),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "max_matched_reference_delta_ms": max(deltas) if deltas else None,
        "median_matched_reference_delta_ms": statistics.median(deltas) if deltas else None,
    }


def write_ppt_ready_plots(paths: aris.Paths, no_retrain_rows: list[dict[str, Any]],
                          drift_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment package set.
        write_json(paths.out / "figures" / "plot_status.json", {
            "status": "SKIPPED",
            "reason": f"matplotlib unavailable: {exc}",
        })
        return

    figures = paths.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if no_retrain_rows:
        by_cycle: dict[int, dict[str, float]] = {}
        for row in no_retrain_rows:
            cycle = int(row.get("cycle_idx", 0))
            bucket = by_cycle.setdefault(cycle, {"max_avg": 0.0, "max_p95": 0.0, "min_recall": 100.0})
            bucket["max_avg"] = max(bucket["max_avg"], float(row.get("avg_latency_ms", 0.0)))
            bucket["max_p95"] = max(bucket["max_p95"], float(row.get("p95_latency_ms", 0.0)))
            bucket["min_recall"] = min(bucket["min_recall"], float(row.get("recall@10", 0.0)))
        cycles = sorted(by_cycle)
        fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
        ax1.plot(cycles, [by_cycle[c]["max_avg"] for c in cycles], marker="o", label="max avg latency")
        ax1.plot(cycles, [by_cycle[c]["max_p95"] for c in cycles], marker="s", label="max p95 latency")
        ax1.axhline(args.latency_ms, color="#b3261e", linewidth=1.2, linestyle="--", label=f"{args.latency_ms:g} ms limit")
        ax1.set_xlabel("cycle")
        ax1.set_ylabel("latency (ms)")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(cycles, [by_cycle[c]["min_recall"] for c in cycles], color="#146c43", marker="^", label="min recall@10")
        ax2.axhline(98.0, color="#146c43", linewidth=1.0, linestyle=":", label="98 recall")
        ax2.set_ylabel("recall@10")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "early_pq_no_retrain_latency_recall_by_cycle.png", dpi=180)
        plt.close(fig)

    if drift_rows:
        matched = [row for row in drift_rows if row.get("matched_reference_recall_delta_ms") is not None]
        if matched:
            values = [float(row["matched_reference_recall_delta_ms"]) for row in matched]
            fig, ax = plt.subplots(figsize=(7.0, 3.6))
            ax.hist(values, bins=min(20, max(5, len(values) // 5)), color="#305f8f", alpha=0.85)
            ax.axvline(0.0, color="#333333", linewidth=1.0)
            ax.set_xlabel("matched-reference latency delta (ms)")
            ax.set_ylabel("bucket count")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "early_pq_matched_reference_delta_hist.png", dpi=180)
            plt.close(fig)

    write_json(figures / "plot_status.json", {
        "status": "OK",
        "figures": [
            "figures/early_pq_no_retrain_latency_recall_by_cycle.png",
            "figures/early_pq_matched_reference_delta_hist.png",
        ],
    })


def finalize_outputs_and_claims(paths: aris.Paths, args: argparse.Namespace) -> None:
    phaseb_selected = flatten_phase_selected(paths, "phaseB", args)
    phasec_selected = flatten_phase_selected(paths, "phaseC", args)
    selected = phaseb_selected + phasec_selected
    no_retrain = [row for row in phasec_selected if row.get("variant") == "no_retrain_across_cycles"]
    drift_rows = read_jsonl(paths.raw / "phaseC_penalty.jsonl")
    phase_d_rows = read_jsonl(paths.raw / "phaseD_pq_core_sweep.jsonl")
    maintenance_rows = delete_insert_rows(paths)

    write_rows(paths, "early_pq_selected_route_l", selected)
    write_rows(paths, "early_pq_no_retrain_dynamic_results", no_retrain)
    write_rows(paths, "early_pq_no_retrain_latency_profile", no_retrain)
    write_rows(paths, "early_pq_drift_compare", drift_rows)
    write_rows(paths, "early_pq_delete_insert_maintenance", maintenance_rows)
    write_rows(paths, "pq_drift_strategy_compare", strategy_rows(selected, phase_d_rows))
    write_rows(paths, "optimized_dynamic_update_results", no_retrain)
    write_rows(paths, "targeted_latency_profile", no_retrain)

    matched_summary = matched_reference_summary(drift_rows)
    no_retrain_summary = summarize_selected(no_retrain, variant="no_retrain_across_cycles")
    selected_summary = summarize_selected(selected)
    write_json(paths.out / "early_pq_matched_reference_summary.json", matched_summary)

    no_retrain_complete = (
        args.npoints >= 1_000_000
        and args.cycles >= 5
        and no_retrain_summary.get("count", 0) == args.cycles * len(args.selector_types.split(",")) * len(args.buckets.split(","))
    )
    no_retrain_gate_pass = (
        no_retrain_summary.get("count", 0) > 0
        and no_retrain_summary.get("goal_pass_count") == no_retrain_summary.get("count")
    )
    no_retrain_pass = no_retrain_complete and no_retrain_gate_pass
    matched_pass = (
        matched_summary.get("count") == args.cycles * len(args.selector_types.split(",")) * len(args.buckets.split(","))
        and matched_summary.get("unmatched_count") == 0
    )
    delete_rows = [row for row in maintenance_rows if row.get("operation") == "mark_delete_then_merge"]
    delete_pass = bool(delete_rows) and all(
        row.get("delete_ms_per_vector") is not None and float(row["delete_ms_per_vector"]) < args.delete_ms_per_vector_limit
        for row in delete_rows
    )

    if no_retrain:
        update_claim(paths, "E2_NO_RETRAIN_DYNAMIC_5CYCLE",
                     scale_gate_status(args, needs_cycles=True, gate_passed=no_retrain_gate_pass,
                                       full_scale_complete=no_retrain_complete),
                     ["early_pq_no_retrain_dynamic_results.jsonl", "early_pq_no_retrain_latency_profile.csv",
                      "raw/phaseC_selected_route_l.jsonl"],
                     f"No-retrain selected rows: {no_retrain_summary}; "
                     f"full_scale_complete={no_retrain_complete}.")
    if drift_rows:
        update_claim(paths, "E3_MATCHED_REFERENCE_DRIFT", scale_status(args, needs_cycles=True, passed=matched_pass),
                     ["early_pq_drift_compare.jsonl", "early_pq_matched_reference_summary.json", "raw/phaseC_penalty.jsonl"],
                     f"Matched-reference summary: matched={matched_summary.get('matched_count')} unmatched={matched_summary.get('unmatched_count')}.")
    if maintenance_rows:
        update_claim(paths, "E4_DELETE_INSERT_MAINTENANCE", scale_status(args, needs_cycles=True, passed=delete_pass),
                     ["early_pq_delete_insert_maintenance.jsonl", "raw/phaseC_delete_steps.jsonl", "raw/phaseC_no_retrain_cycles.jsonl"],
                     f"Delete ms/vector limit={args.delete_ms_per_vector_limit}; rows={len(delete_rows)}.")
    if no_retrain:
        if no_retrain_pass and matched_pass:
            update_claim(paths, "E5_PQ_MAINTENANCE_STRATEGY", scale_status(args, needs_cycles=True, passed=True),
                         ["pq_drift_strategy_compare.jsonl", "early_pq_no_retrain_dynamic_results.jsonl"],
                         "No-retrain early-10k PQ satisfied recall/avg/p95 latency and matched-reference gates; triggered retrain not required for this evidence set.")
        else:
            update_claim(paths, "E5_PQ_MAINTENANCE_STRATEGY", "REQUIRES_OPTIMIZATION",
                         ["pq_drift_strategy_compare.jsonl", "early_pq_no_retrain_dynamic_results.jsonl",
                          "early_pq_drift_compare.jsonl"],
                         "No-retrain early-10k PQ did not satisfy all gates; run triggered/background PQ maintenance experiments before final acceptance.")

    registry = json.loads((paths.out / "claim_registry.json").read_text(encoding="utf-8"))
    write_json(paths.out / "early_pq_claim_registry.json", registry)
    write_json(paths.out / "optimized_claim_registry.json", registry)

    summary = {
        "created_utc": now_stamp(),
        "npoints": args.npoints,
        "early_pq_train_points": args.early_pq_train_points,
        "cycles": args.cycles,
        "delete_fraction": args.delete_fraction,
        "latency_ms_limit": args.latency_ms,
        "selected": selected_summary,
        "phaseC_no_retrain": no_retrain_summary,
        "matched_reference": matched_summary,
        "claims": registry,
        "files": {
            rel: aris.file_record(paths.out / rel, rel)
            for rel in [
                "early_pq_train_10k_summary.json",
                "early_pq_claim_registry.json",
                "optimized_claim_registry.json",
                "early_pq_selected_route_l.csv",
                "early_pq_no_retrain_dynamic_results.csv",
                "early_pq_no_retrain_latency_profile.csv",
                "early_pq_drift_compare.csv",
                "pq_drift_strategy_compare.csv",
                "early_pq_delete_insert_maintenance.csv",
                "targeted_latency_profile.csv",
                "optimized_dynamic_update_results.csv",
            ]
        },
    }
    write_json(paths.out / "summary.json", summary)
    if not args.skip_plots:
        write_ppt_ready_plots(paths, no_retrain, drift_rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["phaseB", "phaseC", "phaseD", "all"], default="all")
    parser.add_argument("--cpu-cap", type=int, default=16)
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--binary-root", type=Path, default=None)
    parser.add_argument("--flat-build-memory-gb", type=int, default=None)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--query-beamwidth", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--npoints", type=int, default=1_000_000)
    parser.add_argument("--early-pq-train-points", type=int, default=10_000)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--bigann-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/data/sift_query_1000.bin"))
    parser.add_argument("--base-labels", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels/base_1m.spmat"))
    parser.add_argument("--query-label-dir", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels"))
    parser.add_argument("--sift100m-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--seed", type=int, default=1162026)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--delete-fraction", type=float, default=0.60)
    parser.add_argument("--delete-ms-per-vector-limit", type=float, default=1.0)
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--buckets", default=",".join(DEFAULT_BUCKETS))
    parser.add_argument("--selector-types", default="intersect,range")
    parser.add_argument("--l-sweep", type=parse_l_sweep, default=DEFAULT_L_SWEEP)
    parser.add_argument("--insert-threads", type=int, default=16)
    parser.add_argument("--merge-threads", type=int, default=16)
    parser.add_argument("--pq-core-sweep", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 4, 8, 16])
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    if args.cpu_start < 0:
        raise ValueError("--cpu-start must be non-negative")
    if args.cpu_cap < 0:
        raise ValueError("--cpu-cap must be non-negative")
    cpu_count = os.cpu_count() or 0
    if args.cpu_cap > 0 and cpu_count > 0 and args.cpu_start + args.cpu_cap > cpu_count:
        raise ValueError(f"--cpu-start + --cpu-cap exceeds available CPUs ({cpu_count})")
    if args.flat_build_memory_gb is None:
        args.flat_build_memory_gb = args.memory_gb
    if args.flat_build_memory_gb <= 0:
        raise ValueError("--flat-build-memory-gb must be positive")
    if args.beamwidth <= 0:
        raise ValueError("--beamwidth must be positive")
    if args.query_beamwidth is None:
        args.query_beamwidth = args.beamwidth
    if args.query_beamwidth <= 0:
        raise ValueError("--query-beamwidth must be positive")
    if not 0.0 < args.delete_fraction < 1.0:
        raise ValueError("--delete-fraction must be in (0, 1)")
    if args.early_pq_train_points <= 0 or args.early_pq_train_points >= args.npoints:
        raise ValueError("--early-pq-train-points must be positive and smaller than --npoints")
    aris.CPU_START = args.cpu_start
    aris.DEFAULT_L_SWEEP[:] = args.l_sweep
    args.seed_points = args.early_pq_train_points
    args.base_bin = args.bigann_bin
    args.phase3_buckets = args.buckets
    args.phase4_buckets = args.buckets
    args.allow_sift1m_segment_fallback = False
    args.phase4_points = args.npoints
    args.phase4_seed_points = args.early_pq_train_points
    args.phase4_flat_threshold = args.early_pq_train_points
    args.phase4_threshold = args.early_pq_train_points


def main() -> int:
    args = parse_args()
    normalize_args(args)
    paths = build_paths(args.repo, args.out_dir)
    pq1m.install_command_logger(paths)
    if not (paths.out / "claim_registry.json").exists():
        write_claim_registry(paths)
    write_env(paths, args)
    common = prepare_common(paths, args)
    buckets = [bucket for bucket in args.buckets.split(",") if bucket]

    if args.phase in {"phaseB", "all"}:
        phase_b_early(paths, args, common, buckets)
    if args.phase in {"phaseC", "all"}:
        phase_c_from_early(paths, args, common, buckets)
    if args.phase in {"phaseD", "all"}:
        phase_d_core_sweep(paths, args, common)
    finalize_outputs_and_claims(paths, args)
    print(paths.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
