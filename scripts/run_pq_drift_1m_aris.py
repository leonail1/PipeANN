#!/usr/bin/env python3
"""ARIS runner for 1M PQ drift experiments.

This script composes existing PipeANN dynamic update driver modes. It is
intentionally orchestration-only and keeps raw calibration rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dynamic_delete_pq_drift_aris as aris  # noqa: E402


DEFAULT_BUCKETS = [
    "u1e-03",
    "u3e-03",
    "u1e-02",
    "u5e-02",
    "u1e-01",
    "u25",
    "u30",
    "u50",
    "u75",
    "u100",
]

DEFAULT_L_SWEEP = [50, 75, 100, 150, 200, 250, 300, 400, 450, 470]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return aris.read_jsonl(path)


def parse_l_sweep(value: str) -> list[int]:
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError("--l-sweep must contain at least one integer")
    if parsed[0] <= 0:
        raise argparse.ArgumentTypeError("--l-sweep values must be positive")
    return parsed


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_file_hash(record: dict[str, Any] | None, role: str) -> str:
    if not record or record.get("exists") is not True or not record.get("hash"):
        raise RuntimeError(f"required file/hash missing for {role}: {record}")
    return str(record["hash"])


def build_paths(repo: Path, out_dir: Path | None) -> aris.Paths:
    out = out_dir or repo / "experiments" / f"pq_drift_1m_aris_{now_stamp()}"
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


def install_command_logger(paths: aris.Paths) -> None:
    original = aris.run_command

    def tracked_run_command(cmd: list[str], *, cwd: Path, log_path: Path, cpu_cap: int = 0,
                            env_extra: dict[str, str] | None = None, check: bool = True):
        record = {
            "time_utc": now_stamp(),
            "cmd": cmd,
            "cwd": str(cwd),
            "log_path": str(log_path),
            "cpu_cap": cpu_cap,
            "cpu_start": getattr(aris, "CPU_START", 0),
            "env_extra": env_extra or {},
            "check": check,
        }
        append_jsonl(paths.raw / "commands.jsonl", record)
        return original(cmd, cwd=cwd, log_path=log_path, cpu_cap=cpu_cap, env_extra=env_extra, check=check)

    aris.run_command = tracked_run_command


def write_claim_registry(paths: aris.Paths) -> None:
    claims = [
        {
            "id": "P2_PHASEB_1M_PQ_DRIFT",
            "claim": "At the configured point scale, no-retrain PQ drift is quantified against direct fresh-retrain PQ.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "P3_PHASEC_CYCLE_DRIFT",
            "claim": "Across configured 60% delete + insert cycles, no-retrain drift penalty is quantified against per-cycle retrain.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "P4_PQ_RETRAIN_CORE_SWEEP",
            "claim": "PQ retraining/recode cost is measured across CPU core counts.",
            "status": "PENDING",
            "evidence": [],
        },
    ]
    write_json(paths.out / "claim_registry.json", {"created_utc": now_stamp(), "claims": claims})


def update_claim(paths: aris.Paths, claim_id: str, status: str, evidence: list[str], note: str) -> None:
    registry = json.loads((paths.out / "claim_registry.json").read_text(encoding="utf-8"))
    for claim in registry["claims"]:
        if claim["id"] == claim_id:
            claim.update({"status": status, "evidence": evidence, "note": note})
            break
    write_json(paths.out / "claim_registry.json", registry)


def scale_status(args: argparse.Namespace, *, needs_cycles: bool = False, all_selected_pass: bool = True) -> str:
    if not all_selected_pass:
        return "EVIDENCE_WITH_FAILED_RECALL"
    if args.npoints < 1_000_000:
        return "EVIDENCE_SMOKE"
    if needs_cycles and args.cycles < 5:
        return "EVIDENCE_SMOKE"
    return "EVIDENCE_MAIN"


def write_env(paths: aris.Paths, args: argparse.Namespace) -> None:
    aris.phase0_inventory(paths, args)
    write_json(paths.evidence / "runner_config.json", jsonable(vars(args) | {
        "l_sweep": args.l_sweep,
        "script": str(Path(__file__).resolve()),
        "script_sha256": aris.sha256_file(Path(__file__).resolve()),
        "note": "This runner uses BigANN/SIFT 6M prefix as a segment source; it must not be described as full SIFT100M.",
    }))


def segment_bin(paths: aris.Paths, source: Path, start: int, count: int, name: str) -> Path:
    dest = paths.data / name
    if dest.exists():
        return dest
    total, dim = aris.read_bin_header(source)
    if start + count > total:
        raise ValueError(f"segment exceeds source: start={start}, count={count}, total={total}")
    aris.copy_bin_segment_with_wrap(source, dest, start, count, total, dim)
    return dest


def prepare_common(paths: aris.Paths, args: argparse.Namespace) -> dict[str, Path]:
    source = aris.as_repo_path(paths.repo, args.bigann_bin)
    labels = aris.as_repo_path(paths.repo, args.base_labels)
    query = aris.as_repo_path(paths.repo, args.query_bin)
    label_dir = aris.as_repo_path(paths.repo, args.query_label_dir)
    data0 = segment_bin(paths, source, 0, args.npoints, f"segment00_{args.npoints}.bin")
    labels_1m = paths.labels / f"base_labels_{args.npoints}.spmat"
    aris.write_spmat_prefix(labels, labels_1m, args.npoints)
    tags = paths.data / f"identity_tags_{args.npoints}.bin"
    aris.write_identity_tag_bin(tags, args.npoints)
    query_label_records: list[dict[str, Any]] = []
    for selector_type in args.selector_types.split(","):
        for bucket in args.buckets.split(","):
            if not bucket:
                continue
            try:
                qlabel = aris.query_label_path(paths, args, selector_type, bucket)
                record = aris.file_record(qlabel, f"query_label_{selector_type}_{bucket}")
            except FileNotFoundError as exc:
                record = {
                    "role": f"query_label_{selector_type}_{bucket}",
                    "selector_type": selector_type,
                    "bucket": bucket,
                    "exists": False,
                    "error": str(exc),
                }
            query_label_records.append(record)
    write_json(paths.evidence / "input_inventory.json", {
        "bigann_bin": aris.file_record(source, "bigann_sift_6m_prefix"),
        "segment0": aris.file_record(data0, "initial_1m_segment"),
        "base_labels": aris.file_record(labels, "source_base_labels"),
        "labels_1m": aris.file_record(labels_1m, "labels_1m"),
        "query_bin": aris.file_record(query, "query_bin"),
        "query_label_dir": {
            "role": "query_label_dir",
            "path": str(label_dir),
            "exists": label_dir.exists(),
            "is_dir": label_dir.is_dir(),
        },
        "query_label_files": query_label_records,
        "identity_tags": aris.file_record(tags, "identity_tags"),
        "dataset_scope": "BigANN/SIFT 6M prefix split into 1M live corpus plus 600k replacement segments; not full SIFT100M.",
        "label_semantics": "Filter labels are synthetic and bound to stable tag ids reused by delete/insert experiments; they are not native labels for the replacement segment vectors.",
    })
    return {"source": source, "data0": data0, "labels": labels_1m, "query": query, "tags": tags}


def build_index(paths: aris.Paths, args: argparse.Namespace, prefix: Path, data_bin: Path, labels: Path,
                cpu_cap: int | None = None) -> Path:
    local_args = copy.copy(args)
    if cpu_cap is not None:
        local_args.cpu_cap = cpu_cap
    aris.build_or_reuse_index(paths, local_args, prefix, data_bin, labels)
    return prefix


def calibrate_variant(paths: aris.Paths, args: argparse.Namespace, phase: str, variant: str, cycle_idx: int,
                      prefix: Path, data_bin: Path, labels: Path, query: Path, tags: Path,
                      buckets: list[str]) -> list[dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []
    for selector_type in args.selector_types.split(","):
        for bucket in buckets:
            qlabel = aris.query_label_path(paths, args, selector_type, bucket)
            truth = paths.truth / f"{phase}_cycle{cycle_idx:02d}_{selector_type}_{bucket}.bin"
            aris.compute_truth(paths, args, data_bin, query, labels, qlabel, selector_type, truth, tag_file=tags)
            cycle_name = f"{phase}_cycle{cycle_idx:02d}_{variant}"
            selected = aris.calibrate_bucket(paths, args, prefix, data_bin, labels, query, qlabel, truth,
                                             selector_type, bucket, cycle_name)
            selected.update({"phase": phase, "variant": variant, "cycle_idx": cycle_idx})
            append_jsonl(paths.raw / f"{phase}_selected_route_l.jsonl", {
                "phase": phase,
                "variant": variant,
                "cycle_idx": cycle_idx,
                "selector_type": selector_type,
                "bucket": bucket,
                "selected": selected,
            })
            selected_rows.append(selected)
    return selected_rows


def calibration_rows(paths: aris.Paths, cycle_name: str, selector_type: str, bucket: str) -> list[dict[str, Any]]:
    return read_jsonl(paths.raw / f"calibration_{cycle_name}_{selector_type}_{bucket}.jsonl")


def compare_penalty(paths: aris.Paths, phase: str, cycle_idx: int, retrain_rows: list[dict[str, Any]],
                    no_retrain_rows: list[dict[str, Any]]) -> None:
    by_key_ref = {(r["selector_type"], r["bucket"]): r for r in retrain_rows}
    by_key_no = {(r["selector_type"], r["bucket"]): r for r in no_retrain_rows}
    for key, ref in sorted(by_key_ref.items()):
        no = by_key_no.get(key)
        if not no:
            continue
        selector_type, bucket = key
        cycle_no = f"{phase}_cycle{cycle_idx:02d}_no_retrain_across_cycles" if phase == "phaseC" else f"{phase}_cycle{cycle_idx:02d}_zero_insert_no_retrain_1m"
        candidates = calibration_rows(paths, cycle_no, selector_type, bucket)
        target = float(ref.get("recall@10", 0.0)) - 0.1
        matched = [r for r in candidates if float(r.get("recall@10", 0.0)) >= target]
        matched_row = min(matched, key=lambda r: float(r.get("avg_latency_us", float("inf")))) if matched else None
        selected_feasible_delta = (no.get("avg_latency_us", 0) - ref.get("avg_latency_us", 0)) / 1000.0
        matched_reference_delta = (
            None
            if matched_row is None
            else (matched_row.get("avg_latency_us", 0) - ref.get("avg_latency_us", 0)) / 1000.0
        )
        append_jsonl(paths.raw / f"{phase}_penalty.jsonl", {
            "phase": phase,
            "cycle_idx": cycle_idx,
            "selector_type": selector_type,
            "bucket": bucket,
            "reference_variant": "retrain_each_cycle" if phase == "phaseC" else "direct_retrain_1m",
            "no_retrain_variant": "no_retrain_across_cycles" if phase == "phaseC" else "zero_insert_no_retrain_1m",
            "reference_recall": ref.get("recall@10"),
            "no_retrain_recall": no.get("recall@10"),
            "reference_avg_latency_ms": ref.get("avg_latency_us", 0) / 1000.0,
            "no_retrain_avg_latency_ms": no.get("avg_latency_us", 0) / 1000.0,
            "selected_feasible_delta_ms": selected_feasible_delta,
            "reference_L": ref.get("search_l"),
            "no_retrain_L": no.get("search_l"),
            "l_uplift": (no.get("search_l") or 0) - (ref.get("search_l") or 0),
            "reference_route": ref.get("route"),
            "no_retrain_route": no.get("route"),
            "matched_reference_target_recall": target,
            "matched_reference_status": "matched" if matched_row else "unmatched",
            "matched_reference_delta_definition": "no-retrain fastest candidate meeting reference selected recall@10 minus 0.1pp, minus reference selected avg latency",
            "matched_reference_recall_delta_ms": matched_reference_delta,
            "matched_reference_avg_latency_ms": None if matched_row is None else matched_row.get("avg_latency_us", 0) / 1000.0,
            "matched_reference_L": None if matched_row is None else matched_row.get("search_l"),
            "matched_reference_route": None if matched_row is None else matched_row.get("route"),
        })


def pq_records_for(paths: aris.Paths, args: argparse.Namespace, variant: str, prefix: Path, points: int,
                   training_points: int, retrained: bool, train_log: Path, recode_log: Path,
                   seed_pivots: Path | None = None) -> dict[str, Any]:
    pq_pivots = Path(str(prefix) + "_pq_pivots.bin")
    pq_codes = Path(str(prefix) + "_pq_compressed.bin")
    code_point_count, code_chunks = aris.read_pq_code_header(pq_codes)
    return {
        "mode": "pq-drift-1m",
        "variant": variant,
        "requested_points": points,
        "live_point_count": points,
        "code_point_count": code_point_count,
        "code_chunks": code_chunks,
        "point_count_consistent": points == code_point_count,
        "pq_bytes": args.pq_bytes,
        "pq_retrained": retrained,
        "pq_train_core_count": args.cpu_cap,
        "pq_train_wall_s": aris.extract_log_seconds(train_log, r"Pivots generated in ([0-9.]+)s"),
        "pq_recode_wall_s": aris.extract_log_seconds(recode_log, r"Compressed data written in: ([0-9.]+)s"),
        "pq_training_points": aris.extract_log_int(train_log, r"Generating PQ pivots with training data of size: ([0-9]+)"),
        "pq_training_corpus_points": training_points,
        "insert_count": 0 if retrained else points,
        "seed_points": training_points,
        "flat_threshold": 0 if retrained else points - 1,
        "flat_build_memory_gb": args.flat_build_memory_gb,
        "zero_insert_path": "direct_build_no_flat_materialization" if retrained else "flat_until_final_materialization",
        "prefix": str(prefix),
        "pq_codebook_hash": aris.file_record(pq_pivots, f"{variant}_pq_pivots"),
        "pq_code_hash": aris.file_record(pq_codes, f"{variant}_pq_codes"),
        "seed_pq_pivots_hash": aris.file_record(seed_pivots, f"{variant}_seed_pivots") if seed_pivots else None,
    }


def validate_pq_record(record: dict[str, Any], expected_live_count: int, require_seed_hash: bool = False) -> None:
    if record.get("live_point_count") != expected_live_count:
        raise RuntimeError(
            f"{record.get('variant')} live_point_count mismatch: "
            f"{record.get('live_point_count')} != {expected_live_count}"
        )
    if record.get("code_point_count") != expected_live_count or record.get("point_count_consistent") is not True:
        raise RuntimeError(f"{record.get('variant')} PQ code count mismatch: {record}")
    require_file_hash(record["pq_codebook_hash"], f"{record.get('variant')}_pq_codebook")
    require_file_hash(record["pq_code_hash"], f"{record.get('variant')}_pq_code")
    if require_seed_hash:
        require_file_hash(record["seed_pq_pivots_hash"], f"{record.get('variant')}_seed_pivots")


def phase_b(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path], buckets: list[str]) -> None:
    points = args.npoints
    if args.seed_points <= 0 or args.seed_points >= points:
        raise ValueError("--seed-points must be positive and smaller than --npoints")
    seed_data = segment_bin(paths, common["source"], 0, args.seed_points, f"phaseB_seed_{args.seed_points}.bin")
    seed_labels = paths.labels / f"phaseB_seed_{args.seed_points}.spmat"
    aris.write_spmat_prefix(common["labels"], seed_labels, args.seed_points)
    seed_prefix = paths.indexes / f"phaseB_seed_pq{args.pq_bytes}_{args.seed_points}"
    build_index(paths, args, seed_prefix, seed_data, seed_labels)
    seed_pivots = Path(str(seed_prefix) + "_pq_pivots.bin")

    direct_prefix = paths.indexes / f"phaseB_direct_retrain_1m_pq{args.pq_bytes}"
    build_index(paths, args, direct_prefix, common["data0"], common["labels"])
    direct_log = paths.logs / f"build_{direct_prefix.name}.log"

    zero_prefix = paths.indexes / f"phaseB_zero_insert_no_retrain_1m_seed{args.seed_points}"
    zero_jsonl = paths.raw / "phaseB_zero_insert.jsonl"
    before = len(read_jsonl(zero_jsonl))
    zero_cmd = aris.driver_base_cmd(paths, args, "zero-insert-only", zero_prefix, zero_jsonl)
    zero_cmd += [
        "--data-bin", str(common["data0"]),
        "--insert-start", "0",
        "--insert-count", str(points),
        "--flat-threshold", str(points - 1),
        "--flat-build-memory-gb", str(args.flat_build_memory_gb),
        "--pq-bytes", str(args.pq_bytes),
        "--flat-pq-pivots", str(seed_pivots),
        "--base-label-file", str(common["labels"]),
    ]
    aris.run_command(zero_cmd, cwd=paths.repo, log_path=paths.logs / "phaseB_zero_insert.log",
                     cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
    zero_row = aris.latest_driver_row(zero_jsonl, before, [
        "mode", "status", "final_index_prefix", "insert_count", "insert_wall_s", "merge_wall_s",
        "live_point_count", "flat_build_memory_gb", "flat_pq_pivots", "main_index_label_size",
        "label_sidecar_loadable", "raw_command",
    ])
    zero_final = Path(zero_row["final_index_prefix"])
    if not zero_final.is_absolute():
        zero_final = paths.repo / zero_final
    actual_zero_live = int(zero_row["live_point_count"])
    if actual_zero_live != points:
        update_claim(paths, "P2_PHASEB_1M_PQ_DRIFT", "FAIL",
                     ["raw/phaseB_zero_insert.jsonl"],
                     f"Zero-insert live count {actual_zero_live} did not match requested {points}.")
        raise RuntimeError(f"phaseB zero-insert live count mismatch: {actual_zero_live} != {points}")
    direct_record = pq_records_for(paths, args, "direct_retrain_1m", direct_prefix, points, points, True,
                                   direct_log, direct_log)
    validate_pq_record(direct_record, points)
    append_jsonl(paths.raw / "phaseB_pq_drift.jsonl", direct_record)
    zero_record = pq_records_for(paths, args, "zero_insert_no_retrain_1m", zero_final, actual_zero_live,
                                 args.seed_points,
                                 False, paths.logs / f"build_{seed_prefix.name}.log", paths.logs / "phaseB_zero_insert.log",
                                 seed_pivots=seed_pivots)
    zero_record.update({
        "insert_count": zero_row.get("insert_count"),
        "insert_wall_s": zero_row.get("insert_wall_s"),
        "merge_wall_s": zero_row.get("merge_wall_s"),
        "flat_threshold": points - 1,
        "flat_build_memory_gb": zero_row.get("flat_build_memory_gb"),
        "zero_insert_path": "flat_until_final_materialization",
        "seed_points": args.seed_points,
        "seed_pivot_hash_matches_final": require_file_hash(zero_record["seed_pq_pivots_hash"], "phaseB_seed_pivots")
        == require_file_hash(zero_record["pq_codebook_hash"], "phaseB_zero_final_pivots"),
    })
    validate_pq_record(zero_record, points, require_seed_hash=True)
    append_jsonl(paths.raw / "phaseB_pq_drift.jsonl", zero_record)
    if not zero_record["seed_pivot_hash_matches_final"]:
        update_claim(paths, "P2_PHASEB_1M_PQ_DRIFT", "FAIL",
                     ["raw/phaseB_pq_drift.jsonl", "raw/phaseB_zero_insert.jsonl"],
                     "No-retrain zero-insert final PQ pivots differ from seed pivots; cannot support no-retrain claim.")
        raise RuntimeError("phaseB no-retrain pivot hash changed; refusing to support no-retrain claim")

    direct_selected = calibrate_variant(paths, args, "phaseB", "direct_retrain_1m", 0, direct_prefix,
                                        common["data0"], common["labels"], common["query"], common["tags"], buckets)
    zero_selected = calibrate_variant(paths, args, "phaseB", "zero_insert_no_retrain_1m", 0, zero_final,
                                      common["data0"], common["labels"], common["query"], common["tags"], buckets)
    compare_penalty(paths, "phaseB", 0, direct_selected, zero_selected)
    phaseb_status = scale_status(
        args,
        all_selected_pass=all(row.get("supports_recall_claim") is True for row in direct_selected + zero_selected),
    )
    update_claim(paths, "P2_PHASEB_1M_PQ_DRIFT", phaseb_status,
                 ["raw/phaseB_pq_drift.jsonl", "raw/phaseB_zero_insert.jsonl", "raw/phaseB_selected_route_l.jsonl",
                  "raw/phaseB_penalty.jsonl"],
                 f"{args.npoints}-point direct fresh-retrain vs zero-insert no-retrain evidence; "
                 f"seed_points={args.seed_points}; zero-insert is flat-until-final materialization.")


def phase_c(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path], buckets: list[str]) -> None:
    points = args.npoints
    current_prefix = paths.indexes / f"phaseC_cycle00_no_retrain_pq{args.pq_bytes}"
    build_index(paths, args, current_prefix, common["data0"], common["labels"])
    initial_pivots = Path(str(current_prefix) + "_pq_pivots.bin")
    initial_pivot_hash = aris.file_record(initial_pivots, "phaseC_initial_pq_pivots")
    replacements: dict[int, bytes] = {}

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
        aris.run_command(delete_cmd, cwd=paths.repo, log_path=paths.logs / f"phaseC_cycle{cycle_idx:02d}_delete_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
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
        aris.run_command(insert_cmd, cwd=paths.repo, log_path=paths.logs / f"phaseC_cycle{cycle_idx:02d}_insert_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
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
            == require_file_hash(initial_pivot_hash, "phaseC_initial_pivots")
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
        })
        if not pivot_unchanged:
            update_claim(paths, "P3_PHASEC_CYCLE_DRIFT", "FAIL",
                         ["raw/phaseC_cycle_inventory.jsonl"],
                         f"No-retrain PQ pivots changed in cycle {cycle_idx}; cannot support no-retrain-across-cycles claim.")
            raise RuntimeError(f"phaseC no-retrain pivot hash changed in cycle {cycle_idx}")

        retrain_selected = calibrate_variant(paths, args, "phaseC", "retrain_each_cycle", cycle_idx, retrain_prefix,
                                             live_data, common["labels"], common["query"], common["tags"], buckets)
        no_retrain_selected = calibrate_variant(paths, args, "phaseC", "no_retrain_across_cycles", cycle_idx, current_prefix,
                                                live_data, common["labels"], common["query"], common["tags"], buckets)
        compare_penalty(paths, "phaseC", cycle_idx, retrain_selected, no_retrain_selected)

    selected = read_jsonl(paths.raw / "phaseC_selected_route_l.jsonl")
    all_selected_pass = all(row.get("selected", {}).get("supports_recall_claim") is True for row in selected)
    phasec_status = scale_status(args, needs_cycles=True, all_selected_pass=all_selected_pass)
    update_claim(paths, "P3_PHASEC_CYCLE_DRIFT", phasec_status,
                 ["raw/phaseC_delete_steps.jsonl", "raw/phaseC_no_retrain_cycles.jsonl",
                  "raw/phaseC_cycle_inventory.jsonl", "raw/phaseC_selected_route_l.jsonl",
                  "raw/phaseC_penalty.jsonl"],
                 f"{args.cycles}-cycle {args.npoints}-live evidence over BigANN/SIFT 6M prefix; not full SIFT100M.")


def phase_d(paths: aris.Paths, args: argparse.Namespace, common: dict[str, Path]) -> None:
    data_bin = common["data0"]
    inventory = read_jsonl(paths.raw / "phaseC_cycle_inventory.jsonl")
    if inventory:
        last_live = Path(inventory[-1]["live_data"]["path"])
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
    update_claim(paths, "P4_PQ_RETRAIN_CORE_SWEEP", scale_status(args),
                 ["raw/phaseD_pq_core_sweep.jsonl"],
                 f"Core sweep reports build, PQ train, and recode wall time for representative {args.npoints}-point live set.")


def summarize(paths: aris.Paths) -> None:
    summary: dict[str, Any] = {"created_utc": now_stamp(), "files": {}}
    for rel in [
        "claim_registry.json",
        "evidence/input_inventory.json",
        "evidence/runner_config.json",
        "raw/commands.jsonl",
        "raw/phaseB_pq_drift.jsonl",
        "raw/phaseB_zero_insert.jsonl",
        "raw/phaseB_selected_route_l.jsonl",
        "raw/phaseB_penalty.jsonl",
        "raw/phaseC_delete_steps.jsonl",
        "raw/phaseC_no_retrain_cycles.jsonl",
        "raw/phaseC_cycle_inventory.jsonl",
        "raw/phaseC_selected_route_l.jsonl",
        "raw/phaseC_penalty.jsonl",
        "raw/phaseD_pq_core_sweep.jsonl",
        "raw/selected_route_l.jsonl",
    ]:
        summary["files"][rel] = aris.file_record(paths.out / rel, rel)
    selected = read_jsonl(paths.raw / "selected_route_l.jsonl")
    if selected:
        summary["selected"] = {
            "count": len(selected),
            "pass_count": sum(1 for r in selected if r.get("supports_recall_claim") is True),
            "min_recall": min(float(r.get("recall@10", 0)) for r in selected),
            "max_avg_latency_ms": max(float(r.get("avg_latency_us", 0)) for r in selected) / 1000.0,
            "max_p95_latency_ms": max(float(r.get("p95_latency_us", 0)) for r in selected) / 1000.0,
            "selection_policy": "post_hoc_retuned_fastest_feasible_recall_ge_98",
        }
    for rel in ["phaseB_penalty.jsonl", "phaseC_penalty.jsonl"]:
        rows = read_jsonl(paths.raw / rel)
        if rows:
            summary[rel] = {
                "count": len(rows),
                "unmatched_count": sum(1 for r in rows if r.get("matched_reference_status") != "matched"),
                "max_selected_feasible_delta_ms": max(float(r.get("selected_feasible_delta_ms", 0)) for r in rows),
                "median_selected_feasible_delta_ms": statistics.median(float(r.get("selected_feasible_delta_ms", 0)) for r in rows),
                "max_matched_reference_recall_delta_ms": max(
                    float(r.get("matched_reference_recall_delta_ms", 0))
                    for r in rows
                    if r.get("matched_reference_recall_delta_ms") is not None
                ) if any(r.get("matched_reference_recall_delta_ms") is not None for r in rows) else None,
            }
    write_json(paths.out / "summary.json", summary)


def maybe_plot(paths: aris.Paths) -> None:
    plotter = SCRIPT_DIR / "plot_pq_drift_aris.py"
    if not plotter.exists():
        return
    aris.run_command([sys.executable, str(plotter), str(paths.out)], cwd=paths.repo,
                     log_path=paths.logs / "plot_pq_drift_aris.log", cpu_cap=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/bak3/lzg/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["phaseB", "phaseC", "phaseD", "all"], default="all")
    parser.add_argument("--cpu-cap", type=int, default=16)
    parser.add_argument("--cpu-start", type=int, default=0,
                        help="First logical CPU used by taskset/numactl when --cpu-cap is positive.")
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--binary-root", type=Path, default=None,
                        help="Directory containing PipeANN test binaries; defaults to build/tests under --repo.")
    parser.add_argument("--flat-build-memory-gb", type=int, default=None,
                        help="Build RAM budget for zero-insert flat materialization; defaults to --memory-gb.")
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--query-beamwidth", type=int, default=None,
                        help="Beamwidth used only for measure-dynamic-search; defaults to --beamwidth.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--npoints", type=int, default=1_000_000)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--bigann-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/data/sift_query_1000.bin"))
    parser.add_argument("--base-labels", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels/base_1m.spmat"))
    parser.add_argument("--query-label-dir", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels"))
    parser.add_argument("--sift100m-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--seed", type=int, default=1162026)
    parser.add_argument("--seed-points", type=int, default=100_000)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--delete-fraction", type=float, default=0.60)
    parser.add_argument("--buckets", default=",".join(DEFAULT_BUCKETS))
    parser.add_argument("--selector-types", default="intersect,range")
    parser.add_argument("--l-sweep", type=parse_l_sweep, default=DEFAULT_L_SWEEP,
                        help="Comma-separated search L candidates used during route/L calibration.")
    parser.add_argument("--insert-threads", type=int, default=16)
    parser.add_argument("--merge-threads", type=int, default=16)
    parser.add_argument("--pq-core-sweep", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 4, 8, 16])
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cpu_start < 0:
        raise ValueError("--cpu-start must be non-negative")
    if args.cpu_cap < 0:
        raise ValueError("--cpu-cap must be non-negative")
    cpu_count = os.cpu_count() or 0
    if args.cpu_cap > 0 and cpu_count > 0 and args.cpu_start + args.cpu_cap > cpu_count:
        raise ValueError(f"--cpu-start + --cpu-cap exceeds available CPUs ({cpu_count})")
    aris.CPU_START = args.cpu_start
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
    aris.DEFAULT_L_SWEEP[:] = args.l_sweep
    # Compatibility for imported helpers that expect these argparse attributes.
    args.base_bin = args.bigann_bin
    args.phase3_buckets = args.buckets
    args.phase4_buckets = args.buckets
    args.allow_sift1m_segment_fallback = False
    args.phase4_points = args.npoints
    args.phase4_seed_points = args.seed_points
    args.phase4_flat_threshold = args.npoints - 1
    args.phase4_threshold = args.seed_points
    paths = build_paths(args.repo, args.out_dir)
    install_command_logger(paths)
    if not (paths.out / "claim_registry.json").exists():
        write_claim_registry(paths)
    write_env(paths, args)
    common = prepare_common(paths, args)
    buckets = [bucket for bucket in args.buckets.split(",") if bucket]
    if args.phase in {"phaseB", "all"}:
        phase_b(paths, args, common, buckets)
    if args.phase in {"phaseC", "all"}:
        phase_c(paths, args, common, buckets)
    if args.phase in {"phaseD", "all"}:
        phase_d(paths, args, common)
    summarize(paths)
    if not args.skip_plots:
        maybe_plot(paths)
    print(paths.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
