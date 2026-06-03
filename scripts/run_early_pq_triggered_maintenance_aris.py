#!/usr/bin/env python3
"""Fast-fail ARIS runner for early-PQ triggered maintenance.

This runner consumes an early-PQ-from-10k baseline and turns the already
observed no-retrain failures into a concrete triggered maintenance strategy:

* keep no-retrain evidence as the drift/fail baseline,
* trigger PQ maintenance at the initial 1M boundary and after each large
  replacement cycle when sentinel buckets fail,
* validate the maintained serving snapshot with targeted sentinel buckets
  before expanding to broader reruns.

The script is deliberately incremental.  It writes small JSONL/CSV/MD artifacts
as soon as partial evidence is available and stops early when a strategy row
breaks the recall/latency gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
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


DEFAULT_SENTINEL_BUCKETS = ["u50", "u75", "u100"]
DEFAULT_L_SWEEP = [50, 75, 100, 150, 200, 250, 300, 400, 450, 470, 500]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def selected_payload(row: dict[str, Any]) -> dict[str, Any]:
    selected = row.get("selected")
    return selected if isinstance(selected, dict) else row


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def avg_ms(row: dict[str, Any]) -> float:
    if row.get("avg_latency_ms") not in (None, ""):
        return float(row["avg_latency_ms"])
    return fnum(row, "avg_latency_us") / 1000.0


def p95_ms(row: dict[str, Any]) -> float:
    if row.get("p95_latency_ms") not in (None, ""):
        return float(row["p95_latency_ms"])
    return fnum(row, "p95_latency_us") / 1000.0


def recall(row: dict[str, Any]) -> float:
    return fnum(row, "recall@10", fnum(row, "recall", 0.0))


def case_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    cycle = int(row.get("cycle_idx") or 0)
    variant = str(row.get("variant") or "")
    selector = str(row.get("selector_type") or "")
    bucket = str(row.get("bucket") or "")
    return cycle, variant, selector, bucket


def case_id(cycle_idx: int, variant: str, selector: str, bucket: str) -> str:
    return f"cycle{cycle_idx:02d}_{variant}_{selector}_{bucket}"


def build_paths(repo: Path, out_dir: Path | None) -> aris.Paths:
    if out_dir is not None:
        out = out_dir if out_dir.is_absolute() else repo / out_dir
    else:
        out = repo / "experiments" / f"v100_early_pq_triggered_maintenance_{now_stamp()}"
    if out.exists():
        raise RuntimeError(f"--out-dir already exists; choose a fresh directory to avoid overwriting evidence: {out}")
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


def normalize_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def write_claim_registry(paths: aris.Paths) -> None:
    registry = {
        "created_utc": now_stamp(),
        "claims": [
            {
                "id": "T1_BASELINE_NO_RETRAIN_FAILS_SENTINELS",
                "claim": "The early-10k no-retrain baseline fails the sentinel recall/latency gate.",
                "status": "PENDING",
                "evidence": [],
            },
            {
                "id": "T2_TRIGGERED_RETRAIN_SELECTED_PASS",
                "claim": "Triggered PQ maintenance snapshots pass recall@10 >= 98 and avg latency < 10ms on sentinel buckets.",
                "status": "PENDING",
                "evidence": [],
            },
            {
                "id": "T3_TRIGGER_POLICY_FAST_FAIL",
                "claim": "The runner stops a strategy as soon as a generated sentinel row violates the gate.",
                "status": "PENDING",
                "evidence": [],
            },
            {
                "id": "T4_BACKGROUND_LOW_CORE_PROFILE",
                "claim": "Low-core/background PQ maintenance timing and foreground interference are profiled separately.",
                "status": "PENDING",
                "evidence": [],
            },
            {
                "id": "T5_DYNAMIC_5CYCLE_STRATEGY",
                "claim": "The selected maintenance strategy is validated across at least five 60% delete/insert cycles.",
                "status": "PENDING",
                "evidence": [],
            },
        ],
    }
    write_json(paths.out / "optimized_claim_registry.json", registry)


def update_claim(paths: aris.Paths, claim_id: str, status: str, evidence: list[str], note: str) -> None:
    path = paths.out / "optimized_claim_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    for claim in registry["claims"]:
        if claim["id"] == claim_id:
            claim.update({"status": status, "evidence": evidence, "note": note})
            break
    write_json(path, registry)


def load_selected_maps(baseline: Path) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    selected: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for rel in ["raw/phaseB_selected_route_l.jsonl", "raw/phaseC_selected_route_l.jsonl"]:
        for wrapper in read_jsonl(baseline / rel):
            row = dict(selected_payload(wrapper))
            row.update(
                {
                    "phase": wrapper.get("phase", row.get("phase")),
                    "variant": wrapper.get("variant", row.get("variant")),
                    "cycle_idx": wrapper.get("cycle_idx", row.get("cycle_idx", 0)),
                    "selector_type": wrapper.get("selector_type", row.get("selector_type")),
                    "bucket": wrapper.get("bucket", row.get("bucket")),
                    "baseline_source_rel": rel,
                }
            )
            selected[case_key(row)] = row
    return selected


def load_inventory(repo: Path, baseline: Path) -> dict[int, dict[str, Any]]:
    inventory: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(baseline / "raw" / "phaseC_cycle_inventory.jsonl"):
        cycle = int(row.get("cycle_idx") or 0)
        item = dict(row)
        for key in ["no_retrain_prefix", "retrain_prefix"]:
            if item.get(key):
                item[key] = str(normalize_repo_path(repo, item[key]))
        if item.get("live_data", {}).get("path"):
            item["live_data"]["path"] = str(normalize_repo_path(repo, item["live_data"]["path"]))
        inventory[cycle] = item
    return inventory


def load_common_from_baseline(repo: Path, baseline: Path) -> dict[str, Path]:
    phase0 = json.loads((baseline / "evidence" / "input_inventory.json").read_text(encoding="utf-8"))
    return {
        "source": normalize_repo_path(repo, phase0["bigann_bin"]["path"]),
        "data0": normalize_repo_path(repo, phase0["segment0"]["path"]),
        "query": normalize_repo_path(repo, phase0["query_bin"]["path"]),
        "labels": normalize_repo_path(repo, phase0["labels_1m"]["path"]),
        "tags": normalize_repo_path(repo, phase0["identity_tags"]["path"]),
    }


def baseline_failure_rows(
    selected: dict[tuple[int, str, str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in selected.values():
        variant = row.get("variant")
        if variant not in {"zero_insert_no_retrain_1m", "no_retrain_across_cycles"}:
            continue
        if row.get("bucket") not in args.sentinel_buckets.split(","):
            continue
        if row.get("selector_type") not in args.selector_types.split(","):
            continue
        reasons = gate_reasons(row, args)
        if reasons:
            item = strategy_row(row, row, "no_retrain_baseline", "baseline_failed", reasons)
            failures.append(item)
    return sorted(failures, key=lambda r: (int(r["cycle_idx"]), r["selector_type"], r["bucket"]))


def gate_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if recall(row) < args.recall_floor:
        reasons.append(f"recall<{args.recall_floor:g}")
    if avg_ms(row) >= args.latency_ms:
        reasons.append(f"avg>={args.latency_ms:g}ms")
    if args.p95_latency_ms > 0 and p95_ms(row) >= args.p95_latency_ms:
        reasons.append(f"p95>={args.p95_latency_ms:g}ms")
    return reasons


def strategy_row(
    source: dict[str, Any],
    selected: dict[str, Any],
    strategy_variant: str,
    trigger_status: str,
    trigger_reasons: list[str],
) -> dict[str, Any]:
    cycle_idx = int(source.get("cycle_idx") or selected.get("cycle_idx") or 0)
    selector = str(source.get("selector_type") or selected.get("selector_type"))
    bucket = str(source.get("bucket") or selected.get("bucket"))
    row = dict(selected)
    row.update(
        {
            "case_id": case_id(cycle_idx, strategy_variant, selector, bucket),
            "cycle_idx": cycle_idx,
            "selector_type": selector,
            "bucket": bucket,
            "strategy_variant": strategy_variant,
            "trigger_status": trigger_status,
            "trigger_reasons": ",".join(trigger_reasons),
            "avg_latency_ms": avg_ms(selected),
            "p95_latency_ms": p95_ms(selected),
            "recall": recall(selected),
            "route": selected.get("route") or selected.get("actual_route"),
            "search_l": selected.get("search_l") or selected.get("chosen_L") or selected.get("configured_L"),
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    reasons = [gate_reasons(row, args) for row in rows]
    return {
        "count": len(rows),
        "recall_pass_count": sum(recall(row) >= args.recall_floor for row in rows),
        "avg_lt_10ms_count": sum(avg_ms(row) < args.latency_ms for row in rows),
        "p95_lt_limit_count": sum(p95_ms(row) < args.p95_latency_ms for row in rows) if args.p95_latency_ms > 0 else None,
        "min_recall": min((recall(row) for row in rows), default=0.0),
        "max_avg_latency_ms": max((avg_ms(row) for row in rows), default=0.0),
        "max_p95_latency_ms": max((p95_ms(row) for row in rows), default=0.0),
        "fail_count": sum(1 for item in reasons if item),
    }


def append_artifacts(paths: aris.Paths, name: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(paths.out / f"{name}.jsonl", rows)
    write_csv(paths.out / f"{name}.csv", rows)


def write_strategy_doc(paths: aris.Paths, args: argparse.Namespace, baseline_summary: dict[str, Any],
                       strategy_summary: dict[str, Any]) -> None:
    text = f"""# PQ Maintenance Strategy

## Decision

Use triggered PQ maintenance instead of early-10k no-retrain serving.  The
initial trigger fires after the 1M online insert completes because the stopped
baseline already shows sentinel latency above {args.latency_ms:g} ms.  Later
cycles trigger after each 60% delete/new-vector insert if any sentinel row
violates recall@10 >= {args.recall_floor:g} or avg latency < {args.latency_ms:g} ms.

## Fast-Fail Scope

- Sentinel buckets: `{args.sentinel_buckets}`.
- Selector types: `{args.selector_types}`.
- Existing baseline is reused when the matching retrain snapshot already has a
  reviewed selected row.
- Missing sentinel rows are reported as incomplete instead of being inferred.
- Any generated strategy row that violates the gate stops the current strategy.

## Current Evidence Summary

- Baseline no-retrain sentinel failures: `{baseline_summary}`.
- Triggered retrain sentinel summary: `{strategy_summary}`.

## Engineering Interpretation

The accepted foreground serving point is the retrained snapshot after the
trigger has completed.  Background/low-core build interference is measured as a
separate artifact so the blocking and non-blocking maintenance costs are not
mixed with selected query latency.
"""
    (paths.out / "pq_maintenance_strategy.md").write_text(text, encoding="utf-8")


def ensure_calibration_args(args: argparse.Namespace) -> None:
    aris.CPU_START = args.cpu_start
    aris.DEFAULT_L_SWEEP[:] = args.l_sweep
    args.base_bin = args.bigann_bin
    args.phase3_buckets = args.sentinel_buckets
    args.phase4_buckets = args.sentinel_buckets
    args.allow_sift1m_segment_fallback = False
    args.phase4_points = args.npoints
    args.phase4_seed_points = args.seed_points
    args.phase4_flat_threshold = args.npoints - 1
    args.phase4_threshold = args.seed_points
    if args.query_beamwidth is None:
        args.query_beamwidth = args.beamwidth


def synthesize(paths: aris.Paths, args: argparse.Namespace) -> dict[str, Any]:
    baseline = normalize_repo_path(paths.repo, args.baseline_dir)
    selected = load_selected_maps(baseline)
    inventory = load_inventory(paths.repo, baseline)
    failures = baseline_failure_rows(selected, args)
    append_artifacts(paths, "targeted_latency_profile", failures)
    baseline_summary = summarize_rows(failures, args)
    update_claim(
        paths,
        "T1_BASELINE_NO_RETRAIN_FAILS_SENTINELS",
        "PASS" if failures else "FAIL",
        ["targeted_latency_profile.jsonl", str(baseline / "raw" / "phaseC_selected_route_l.jsonl")],
        f"Baseline sentinel failure summary: {baseline_summary}",
    )
    update_claim(
        paths,
        "T3_TRIGGER_POLICY_FAST_FAIL",
        "PENDING",
        ["pq_maintenance_strategy.md"],
        "Fast-fail guard is defined but not exercised until targeted strategy rows are generated.",
    )
    return {"selected": selected, "inventory": inventory, "failures": failures, "baseline_summary": baseline_summary}


def targeted(paths: aris.Paths, args: argparse.Namespace) -> None:
    state = synthesize(paths, args)
    baseline = normalize_repo_path(paths.repo, args.baseline_dir)
    selected = state["selected"]
    strategy_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    cycles = cycles_to_check(args, selected)
    buckets = [item for item in args.sentinel_buckets.split(",") if item]
    selectors = [item for item in args.selector_types.split(",") if item]

    for cycle_idx in cycles:
        for selector in selectors:
            for bucket in buckets:
                no_key = (cycle_idx, "no_retrain_across_cycles", selector, bucket)
                retrain_key = (cycle_idx, "retrain_each_cycle", selector, bucket)
                no_row = selected.get(no_key)
                if no_row is None and cycle_idx == 0:
                    no_row = selected.get((0, "zero_insert_no_retrain_1m", selector, bucket))
                    retrain_key = (0, "direct_retrain_1m", selector, bucket)
                retrain_row = selected.get(retrain_key)
                if no_row is None or retrain_row is None:
                    compare_rows.append(
                        {
                            "cycle_idx": cycle_idx,
                            "selector_type": selector,
                            "bucket": bucket,
                            "strategy_variant": "triggered_retrain",
                            "status": "missing_evidence",
                            "has_no_retrain": no_row is not None,
                            "has_retrain": retrain_row is not None,
                        }
                    )
                    continue
                trigger_reasons = gate_reasons(no_row, args) or ["replacement_ratio>=0.60"]
                maintained = strategy_row(no_row, retrain_row, "triggered_retrain", "triggered", trigger_reasons)
                maintained["serving_source"] = "retrain_each_cycle_snapshot"
                maintained["baseline_no_retrain_avg_latency_ms"] = avg_ms(no_row)
                maintained["baseline_no_retrain_p95_latency_ms"] = p95_ms(no_row)
                maintained["baseline_no_retrain_recall"] = recall(no_row)
                strategy_rows.append(maintained)
                compare_rows.append(
                    {
                        "cycle_idx": cycle_idx,
                        "selector_type": selector,
                        "bucket": bucket,
                        "trigger_reasons": ",".join(trigger_reasons),
                        "no_retrain_recall": recall(no_row),
                        "no_retrain_avg_latency_ms": avg_ms(no_row),
                        "no_retrain_p95_latency_ms": p95_ms(no_row),
                        "triggered_retrain_recall": recall(retrain_row),
                        "triggered_retrain_avg_latency_ms": avg_ms(retrain_row),
                        "triggered_retrain_p95_latency_ms": p95_ms(retrain_row),
                        "triggered_retrain_route": retrain_row.get("route") or retrain_row.get("actual_route"),
                        "triggered_retrain_L": retrain_row.get("search_l") or retrain_row.get("chosen_L"),
                        "status": "ok",
                    }
                )
                reasons = gate_reasons(maintained, args)
                if reasons:
                    write_fast_fail(paths, args, maintained, reasons, strategy_rows, compare_rows)
                    raise RuntimeError(f"strategy fast-fail at {maintained['case_id']}: {reasons}")

    append_artifacts(paths, "optimized_dynamic_update_results", strategy_rows)
    append_artifacts(paths, "pq_drift_strategy_compare", compare_rows)
    append_artifacts(paths, "pq_retrain_interference_profile", [])
    strategy_summary = summarize_rows(strategy_rows, args)
    write_strategy_doc(paths, args, state["baseline_summary"], strategy_summary)
    write_json(
        paths.out / "summary.json",
        {
            "created_utc": now_stamp(),
            "phase": "targeted",
            "baseline_dir": str(baseline),
            "cycles_checked": cycles,
            "sentinel_buckets": buckets,
            "selector_types": selectors,
            "baseline_summary": state["baseline_summary"],
            "strategy_summary": strategy_summary,
            "missing_evidence_rows": sum(1 for row in compare_rows if row.get("status") == "missing_evidence"),
            "fast_fail": False,
        },
    )
    missing_count = sum(1 for row in compare_rows if row.get("status") == "missing_evidence")
    expected_count = len(cycles) * len(selectors) * len(buckets)
    status = (
        "PASS"
        if strategy_rows
        and len(strategy_rows) == expected_count
        and missing_count == 0
        and strategy_summary["fail_count"] == 0
        else "INCOMPLETE"
    )
    update_claim(
        paths,
        "T2_TRIGGERED_RETRAIN_SELECTED_PASS",
        status,
        ["optimized_dynamic_update_results.jsonl", "pq_drift_strategy_compare.jsonl"],
        f"Triggered retrain sentinel summary: {strategy_summary}",
    )
    update_claim(
        paths,
        "T4_BACKGROUND_LOW_CORE_PROFILE",
        "PENDING",
        ["pq_retrain_interference_profile.jsonl"],
        "Background interference phase not run yet.",
    )
    update_claim(
        paths,
        "T3_TRIGGER_POLICY_FAST_FAIL",
        "READY" if status == "PASS" else "PENDING",
        ["optimized_dynamic_update_results.jsonl", "summary.json"],
        f"Fast-fail guard evaluated {len(strategy_rows)} generated strategy rows without triggering; missing_evidence_rows={missing_count}.",
    )
    update_claim(
        paths,
        "T5_DYNAMIC_5CYCLE_STRATEGY",
        "INCOMPLETE",
        ["summary.json"],
        "Targeted sentinel pass precedes broader 5-cycle strategy rerun.",
    )


def latest_or_fail(jsonl: Path, before: int, fields: list[str]) -> dict[str, Any]:
    return aris.latest_driver_row(jsonl, before, fields)


def build_index_with_cpu(paths: aris.Paths, args: argparse.Namespace, prefix: Path, data_bin: Path, labels: Path,
                         cpu_cap: int) -> dict[str, Any]:
    local_args = copy.copy(args)
    local_args.cpu_cap = cpu_cap
    started = time.time()
    aris.build_or_reuse_index(paths, local_args, prefix, data_bin, labels)
    elapsed = time.time() - started
    log_path = paths.logs / f"build_{prefix.name}.log"
    return {
        "prefix": str(prefix),
        "build_wall_s": elapsed,
        "cpu_cap": cpu_cap,
        "pq_train_wall_s": aris.extract_log_seconds(log_path, r"Pivots generated in ([0-9.]+)s"),
        "pq_recode_wall_s": aris.extract_log_seconds(log_path, r"Compressed data written in: ([0-9.]+)s"),
        "pq_training_points": aris.extract_log_int(log_path, r"Generating PQ pivots with training data of size: ([0-9]+)"),
        "pq_codebook_hash": aris.file_record(Path(str(prefix) + "_pq_pivots.bin"), f"{prefix.name}_pq_pivots"),
        "pq_code_hash": aris.file_record(Path(str(prefix) + "_pq_compressed.bin"), f"{prefix.name}_pq_codes"),
        "log_path": str(log_path),
    }


def calibrate_one(paths: aris.Paths, args: argparse.Namespace, prefix: Path, live_data: Path, labels: Path,
                  query: Path, tags: Path, selector: str, bucket: str, cycle_name: str) -> dict[str, Any]:
    qlabel = aris.query_label_path(paths, args, selector, bucket)
    truth = paths.truth / f"{cycle_name}_{selector}_{bucket}.bin"
    aris.compute_truth(paths, args, live_data, query, labels, qlabel, selector, truth, tag_file=tags)
    selected = aris.calibrate_bucket(paths, args, prefix, live_data, labels, query, qlabel, truth,
                                     selector, bucket, cycle_name)
    return selected


def load_initial_maintained_prefix(repo: Path, baseline: Path) -> Path:
    summary_path = baseline / "early_pq_train_10k_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("direct_prefix"):
            return normalize_repo_path(repo, summary["direct_prefix"])
    fallback = baseline / "indexes" / "phaseB_direct_retrain_1m_pq16"
    return normalize_repo_path(repo, fallback)


def append_chain_csvs(paths: aris.Paths) -> None:
    for stem in [
        "optimized_dynamic_update_results",
        "pq_drift_strategy_compare",
        "pq_retrain_interference_profile",
        "early_pq_delete_insert_maintenance",
    ]:
        rows = read_jsonl(paths.out / f"{stem}.jsonl")
        write_csv(paths.out / f"{stem}.csv", rows)


def expected_chain_selected_rows(args: argparse.Namespace) -> int:
    selectors = [item for item in args.selector_types.split(",") if item]
    buckets = [item for item in args.sentinel_buckets.split(",") if item]
    return args.chain_cycles * len(selectors) * len(buckets)


def chain_summary(paths: aris.Paths, args: argparse.Namespace, rows: list[dict[str, Any]],
                  maintenance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = expected_chain_selected_rows(args)
    return {
        "created_utc": now_stamp(),
        "phase": "chain",
        "chain_cycles": args.chain_cycles,
        "sentinel_buckets": [item for item in args.sentinel_buckets.split(",") if item],
        "selector_types": [item for item in args.selector_types.split(",") if item],
        "expected_selected_rows": expected_rows,
        "observed_selected_rows": len(rows),
        "strategy_summary": summarize_rows(rows, args),
        "maintenance_rows": len(maintenance_rows),
        "max_delete_ms_per_vector": max((fnum(row, "delete_ms_per_vector") for row in maintenance_rows), default=0.0),
        "max_delete_merge_s": max((fnum(row, "delete_merge_s") for row in maintenance_rows), default=0.0),
        "max_insert_merge_s": max((fnum(row, "insert_merge_s") for row in maintenance_rows), default=0.0),
        "max_maintenance_build_wall_s": max((fnum(row, "maintenance_build_wall_s") for row in maintenance_rows), default=0.0),
        "max_pq_train_wall_s": max((fnum(row, "pq_train_wall_s") for row in maintenance_rows), default=0.0),
        "max_pq_recode_wall_s": max((fnum(row, "pq_recode_wall_s") for row in maintenance_rows), default=0.0),
    }


def run_chain(paths: aris.Paths, args: argparse.Namespace) -> None:
    baseline = normalize_repo_path(paths.repo, args.baseline_dir)
    common = load_common_from_baseline(paths.repo, baseline)
    current_prefix = load_initial_maintained_prefix(paths.repo, baseline)
    selectors = [item for item in args.selector_types.split(",") if item]
    buckets = [item for item in args.sentinel_buckets.split(",") if item]
    replacements: dict[int, bytes] = {}
    selected_rows: list[dict[str, Any]] = []
    maintenance_rows: list[dict[str, Any]] = []

    write_strategy_doc(paths, args, {"source": "chain_phase"}, {})
    update_claim(
        paths,
        "T3_TRIGGER_POLICY_FAST_FAIL",
        "READY",
        ["pq_maintenance_strategy.md"],
        "Chain phase writes each selected row immediately and stops on the first recall/latency violation.",
    )

    for cycle_idx in range(1, args.chain_cycles + 1):
        cycle_started = time.time()
        delete_ids = paths.data / f"chain_cycle{cycle_idx:02d}_delete_ids_60pct.txt"
        delete_count = aris.make_delete_ids(delete_ids, args.npoints, args.delete_fraction, args.seed + cycle_idx)
        after_delete = paths.indexes / f"chain_cycle{cycle_idx:02d}_after_delete_merge"
        delete_jsonl = paths.raw / "chain_delete_steps.jsonl"
        before = len(read_jsonl(delete_jsonl))
        delete_cmd = aris.driver_base_cmd(paths, args, "delete-batch", current_prefix, delete_jsonl)
        delete_cmd += [
            "--dest-prefix", str(after_delete),
            "--delete-id-file", str(delete_ids),
            "--delete-count", str(delete_count),
            "--data-bin", str(common["data0"]),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(delete_cmd, cwd=paths.repo, log_path=paths.logs / f"chain_cycle{cycle_idx:02d}_delete_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        delete_row = latest_or_fail(delete_jsonl, before, [
            "mode", "status", "source_prefix", "dest_prefix", "delete_count", "deleted_tag_hash",
            "delete_scope", "delete_elapsed_s", "merge_elapsed_s", "live_point_count", "raw_command",
        ])

        insert_segment = pq1m.segment_bin(paths, common["source"], cycle_idx * args.npoints, delete_count,
                                          f"chain_cycle{cycle_idx:02d}_insert_vectors.bin")
        after_insert = paths.indexes / f"chain_cycle{cycle_idx:02d}_after_insert_pre_retrain"
        insert_jsonl = paths.raw / "chain_insert_steps.jsonl"
        before = len(read_jsonl(insert_jsonl))
        insert_cmd = aris.driver_base_cmd(paths, args, "insert-only", after_delete, insert_jsonl)
        insert_cmd += [
            "--dest-prefix", str(after_insert),
            "--data-bin", str(insert_segment),
            "--insert-start", "0",
            "--insert-count", str(delete_count),
            "--insert-tag-file", str(delete_ids),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(insert_cmd, cwd=paths.repo, log_path=paths.logs / f"chain_cycle{cycle_idx:02d}_insert_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        insert_row = latest_or_fail(insert_jsonl, before, [
            "mode", "status", "source_prefix", "dest_prefix", "insert_count", "insert_scope",
            "inserted_tag_hash", "insert_elapsed_s", "merge_elapsed_s", "live_point_count", "raw_command",
        ])

        deleted_tags = [int(line) for line in delete_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        replacements.update(aris.load_segment_replacements(insert_segment, deleted_tags))
        live_data = paths.data / f"chain_cycle{cycle_idx:02d}_live_data_by_tag.bin"
        aris.materialize_live_data(common["data0"], replacements, live_data, args.npoints)

        maintained_prefix = paths.indexes / f"chain_cycle{cycle_idx:02d}_triggered_retrain"
        build_start = {
            "event": "triggered_retrain_build_start",
            "cycle_idx": cycle_idx,
            "created_utc": now_stamp(),
            "maintenance_prefix": str(maintained_prefix),
            "live_data": str(live_data),
            "maintenance_cpu_cap": args.maintenance_cpu_cap,
            "trigger_policy": args.trigger_policy,
        }
        append_jsonl(paths.raw / "chain_build_events.jsonl", build_start)
        write_json(paths.out / "summary.json", {
            "phase": "chain",
            "status": "build_in_progress",
            "last_started_cycle": cycle_idx,
            "last_build_event": build_start,
            "completed_selected_rows": len(selected_rows),
            "completed_maintenance_rows": len(maintenance_rows),
        })
        build_row = build_index_with_cpu(paths, args, maintained_prefix, live_data, common["labels"],
                                         args.maintenance_cpu_cap)
        append_jsonl(paths.raw / "chain_build_events.jsonl", {
            "event": "triggered_retrain_build_complete",
            "cycle_idx": cycle_idx,
            "created_utc": now_stamp(),
            **build_row,
        })
        current_prefix = maintained_prefix

        maintenance = {
            "cycle_idx": cycle_idx,
            "trigger_policy": args.trigger_policy,
            "delete_count": delete_count,
            "delete_ms_per_vector": fnum(delete_row, "delete_elapsed_s") * 1000.0 / max(delete_count, 1),
            "delete_elapsed_s": delete_row.get("delete_elapsed_s"),
            "delete_merge_s": delete_row.get("merge_elapsed_s"),
            "insert_count": insert_row.get("insert_count"),
            "insert_elapsed_s": insert_row.get("insert_elapsed_s"),
            "insert_merge_s": insert_row.get("merge_elapsed_s"),
            "maintenance_prefix": str(maintained_prefix),
            "maintenance_build_wall_s": build_row.get("build_wall_s"),
            "pq_train_wall_s": build_row.get("pq_train_wall_s"),
            "pq_recode_wall_s": build_row.get("pq_recode_wall_s"),
            "maintenance_cpu_cap": args.maintenance_cpu_cap,
            "cycle_elapsed_before_search_s": time.time() - cycle_started,
        }
        maintenance_rows.append(maintenance)
        append_jsonl(paths.raw / "chain_maintenance.jsonl", maintenance)
        append_jsonl(paths.out / "early_pq_delete_insert_maintenance.jsonl", maintenance)
        append_jsonl(paths.out / "pq_retrain_interference_profile.jsonl", {
            "cycle_idx": cycle_idx,
            "maintenance_kind": "triggered_full_retrain",
            "maintenance_cpu_cap": args.maintenance_cpu_cap,
            "maintenance_build_wall_s": build_row.get("build_wall_s"),
            "pq_train_wall_s": build_row.get("pq_train_wall_s"),
            "pq_recode_wall_s": build_row.get("pq_recode_wall_s"),
            "foreground_interference_status": "not_overlapped_in_chain_phase",
        })

        cycle_selected: list[dict[str, Any]] = []
        for selector in selectors:
            for bucket in buckets:
                cycle_name = f"chain_cycle{cycle_idx:02d}_triggered_retrain"
                selected = calibrate_one(paths, args, maintained_prefix, live_data, common["labels"],
                                         common["query"], common["tags"], selector, bucket, cycle_name)
                selected.update({
                    "phase": "chain",
                    "variant": "triggered_retrain_chain",
                    "strategy_variant": "triggered_retrain_chain",
                    "cycle_idx": cycle_idx,
                    "selector_type": selector,
                    "bucket": bucket,
                    "case_id": case_id(cycle_idx, "triggered_retrain_chain", selector, bucket),
                    "trigger_status": "triggered",
                    "trigger_reasons": args.trigger_policy,
                    "serving_source": "actual_triggered_chain_snapshot",
                    "avg_latency_ms": avg_ms(selected),
                    "p95_latency_ms": p95_ms(selected),
                    "recall": recall(selected),
                    "route": selected.get("route") or selected.get("actual_route"),
                    "search_l": selected.get("search_l") or selected.get("chosen_L") or selected.get("configured_L"),
                    "maintenance_prefix": str(maintained_prefix),
                })
                append_jsonl(paths.raw / "chain_selected_route_l.jsonl", selected)
                append_jsonl(paths.out / "optimized_dynamic_update_results.jsonl", selected)
                append_jsonl(paths.out / "pq_drift_strategy_compare.jsonl", {
                    "cycle_idx": cycle_idx,
                    "selector_type": selector,
                    "bucket": bucket,
                    "strategy_variant": "triggered_retrain_chain",
                    "triggered_retrain_recall": recall(selected),
                    "triggered_retrain_avg_latency_ms": avg_ms(selected),
                    "triggered_retrain_p95_latency_ms": p95_ms(selected),
                    "triggered_retrain_route": selected.get("route"),
                    "triggered_retrain_L": selected.get("search_l"),
                    "status": "ok",
                })
                selected_rows.append(selected)
                cycle_selected.append(selected)
                reasons = gate_reasons(selected, args)
                if reasons:
                    write_fast_fail(paths, args, selected, reasons, selected_rows,
                                    read_jsonl(paths.out / "pq_drift_strategy_compare.jsonl"))
                    append_chain_csvs(paths)
                    write_json(paths.out / "summary.json", chain_summary(paths, args, selected_rows, maintenance_rows))
                    raise RuntimeError(f"chain fast-fail at {selected['case_id']}: {reasons}")
        cycle_summary = chain_summary(paths, args, selected_rows, maintenance_rows)
        cycle_summary["last_completed_cycle"] = cycle_idx
        cycle_summary["last_cycle_summary"] = summarize_rows(cycle_selected, args)
        append_jsonl(paths.raw / "chain_cycle_summary.jsonl", cycle_summary)
        write_json(paths.out / "summary.json", cycle_summary)
        append_chain_csvs(paths)

    final = chain_summary(paths, args, selected_rows, maintenance_rows)
    final["completed_cycles"] = args.chain_cycles
    write_json(paths.out / "summary.json", final)
    append_chain_csvs(paths)
    status = (
        "PASS"
        if args.chain_cycles >= 5
        and final["observed_selected_rows"] >= final["expected_selected_rows"]
        and final["strategy_summary"]["fail_count"] == 0
        else "INCOMPLETE"
    )
    update_claim(
        paths,
        "T2_TRIGGERED_RETRAIN_SELECTED_PASS",
        status,
        ["optimized_dynamic_update_results.jsonl", "pq_drift_strategy_compare.jsonl"],
        f"Triggered chain selected summary: {final['strategy_summary']}",
    )
    update_claim(
        paths,
        "T4_BACKGROUND_LOW_CORE_PROFILE",
        "EVIDENCE_CHAIN_TIMING",
        ["pq_retrain_interference_profile.jsonl", "early_pq_delete_insert_maintenance.jsonl"],
        "Chain phase records PQ train/recode/build times; foreground-overlap evidence remains prior/independent.",
    )
    update_claim(
        paths,
        "T5_DYNAMIC_5CYCLE_STRATEGY",
        status,
        ["summary.json", "optimized_dynamic_update_results.jsonl", "early_pq_delete_insert_maintenance.jsonl"],
        (
            f"Triggered chain completed {args.chain_cycles} cycles with "
            f"{final['observed_selected_rows']}/{final['expected_selected_rows']} selected rows."
        ),
    )


def write_fast_fail(paths: aris.Paths, args: argparse.Namespace, row: dict[str, Any], reasons: list[str],
                    strategy_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]]) -> None:
    append_artifacts(paths, "optimized_dynamic_update_results", strategy_rows)
    append_artifacts(paths, "pq_drift_strategy_compare", compare_rows)
    payload = {
        "created_utc": now_stamp(),
        "fast_fail": True,
        "case_id": row.get("case_id"),
        "reasons": reasons,
        "row": row,
        "latency_ms_limit": args.latency_ms,
        "recall_floor": args.recall_floor,
    }
    write_json(paths.out / "fast_fail_status.json", payload)
    update_claim(
        paths,
        "T2_TRIGGERED_RETRAIN_SELECTED_PASS",
        "FAIL",
        ["fast_fail_status.json", "optimized_dynamic_update_results.jsonl"],
        f"Triggered retrain strategy failed early: {payload}",
    )
    update_claim(
        paths,
        "T3_TRIGGER_POLICY_FAST_FAIL",
        "PASS",
        ["fast_fail_status.json", "optimized_dynamic_update_results.jsonl"],
        f"Fast-fail guard stopped strategy at {row.get('case_id')}: {reasons}",
    )


def has_complete_cycle(selected: dict[tuple[int, str, str, str], dict[str, Any]], cycle: int,
                       selectors: list[str], buckets: list[str]) -> bool:
    if cycle == 0:
        no_variant = "zero_insert_no_retrain_1m"
        retrain_variant = "direct_retrain_1m"
    else:
        no_variant = "no_retrain_across_cycles"
        retrain_variant = "retrain_each_cycle"
    for selector in selectors:
        for bucket in buckets:
            if (cycle, no_variant, selector, bucket) not in selected:
                return False
            if (cycle, retrain_variant, selector, bucket) not in selected:
                return False
    return True


def cycles_to_check(args: argparse.Namespace, selected: dict[tuple[int, str, str, str], dict[str, Any]]) -> list[int]:
    if args.cycles_to_check:
        return [int(item) for item in args.cycles_to_check.split(",") if item]
    selectors = [item for item in args.selector_types.split(",") if item]
    buckets = [item for item in args.sentinel_buckets.split(",") if item]
    existing: list[int] = []
    if has_complete_cycle(selected, 0, selectors, buckets):
        existing.append(0)
    dynamic_cycles = sorted({cycle for cycle, _variant, _selector, _bucket in selected if cycle > 0})
    for cycle in dynamic_cycles:
        if has_complete_cycle(selected, cycle, selectors, buckets):
            existing.append(cycle)
    if args.max_cycles > 0:
        existing = existing[: args.max_cycles]
    return existing


def parse_l_sweep(value: str) -> list[int]:
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError("empty L sweep")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiments/v100_early_pq_10k_full_20260602T031600Z"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["synthesize", "targeted", "chain"], default="targeted")
    parser.add_argument("--binary-root", type=Path, default=Path("build_reviewed_20260602_querycache/tests"))
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--cpu-cap", type=int, default=16)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--insert-threads", type=int, default=16)
    parser.add_argument("--merge-threads", type=int, default=16)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--query-beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--npoints", type=int, default=1_000_000)
    parser.add_argument("--seed-points", type=int, default=100_000)
    parser.add_argument("--bigann-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--query-label-dir", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels"))
    parser.add_argument("--seed", type=int, default=1162026)
    parser.add_argument("--delete-fraction", type=float, default=0.60)
    parser.add_argument("--sentinel-buckets", default=",".join(DEFAULT_SENTINEL_BUCKETS))
    parser.add_argument("--selector-types", default="intersect,range")
    parser.add_argument("--cycles-to-check", default="")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--chain-cycles", type=int, default=5)
    parser.add_argument("--maintenance-cpu-cap", type=int, default=16)
    parser.add_argument("--trigger-policy", default="always_after_initial_1m_and_each_60pct_replacement")
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--p95-latency-ms", type=float, default=0.0)
    parser.add_argument("--recall-floor", type=float, default=98.0)
    parser.add_argument("--l-sweep", type=parse_l_sweep, default=DEFAULT_L_SWEEP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    if args.cpu_start < 0 or args.cpu_cap <= 0:
        raise ValueError("CPU start/cap must be positive")
    if args.maintenance_cpu_cap <= 0:
        raise ValueError("--maintenance-cpu-cap must be positive")
    cpu_count = os.cpu_count() or 0
    if cpu_count and args.cpu_start + args.cpu_cap > cpu_count:
        raise ValueError(f"CPU range exceeds available CPU count {cpu_count}")
    if args.binary_root and not args.binary_root.is_absolute():
        args.binary_root = args.repo / args.binary_root
    ensure_calibration_args(args)
    paths = build_paths(args.repo, args.out_dir)
    if not (paths.out / "optimized_claim_registry.json").exists():
        write_claim_registry(paths)
    write_json(
        paths.evidence / "runner_config.json",
        {
            "created_utc": now_stamp(),
            "script": str(Path(__file__).resolve()),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    if args.phase == "synthesize":
        state = synthesize(paths, args)
        write_strategy_doc(paths, args, state["baseline_summary"], {})
        write_json(paths.out / "summary.json", {"phase": "synthesize", "baseline_summary": state["baseline_summary"]})
    elif args.phase == "targeted":
        targeted(paths, args)
    elif args.phase == "chain":
        run_chain(paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
