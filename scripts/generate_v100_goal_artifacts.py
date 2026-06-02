#!/usr/bin/env python3
"""Generate final v100 PipeANN goal artifacts from reviewed evidence.

This script is intentionally post-processing only. It reads already-produced
JSONL/CSV evidence from the v100 baseline, Supersector32K replay, explicit
materialization run, and background-maintenance interference run, then writes
small report artifacts and PPT-ready figures. It never builds or repacks an
index and it never writes large binary files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DPI = 220
BUCKETS = ["u1e-03", "u3e-03", "u1e-02", "u5e-02", "u1e-01", "u25", "u30", "u50", "u75", "u100"]
VARIANTS = ["retrain_each_cycle", "no_retrain_across_cycles"]
SELECTORS = ["intersect", "range"]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def payload(row: dict[str, Any]) -> dict[str, Any]:
    selected = row.get("selected")
    return selected if isinstance(selected, dict) else row


def dynamic_case_id(row: dict[str, Any], *, variant: str | None = None) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    cycle = parse_cycle_idx(row.get("cycle_idx") or row.get("cycle"))
    row_variant = variant or str(row.get("variant") or "")
    selector = str(row.get("selector_type") or "")
    bucket = str(row.get("bucket") or "")
    if not cycle or not row_variant or not selector or not bucket:
        return ""
    return f"cycle{cycle:02d}_{row_variant}_{selector}_{bucket}"


def canonical_dynamic_tuple(row: dict[str, Any]) -> tuple[int, str, str, str] | None:
    cycle = parse_cycle_idx(row.get("cycle_idx") or row.get("cycle"))
    variant = str(row.get("variant") or "")
    selector = str(row.get("selector_type") or "")
    bucket = str(row.get("bucket") or "")
    case_id = str(row.get("case_id") or "")
    match = re.match(r"^cycle0*([0-9]+)_(retrain_each_cycle|no_retrain_across_cycles)_(intersect|range)_(.+)$", case_id)
    if match:
        cycle = int(match.group(1))
        variant = variant or match.group(2)
        selector = selector or match.group(3)
        bucket = bucket or match.group(4)
    if not cycle or not variant or not selector or not bucket:
        return None
    return (cycle, variant, selector, bucket)


def expected_dynamic_tuples(cycles: int) -> set[tuple[int, str, str, str]]:
    return {
        (cycle, variant, selector, bucket)
        for cycle in range(1, cycles + 1)
        for variant in VARIANTS
        for selector in SELECTORS
        for bucket in BUCKETS
    }


def parse_cycle_idx(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"cycle0*([0-9]+)", str(value))
        return int(match.group(1)) if match else 0


def command_query_limit(row: dict[str, Any]) -> float | None:
    for key in ["query_count", "queries", "query_limit"]:
        value = maybe_float(row.get(key))
        if value is not None:
            return value
    command = str(row.get("raw_command") or row.get("replay_invocation") or "")
    match = re.search(r"--query-limit\s+([0-9]+)", command)
    if match:
        return float(match.group(1))
    return None


def command_beamwidth(row: dict[str, Any]) -> int | None:
    for key in ["query_beamwidth", "beamwidth", "selected_beamwidth"]:
        value = maybe_float(row.get(key))
        if value is not None:
            return int(value)
    command = str(row.get("raw_command") or row.get("replay_invocation") or "")
    matches = re.findall(r"--beamwidth\s+([0-9]+)", command)
    if matches:
        return int(matches[-1])
    return None


def copy_small_artifact(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def selected_v3_prefixes(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("v3_source_prefix") or row.get("source_prefix") or "")
        for row in rows
        if row.get("v3_source_prefix") or row.get("source_prefix")
    }


def selected_case_prefix_pairs(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        case_id = dynamic_case_id(row)
        prefix = str(row.get("v3_source_prefix") or row.get("source_prefix") or "")
        if case_id and prefix:
            pairs.add((case_id, prefix))
    return pairs


def repack_v3_prefixes(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row.get("dst_prefix") or "") for row in rows if row.get("dst_prefix")}


def space_v3_prefixes(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row.get("v3_source_prefix") or "") for row in rows if row.get("v3_source_prefix")}


def load_selected(v3_replay_dir: Path) -> list[dict[str, Any]]:
    rows = [payload(row) for row in read_jsonl(v3_replay_dir / "optimized_dynamic_update_results.jsonl")]
    for row in rows:
        case_id = dynamic_case_id(row)
        if case_id:
            row.setdefault("case_id", case_id)
    return rows

def annotate_serving_rows(rows: list[dict[str, Any]], selected_by_row_index: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        selected_row: dict[str, Any] = {}
        if selected_by_row_index is not None:
            row_index = item.get("row_index")
            if row_index not in (None, ""):
                selected_row = selected_by_row_index.get(int(row_index), {})
        beamwidth = command_beamwidth(item)
        selected_beamwidth = command_beamwidth(selected_row)
        if selected_beamwidth is not None:
            beamwidth = selected_beamwidth
        if beamwidth is not None:
            item["beamwidth"] = beamwidth
        for key in ["route", "search_l"]:
            selected_value = selected_row.get(key)
            if selected_value not in (None, ""):
                item[key] = selected_value
        item["serving_configuration"] = "selected route/L/beamwidth"
        annotated.append(item)
    return annotated


def build_pq_compare(
    baseline_dir: Path, selected: list[dict[str, Any]], pq_replay_dir: Path | None
) -> list[dict[str, Any]]:
    selected_by_case = {dynamic_case_id(row): row for row in selected if dynamic_case_id(row)}
    pq_replay_rows: list[dict[str, Any]] = []
    if pq_replay_dir:
        pq_replay_rows = [payload(row) for row in read_jsonl(pq_replay_dir / "optimized_dynamic_update_results.jsonl")]
        if not pq_replay_rows:
            pq_replay_rows = [payload(row) for row in read_jsonl(pq_replay_dir / "raw" / "selected_super32k.jsonl")]
    pq_by_case = {dynamic_case_id(row): row for row in pq_replay_rows if dynamic_case_id(row)}

    compare: list[dict[str, Any]] = []
    for row in read_jsonl(baseline_dir / "raw" / "phaseC_penalty.jsonl"):
        case_id = dynamic_case_id(row, variant=str(row.get("no_retrain_variant") or "no_retrain_across_cycles"))
        selected_row = pq_by_case.get(case_id) or selected_by_case.get(case_id) or {}
        compare.append(
            {
                "case_id": case_id,
                "cycle_idx": row.get("cycle_idx"),
                "selector_type": row.get("selector_type"),
                "bucket": row.get("bucket"),
                "reference_variant": row.get("reference_variant"),
                "reference_route": row.get("reference_route"),
                "reference_L": row.get("reference_L"),
                "reference_recall": row.get("reference_recall"),
                "reference_avg_latency_ms": row.get("reference_avg_latency_ms"),
                "no_retrain_variant": row.get("no_retrain_variant"),
                "no_retrain_route": row.get("no_retrain_route"),
                "no_retrain_L": row.get("no_retrain_L"),
                "no_retrain_recall": row.get("no_retrain_recall"),
                "no_retrain_avg_latency_ms": row.get("no_retrain_avg_latency_ms"),
                "matched_reference_status": row.get("matched_reference_status"),
                "matched_reference_target_recall": row.get("matched_reference_target_recall"),
                "matched_reference_route": row.get("matched_reference_route"),
                "matched_reference_L": row.get("matched_reference_L"),
                "matched_reference_avg_latency_ms": row.get("matched_reference_avg_latency_ms"),
                "v3_route": selected_row.get("route") or selected_row.get("actual_route"),
                "v3_L": selected_row.get("search_l") or selected_row.get("chosen_L") or selected_row.get("configured_L"),
                "v3_beamwidth": command_beamwidth(selected_row),
                "v3_recall@10": selected_row.get("recall@10"),
                "v3_avg_latency_ms": (
                    fnum(selected_row, "avg_latency_us") / 1000.0 if selected_row.get("avg_latency_us") is not None else None
                ),
                "v3_p95_latency_ms": (
                    fnum(selected_row, "p95_latency_us") / 1000.0 if selected_row.get("p95_latency_us") is not None else None
                ),
            }
        )
    return compare


def selected_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall_values = [maybe_float(row.get("recall@10")) for row in rows]
    avg_us_values = [maybe_float(row.get("avg_latency_us")) for row in rows]
    p95_us_values = [maybe_float(row.get("p95_latency_us")) for row in rows]
    recalls = [value for value in recall_values if value is not None]
    avg_ms = [value / 1000.0 for value in avg_us_values if value is not None]
    p95_ms = [value / 1000.0 for value in p95_us_values if value is not None]
    case_ids = [dynamic_case_id(row) for row in rows]
    nonempty_case_ids = [case_id for case_id in case_ids if case_id]
    duplicate_case_count = len(nonempty_case_ids) - len(set(nonempty_case_ids))
    tuples = [canonical_dynamic_tuple(row) for row in rows]
    present_tuples = {item for item in tuples if item is not None}
    query_counts = [command_query_limit(row) for row in rows]
    beamwidths = [command_beamwidth(row) for row in rows]
    beamwidth_counts = Counter(str(value) for value in beamwidths if value is not None)
    prefix_count = len(selected_v3_prefixes(rows))
    prefix_present_count = sum(1 for row in rows if row.get("v3_source_prefix") or row.get("source_prefix"))
    return {
        "selected_count": len(rows),
        "unique_case_count": len(set(nonempty_case_ids)),
        "missing_case_id_count": len(rows) - len(nonempty_case_ids),
        "duplicate_case_count": duplicate_case_count,
        "dynamic_tuple_count": len(present_tuples),
        "missing_dynamic_tuple_count": sum(1 for item in tuples if item is None),
        "dynamic_tuples": sorted("|".join(map(str, item)) for item in present_tuples),
        "status_ok_count": sum(row.get("status") == "ok" for row in rows),
        "prefix_present_count": prefix_present_count,
        "unique_prefix_count": prefix_count,
        "query_count_present_count": sum(value is not None for value in query_counts),
        "min_query_count": min((value for value in query_counts if value is not None), default=0.0),
        "beamwidth_present_count": sum(value is not None for value in beamwidths),
        "beamwidth_distribution": dict(sorted(beamwidth_counts.items(), key=lambda item: int(item[0]))),
        "beamwidth_retuned_rows": sum(value is not None and value != 4 for value in beamwidths),
        "serving_configuration": "selected route/L/beamwidth",
        "recall_present_count": len(recalls),
        "avg_latency_present_count": len(avg_ms),
        "p95_latency_present_count": len(p95_ms),
        "recall_pass_count": sum(value >= 98.0 for value in recalls),
        "avg_lt_10ms_count": sum(value < 10.0 for value in avg_ms),
        "p95_lt_10ms_count": sum(value < 10.0 for value in p95_ms),
        "min_recall": min(recalls, default=0.0),
        "max_avg_latency_ms": max(avg_ms, default=0.0),
        "max_p95_latency_ms": max(p95_ms, default=0.0),
        "routes": dict(Counter(str(row.get("route") or row.get("actual_route") or "") for row in rows)),
    }


def read_invariant_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_values: list[float] = []
    violations: list[str] = []
    for idx, row in enumerate(rows):
        case_id = dynamic_case_id(row) or f"row_{idx}"
        if row.get("layout_version") != 3:
            violations.append(f"{case_id}:layout_version={row.get('layout_version')}")
        if row.get("layout_variant") not in ("page_aware_slots", None):
            violations.append(f"{case_id}:layout_variant={row.get('layout_variant')}")
        if int(row.get("physical_read_unit_bytes") or 0) != 4096:
            violations.append(f"{case_id}:physical_read_unit_bytes={row.get('physical_read_unit_bytes')}")
        if int(row.get("per_node_read_request_bytes") or 0) != 4096:
            violations.append(f"{case_id}:per_node_read_request_bytes={row.get('per_node_read_request_bytes')}")
        mean_n_4k = maybe_float(row.get("mean_n_4k"))
        mean_read_size = maybe_float(row.get("mean_read_size"))
        if not mean_n_4k or mean_read_size is None:
            violations.append(f"{case_id}:missing_query_stats")
            continue
        ratio = mean_read_size / mean_n_4k
        ratio_values.append(ratio)
        if abs(ratio - 4096.0) > 1e-6:
            violations.append(f"{case_id}:bytes_per_4k={ratio:.6f}")
    return {
        "rows": len(rows),
        "ratio_checked_rows": len(ratio_values),
        "min_bytes_per_4k": min(ratio_values, default=0.0),
        "max_bytes_per_4k": max(ratio_values, default=0.0),
        "violations": violations[:20],
        "violation_count": len(violations),
    }


def repack_layout_metrics(repack_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    ok_prefixes: set[str] = set()
    for idx, row in enumerate(repack_rows):
        prefix = row.get("dst_prefix") or f"repack_{idx}"
        if row.get("status") not in ("ok", "exists_current"):
            violations.append(f"{prefix}:status={row.get('status')}")
        else:
            ok_prefixes.add(str(prefix))
        if row.get("layout_version") != 3:
            violations.append(f"{prefix}:layout_version={row.get('layout_version')}")
        if row.get("layout_variant") != "page_aware_slots":
            violations.append(f"{prefix}:layout_variant={row.get('layout_variant')}")
        if int(row.get("layout_block_bytes") or 0) != 32768:
            violations.append(f"{prefix}:layout_block_bytes={row.get('layout_block_bytes')}")
        if int(row.get("layout_read_page_bytes") or row.get("read_page_bytes") or 0) != 4096:
            violations.append(f"{prefix}:layout_read_page_bytes={row.get('layout_read_page_bytes')}")
        label_size = maybe_float(row.get("label_size"))
        if label_size is None or int(label_size) != 0:
            violations.append(f"{prefix}:label_size={row.get('label_size')}")
        straddling = maybe_float(row.get("straddling_slots_per_block"))
        avg_pages = maybe_float(row.get("avg_4k_pages_per_record"))
        nodes_per_block = maybe_float(row.get("layout_nodes_per_block"))
        if straddling is None or avg_pages is None or nodes_per_block is None or nodes_per_block <= 0:
            violations.append(f"{prefix}:missing_packed_slot_metrics")
        else:
            expected_avg = 1.0 + straddling / nodes_per_block
            if straddling <= 0 or avg_pages <= 1.0 or abs(avg_pages - expected_avg) > 1e-9:
                violations.append(f"{prefix}:invalid_packed_slot_metrics")
    return {
        "repack_rows": len(repack_rows),
        "v3_prefixes": sorted(ok_prefixes),
        "layout_v3_page_aware_rows": sum(
            1
            for row in repack_rows
            if row.get("layout_version") == 3
            and row.get("layout_variant") == "page_aware_slots"
            and int(row.get("layout_block_bytes") or 0) == 32768
            and int(row.get("layout_read_page_bytes") or row.get("read_page_bytes") or 0) == 4096
        ),
        "label_size_zero_rows": sum(1 for row in repack_rows if maybe_float(row.get("label_size")) == 0.0),
        "straddling_slots_per_block": sorted(
            {row.get("straddling_slots_per_block") for row in repack_rows if row.get("straddling_slots_per_block") is not None}
        ),
        "avg_4k_pages_per_record": sorted(
            {row.get("avg_4k_pages_per_record") for row in repack_rows if row.get("avg_4k_pages_per_record") is not None}
        ),
        "violations": violations[:20],
        "violation_count": len(violations),
    }


def space_metrics(space_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [maybe_float(row.get("strict_total_over_raw_x")) for row in space_rows]
    excess = [maybe_float(row.get("strict_excess_over_raw_x")) for row in space_rows]
    invalid_rows: list[str] = []
    unique_v3_prefixes = space_v3_prefixes(space_rows)
    for idx, row in enumerate(space_rows):
        total = maybe_float(row.get("strict_total_over_raw_x"))
        extra = maybe_float(row.get("strict_excess_over_raw_x"))
        raw = maybe_float(row.get("raw_vector_file_bytes"))
        serving = maybe_float(row.get("strict_serving_bytes"))
        component_names = [
            "disk_index",
            "pq_codes",
            "pq_pivots",
            "labels_sidecar",
            "hybrid_meta",
            "disk_tags",
            "mem_index",
            "mem_index_tags",
        ]
        components = [maybe_float(row.get(name)) for name in component_names]
        if total is None or extra is None or raw is None or serving is None or raw <= 0 or serving <= 0:
            invalid_rows.append(f"row_{idx}:missing_or_invalid_ratio_or_bytes")
            continue
        if any(value is None for value in components):
            invalid_rows.append(f"row_{idx}:missing_space_component")
            continue
        if any((value or 0.0) < 0 for value in components):
            invalid_rows.append(f"row_{idx}:negative_space_component")
            continue
        component_sum = sum(value or 0.0 for value in components)
        if abs(component_sum - serving) > 0.5:
            invalid_rows.append(f"row_{idx}:component_sum_mismatch={component_sum}/{serving}")
        if abs(total - serving / raw) > 1e-9:
            invalid_rows.append(f"row_{idx}:total_ratio_mismatch")
        if abs(extra - (serving - raw) / raw) > 1e-9:
            invalid_rows.append(f"row_{idx}:excess_ratio_mismatch")
    totals = [value for value in totals if value is not None]
    excess = [value for value in excess if value is not None]
    return {
        "space_rows": len(space_rows),
        "valid_ratio_rows": len(space_rows) - len(invalid_rows),
        "unique_v3_prefix_count": len(unique_v3_prefixes),
        "v3_prefixes": sorted(unique_v3_prefixes),
        "invalid_rows": invalid_rows[:20],
        "invalid_row_count": len(invalid_rows),
        "max_strict_total_over_raw_x": max(totals, default=0.0),
        "max_strict_excess_over_raw_x": max(excess, default=0.0),
        "min_strict_total_over_raw_x": min(totals, default=0.0),
        "min_strict_excess_over_raw_x": min(excess, default=0.0),
    }


def explicit_evidence_metrics(
    explicitmat_dir: Path | None,
    baseline_dir: Path,
    expected_cycles: int,
    expected_delete_count: int,
    min_delete_fraction: float,
    expected_zero_insert_count: int,
) -> dict[str, Any]:
    zero_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    if explicitmat_dir:
        zero_rows = read_jsonl(explicitmat_dir / "raw" / "phaseB_zero_insert.jsonl")
        selected_rows = [payload(row) for row in read_jsonl(explicitmat_dir / "raw" / "phaseB_selected_route_l.jsonl")]
    delete_rows = read_jsonl(baseline_dir / "raw" / "phaseC_delete_steps.jsonl")
    valid_delete_volume_rows = 0
    valid_delete_timing_rows = 0
    valid_merge_rows = 0
    valid_delete_cycles: set[int] = set()
    valid_merge_cycles: set[int] = set()
    delete_volume_violations: list[str] = []
    for idx, row in enumerate(delete_rows):
        cycle_idx = parse_cycle_idx(row.get("cycle_idx") or row.get("cycle") or row.get("dest_prefix"))
        delete_count = fnum(row, "delete_count")
        live_after_delete = maybe_float(row.get("live_point_count"))
        if live_after_delete is None:
            delete_volume_violations.append(f"row_{idx}:missing_live_point_count")
            source_live = 0.0
        else:
            source_live = live_after_delete + delete_count
        fraction = delete_count / source_live if source_live > 0 else 0.0
        row_ok = row.get("status") == "ok"
        if not row_ok:
            delete_volume_violations.append(f"row_{idx}:status={row.get('status')}")
        volume_ok = delete_count >= expected_delete_count and fraction >= min_delete_fraction
        if not volume_ok:
            delete_volume_violations.append(
                f"row_{idx}:delete_count={delete_count:.0f}:delete_fraction={fraction:.6f}"
            )
        if row_ok and volume_ok:
            valid_delete_volume_rows += 1
            if cycle_idx:
                valid_delete_cycles.add(cycle_idx)
        delete_elapsed = maybe_float(row.get("delete_elapsed_s"))
        if delete_elapsed is None or delete_elapsed <= 0:
            delete_volume_violations.append(f"row_{idx}:delete_elapsed_s={row.get('delete_elapsed_s')}")
        elif row_ok and volume_ok:
            valid_delete_timing_rows += 1
        if row_ok and row.get("merge_elapsed_s") is not None and fnum(row, "merge_elapsed_s") > 0:
            valid_merge_rows += 1
            if cycle_idx:
                valid_merge_cycles.add(cycle_idx)
    delete_ms_per_vector = [
        fnum(row, "delete_elapsed_s") * 1000.0 / fnum(row, "delete_count")
        for row in delete_rows
        if fnum(row, "delete_count") > 0
    ]
    merge_s = [fnum(row, "merge_elapsed_s") for row in delete_rows if row.get("merge_elapsed_s") is not None]
    zero = zero_rows[-1] if zero_rows else {}
    selected_recalls_raw = [maybe_float(row.get("recall@10")) for row in selected_rows]
    selected_avg_us_raw = [maybe_float(row.get("avg_latency_us")) for row in selected_rows]
    selected_p95_us_raw = [maybe_float(row.get("p95_latency_us")) for row in selected_rows]
    selected_status_ok = sum(row.get("status") == "ok" for row in selected_rows)
    selected_recalls = [value for value in selected_recalls_raw if value is not None]
    selected_avg_ms = [value / 1000.0 for value in selected_avg_us_raw if value is not None]
    selected_p95_ms = [value / 1000.0 for value in selected_p95_us_raw if value is not None]
    sidecar_ok = bool(zero) and zero.get("label_storage_mode") == "sidecar"
    main_label_size = maybe_float(zero.get("main_index_label_size"))
    main_label_zero = bool(zero) and main_label_size is not None and int(main_label_size) == 0
    zero_insert_count = maybe_float(zero.get("insert_count"))
    materialize_wall_s = maybe_float(zero.get("materialize_wall_s"))
    zero_merge_wall_s = maybe_float(zero.get("merge_wall_s"))
    metrics = {
        "zero_insert_rows": len(zero_rows),
        "zero_insert_count": zero.get("insert_count"),
        "expected_zero_insert_count": expected_zero_insert_count,
        "materialize_wall_s": zero.get("materialize_wall_s"),
        "zero_insert_merge_wall_s": zero.get("merge_wall_s"),
        "label_storage_mode": zero.get("label_storage_mode"),
        "main_index_label_size": zero.get("main_index_label_size"),
        "label_sidecar_loadable": zero.get("label_sidecar_loadable"),
        "phaseB_selected_rows": len(selected_rows),
        "phaseB_selected_recall_present_count": len(selected_recalls),
        "phaseB_selected_avg_latency_present_count": len(selected_avg_ms),
        "phaseB_selected_p95_latency_present_count": len(selected_p95_ms),
        "phaseB_selected_status_ok_count": selected_status_ok,
        "phaseB_selected_recall_pass": sum(value >= 98.0 for value in selected_recalls),
        "phaseB_selected_avg_lt_10ms": sum(value < 10.0 for value in selected_avg_ms),
        "phaseB_selected_p95_lt_10ms": sum(value < 10.0 for value in selected_p95_ms),
        "delete_rows": len(delete_rows),
        "valid_delete_volume_rows": valid_delete_volume_rows,
        "valid_delete_cycles": sorted(valid_delete_cycles),
        "valid_delete_timing_rows": valid_delete_timing_rows,
        "expected_delete_count": expected_delete_count,
        "min_delete_fraction": min_delete_fraction,
        "valid_merge_rows": valid_merge_rows,
        "valid_merge_cycles": sorted(valid_merge_cycles),
        "delete_volume_violations": delete_volume_violations[:20],
        "delete_volume_violation_count": len(delete_volume_violations),
        "max_delete_ms_per_vector": max(delete_ms_per_vector, default=0.0),
        "max_delete_merge_s": max(merge_s, default=0.0),
    }
    expected_cycle_set = set(range(1, expected_cycles + 1))
    metrics["pass"] = (
        len(zero_rows) > 0
        and zero.get("status") == "ok"
        and zero_insert_count is not None
        and zero_insert_count >= expected_zero_insert_count
        and materialize_wall_s is not None
        and materialize_wall_s > 0
        and zero_merge_wall_s is not None
        and zero_merge_wall_s > 0
        and sidecar_ok
        and main_label_zero
        and zero.get("label_sidecar_loadable") is True
        and len(selected_rows) > 0
        and selected_status_ok == len(selected_rows)
        and metrics["phaseB_selected_recall_present_count"] == len(selected_rows)
        and metrics["phaseB_selected_avg_latency_present_count"] == len(selected_rows)
        and metrics["phaseB_selected_p95_latency_present_count"] == len(selected_rows)
        and metrics["phaseB_selected_recall_pass"] == len(selected_rows)
        and metrics["phaseB_selected_avg_lt_10ms"] == len(selected_rows)
        and metrics["phaseB_selected_p95_lt_10ms"] == len(selected_rows)
        and len(delete_rows) >= expected_cycles
        and valid_delete_volume_rows >= expected_cycles
        and valid_delete_cycles.issuperset(expected_cycle_set)
        and valid_delete_timing_rows >= expected_cycles
        and valid_merge_rows >= expected_cycles
        and valid_merge_cycles.issuperset(expected_cycle_set)
        and len(delete_volume_violations) == 0
        and metrics["max_delete_ms_per_vector"] < 1.0
        and metrics["max_delete_merge_s"] < 180.0
    )
    return metrics


def background_evidence_metrics(
    background_dir: Path | None,
    v3_replay_dir: Path,
    selected_rows: list[dict[str, Any]],
    selected_prefixes: set[str],
    *,
    expected_foreground_rows: int,
    expected_query_limit: int,
    expected_case_count: int,
) -> dict[str, Any]:
    if background_dir is None:
        return {
            "expected_foreground_rows": expected_foreground_rows,
            "expected_query_limit": expected_query_limit,
            "expected_case_count": expected_case_count,
            "violation_count": 1,
            "violations": ["missing_background_dir"],
            "pass": False,
        }
    summary = read_json(background_dir / "background_interference_summary.json")
    config = read_json(background_dir / "evidence" / "runner_config.json")
    foreground_rows = read_jsonl(background_dir / "foreground_search_results.jsonl")
    selected_pairs = selected_case_prefix_pairs(selected_rows)
    selected_by_index: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(selected_rows):
        try:
            selected_idx = int(row.get("row_index", idx))
        except (TypeError, ValueError):
            selected_idx = idx
        selected_by_index[selected_idx] = row

    def selected_for_foreground(row: dict[str, Any]) -> dict[str, Any]:
        try:
            idx = int(row.get("source_row_index"))
        except (TypeError, ValueError):
            return {}
        return selected_by_index.get(idx, {})

    def foreground_case_id(row: dict[str, Any]) -> str:
        return dynamic_case_id(row) or dynamic_case_id(selected_for_foreground(row))

    def foreground_v3_prefix(row: dict[str, Any]) -> str:
        return str(
            row.get("v3_source_prefix")
            or row.get("source_v3_source_prefix")
            or row.get("source_prefix")
            or selected_for_foreground(row).get("v3_source_prefix")
            or selected_for_foreground(row).get("source_prefix")
            or ""
        )

    def foreground_case_prefix_pair(row: dict[str, Any]) -> tuple[str, str] | None:
        case_id = foreground_case_id(row)
        prefix = foreground_v3_prefix(row)
        return (case_id, prefix) if case_id and prefix else None

    background_rows = read_jsonl(background_dir / "background_maintenance_results.jsonl")
    violations: list[str] = []
    if summary.get("claim_status") != "PASS":
        violations.append(f"claim_status={summary.get('claim_status')}")
    if summary.get("background_kind") != "full_build_pq_retrain":
        violations.append(f"background_kind={summary.get('background_kind')}")
    if int(summary.get("background_cpu_cap") or 0) <= 0 or int(summary.get("background_cpu_cap") or 0) > 4:
        violations.append(f"background_cpu_cap={summary.get('background_cpu_cap')}")
    if len(background_rows) != 1:
        violations.append(f"background_rows={len(background_rows)}")
        background_row: dict[str, Any] = {}
    else:
        background_row = background_rows[0]
    if background_row:
        for field in ["background_kind", "background_cpu_range", "background_cpu_cap"]:
            if background_row.get(field) != summary.get(field):
                violations.append(f"background_{field}_mismatch={background_row.get(field)}/{summary.get(field)}")
        if background_row.get("status") != "ok":
            violations.append(f"background_status={background_row.get('status')}")
        raw_train = maybe_float(background_row.get("pq_train_wall_s"))
        raw_recode = maybe_float(background_row.get("pq_recode_wall_s"))
        raw_elapsed = maybe_float(background_row.get("background_elapsed_wall_s"))
        if raw_train is None or abs(raw_train - fnum(summary, "background_pq_train_wall_s")) > 1e-6:
            violations.append(f"background_pq_train_mismatch={background_row.get('pq_train_wall_s')}/{summary.get('background_pq_train_wall_s')}")
        if raw_recode is None or abs(raw_recode - fnum(summary, "background_pq_recode_wall_s")) > 1e-6:
            violations.append(f"background_pq_recode_mismatch={background_row.get('pq_recode_wall_s')}/{summary.get('background_pq_recode_wall_s')}")
        if raw_elapsed is None or abs(raw_elapsed - fnum(summary, "background_elapsed_wall_s")) > 1.0:
            violations.append(f"background_elapsed_mismatch={background_row.get('background_elapsed_wall_s')}/{summary.get('background_elapsed_wall_s')}")
    if int(summary.get("foreground_rows") or 0) < expected_foreground_rows:
        violations.append(f"foreground_rows={summary.get('foreground_rows')}")
    if len(foreground_rows) != int(summary.get("foreground_rows") or -1):
        violations.append(f"foreground_row_count_mismatch={len(foreground_rows)}/{summary.get('foreground_rows')}")
    if int(summary.get("during_background_rows") or 0) <= 0:
        violations.append(f"during_background_rows={summary.get('during_background_rows')}")
    if int(config.get("query_limit") or 0) < expected_query_limit:
        violations.append(f"query_limit={config.get('query_limit')}")
    selected_jsonl = config.get("foreground_selected_jsonl")
    expected_selected_jsonl = (v3_replay_dir / "raw" / "selected_super32k.jsonl").resolve()
    expected_raw_dir = str((v3_replay_dir / "raw").resolve())
    configured_selected_rows: list[dict[str, Any]] = []
    if not selected_jsonl:
        violations.append("foreground_selected_jsonl=missing")
    else:
        selected_path = resolve(selected_jsonl).resolve()
        if not selected_path.exists():
            violations.append(f"foreground_selected_jsonl_missing={selected_path}")
        configured_selected_rows = [payload(row) for row in read_jsonl(selected_path)] if selected_path.exists() else []
        configured_cases = {dynamic_case_id(row) for row in configured_selected_rows if dynamic_case_id(row)}
        selected_cases = {dynamic_case_id(row) for row in selected_rows if dynamic_case_id(row)}
        configured_pairs = selected_case_prefix_pairs(configured_selected_rows)
        selected_digest = sha256_file(expected_selected_jsonl)
        configured_digest = sha256_file(selected_path)
        selected_path_ok = (
            selected_path == expected_selected_jsonl
            or (selected_digest and configured_digest and selected_digest == configured_digest)
            or (bool(configured_pairs) and configured_pairs == selected_pairs)
        )
        if not selected_path_ok:
            violations.append(f"foreground_selected_jsonl={selected_path}")
        if not configured_cases or configured_cases != selected_cases:
            violations.append(
                f"foreground_selected_case_set_mismatch={len(configured_cases)}/{len(selected_cases)}"
            )
    if int(config.get("background_cpu_cap") or summary.get("background_cpu_cap") or 0) > 4:
        violations.append(f"config_background_cpu_cap={config.get('background_cpu_cap')}")
    row_prefixes = selected_v3_prefixes(foreground_rows)
    if not row_prefixes:
        violations.append("foreground_prefixes=empty")
    if not selected_prefixes:
        violations.append("selected_prefixes=empty")
    if row_prefixes and selected_prefixes and not row_prefixes.issubset(selected_prefixes):
        violations.append(f"foreground_prefixes_not_in_selected={sorted(row_prefixes - selected_prefixes)}")
    foreground_cases = {foreground_case_id(row) for row in foreground_rows if foreground_case_id(row)}
    if not foreground_cases:
        row_index = int(config.get("foreground_row_index") or -1)
        if 0 <= row_index < len(configured_selected_rows):
            inferred_case = dynamic_case_id(configured_selected_rows[row_index])
            if inferred_case:
                foreground_cases.add(inferred_case)
    selected_cases = {dynamic_case_id(row) for row in selected_rows if dynamic_case_id(row)}
    if not foreground_cases:
        violations.append("foreground_cases=empty")
    elif not foreground_cases.issubset(selected_cases):
        violations.append(f"foreground_cases_not_in_selected={sorted(foreground_cases - selected_cases)}")
    if len(foreground_cases) < expected_case_count:
        violations.append(f"foreground_case_count={len(foreground_cases)}/{expected_case_count}")
    conditions = Counter(str(row.get("condition") or "") for row in foreground_rows)
    baseline_expected = str(config.get("baseline_schedule") or summary.get("baseline_schedule") or "") != "none"
    if (baseline_expected and conditions.get("baseline", 0) <= 0) or conditions.get("during_background", 0) <= 0:
        violations.append(f"foreground_conditions={dict(conditions)}")
    expected_baseline_rows = int(summary["baseline_rows"]) if summary.get("baseline_rows") is not None else -1
    if conditions.get("baseline", 0) != expected_baseline_rows:
        violations.append(f"baseline_rows_mismatch={conditions.get('baseline', 0)}/{summary.get('baseline_rows')}")
    if conditions.get("during_background", 0) != int(summary.get("during_background_rows") or -1):
        violations.append(
            f"during_rows_mismatch={conditions.get('during_background', 0)}/{summary.get('during_background_rows')}"
    )
    during_rows = int(summary.get("during_background_rows") or 0)
    during = [row for row in foreground_rows if row.get("condition") == "during_background"]
    during_cases = {foreground_case_id(row) for row in during if foreground_case_id(row)}
    if not during_cases:
        row_index = int(config.get("foreground_row_index") or -1)
        if 0 <= row_index < len(configured_selected_rows):
            inferred_case = dynamic_case_id(configured_selected_rows[row_index])
            if inferred_case:
                during_cases.add(inferred_case)
    if not during_cases:
        violations.append("during_cases=empty")
    elif not during_cases.issubset(selected_cases):
        violations.append(f"during_cases_not_in_selected={sorted(during_cases - selected_cases)}")
    if len(during_cases) < expected_case_count:
        violations.append(f"during_case_count={len(during_cases)}/{expected_case_count}")
    during_pairs = {pair for pair in (foreground_case_prefix_pair(row) for row in during) if pair is not None}
    if not during_pairs:
        row_index = int(config.get("foreground_row_index") or -1)
        if 0 <= row_index < len(configured_selected_rows):
            inferred_case = dynamic_case_id(configured_selected_rows[row_index])
            inferred_prefix = str(
                configured_selected_rows[row_index].get("v3_source_prefix")
                or configured_selected_rows[row_index].get("source_prefix")
                or ""
            )
            if inferred_case and inferred_prefix:
                during_pairs.add((inferred_case, inferred_prefix))
    if during_pairs != selected_pairs:
        missing = sorted(selected_pairs - during_pairs)
        extra = sorted(during_pairs - selected_pairs)
        violations.append(f"during_case_prefix_pair_mismatch=missing:{len(missing)} extra:{len(extra)}")
    recomputed_recall_pass = sum(1 for row in during if fnum(row, "recall@10", -1.0) >= 98.0)
    recomputed_avg_pass = sum(1 for row in during if 0.0 < fnum(row, "avg_latency_us", 1e18) < 10000.0)
    recomputed_p95_pass = sum(1 for row in during if 0.0 < fnum(row, "p95_latency_us", 1e18) < 10000.0)
    recomputed_max_avg = max((fnum(row, "avg_latency_us") / 1000.0 for row in during), default=0.0)
    recomputed_max_p95 = max((fnum(row, "p95_latency_us") / 1000.0 for row in during), default=0.0)
    if int(summary.get("during_recall_pass") or 0) != during_rows:
        violations.append(f"during_recall_pass={summary.get('during_recall_pass')}/{during_rows}")
    if recomputed_recall_pass != during_rows:
        violations.append(f"recomputed_during_recall_pass={recomputed_recall_pass}/{during_rows}")
    if int(summary.get("during_avg_lt_10ms_pass") or 0) != during_rows:
        violations.append(f"during_avg_lt_10ms_pass={summary.get('during_avg_lt_10ms_pass')}/{during_rows}")
    if recomputed_avg_pass != during_rows:
        violations.append(f"recomputed_during_avg_lt_10ms_pass={recomputed_avg_pass}/{during_rows}")
    if int(summary.get("during_p95_lt_10ms_pass") or 0) != during_rows:
        violations.append(f"during_p95_lt_10ms_pass={summary.get('during_p95_lt_10ms_pass')}/{during_rows}")
    if recomputed_p95_pass != during_rows:
        violations.append(f"recomputed_during_p95_lt_10ms_pass={recomputed_p95_pass}/{during_rows}")
    if abs(fnum(summary, "during_max_avg_latency_ms") - recomputed_max_avg) > 1e-6:
        violations.append(f"during_max_avg_mismatch={summary.get('during_max_avg_latency_ms')}/{recomputed_max_avg}")
    if abs(fnum(summary, "during_max_p95_latency_ms") - recomputed_max_p95) > 1e-6:
        violations.append(f"during_max_p95_mismatch={summary.get('during_max_p95_latency_ms')}/{recomputed_max_p95}")
    required_latency_fields = ["during_max_avg_latency_ms", "during_max_p95_latency_ms"]
    if baseline_expected:
        required_latency_fields.append("baseline_max_avg_latency_ms")
    for field in required_latency_fields:
        value = maybe_float(summary.get(field))
        if value is None or value <= 0 or value >= 10.0:
            violations.append(f"{field}={summary.get(field)}")
    for field in ["background_elapsed_wall_s", "background_pq_train_wall_s", "background_pq_recode_wall_s"]:
        value = maybe_float(summary.get(field))
        if value is None or value <= 0:
            violations.append(f"{field}={summary.get(field)}")
    for idx, row in enumerate(foreground_rows):
        if row.get("status") != "ok":
            violations.append(f"foreground_{idx}:status={row.get('status')}")
        for field in ["recall@10", "avg_latency_us", "p95_latency_us", "mean_n_4k", "mean_read_size"]:
            if maybe_float(row.get(field)) is None:
                violations.append(f"foreground_{idx}:missing_{field}")
                break
        if int(row.get("physical_read_unit_bytes") or 0) != 4096:
            violations.append(f"foreground_{idx}:physical_read_unit_bytes={row.get('physical_read_unit_bytes')}")
        if int(row.get("per_node_read_request_bytes") or 0) != 4096:
            violations.append(f"foreground_{idx}:per_node_read_request_bytes={row.get('per_node_read_request_bytes')}")
        if row.get("source_layout_metadata_valid") is not True:
            violations.append(f"foreground_{idx}:source_layout_metadata_valid={row.get('source_layout_metadata_valid')}")
        if row.get("source_prefix_matches_v3") is not True:
            violations.append(f"foreground_{idx}:source_prefix_matches_v3={row.get('source_prefix_matches_v3')}")
        if int(row.get("actual_layout_version") or 0) != 3:
            violations.append(f"foreground_{idx}:actual_layout_version={row.get('actual_layout_version')}")
        if int(row.get("actual_layout_block_bytes") or 0) != 32768:
            violations.append(f"foreground_{idx}:actual_layout_block_bytes={row.get('actual_layout_block_bytes')}")
        if int(row.get("actual_layout_read_page_bytes") or 0) != 4096:
            violations.append(f"foreground_{idx}:actual_layout_read_page_bytes={row.get('actual_layout_read_page_bytes')}")
        n4k = maybe_float(row.get("mean_n_4k"))
        read_size = maybe_float(row.get("mean_read_size"))
        if n4k is None or n4k <= 0 or read_size is None:
            violations.append(f"foreground_{idx}:invalid_read_stats")
        elif abs(read_size / n4k - 4096.0) > 1e-6:
            violations.append(f"foreground_{idx}:bytes_per_4k={read_size / n4k:.6f}")
    return {
        **summary,
        "runner_config_query_limit": config.get("query_limit"),
        "foreground_selected_jsonl": config.get("foreground_selected_jsonl"),
        "background_raw_rows": len(background_rows),
        "expected_foreground_selected_jsonl": str(expected_selected_jsonl),
        "expected_v3_replay_raw_dir": expected_raw_dir,
        "foreground_prefixes": sorted(row_prefixes),
        "selected_v3_prefixes": sorted(selected_prefixes),
        "foreground_cases": sorted(foreground_cases),
        "during_cases": sorted(during_cases),
        "during_case_prefix_pair_count": len(during_pairs),
        "selected_case_prefix_pair_count": len(selected_pairs),
        "expected_case_count": expected_case_count,
        "recomputed_during_recall_pass": recomputed_recall_pass,
        "recomputed_during_avg_lt_10ms_pass": recomputed_avg_pass,
        "recomputed_during_p95_lt_10ms_pass": recomputed_p95_pass,
        "expected_foreground_rows": expected_foreground_rows,
        "expected_query_limit": expected_query_limit,
        "violation_count": len(violations),
        "violations": violations[:20],
        "pass": len(violations) == 0,
    }


def svg_escape(value: Any) -> str:
    import html

    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: Any, *, size: int = 14, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#111827">{svg_escape(text)}</text>'
    )


def svg_doc(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_doc(width, height, body), encoding="utf-8")


def plot_figures_svg(
    out: Path, selected: list[dict[str, Any]], space_rows: list[dict[str, Any]], background: dict[str, Any]
) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if selected:
        width, height = 900, 520
        left, right, top, bottom = 78, 35, 58, 72
        plot_w = width - left - right
        plot_h = height - top - bottom
        avg_ms = [fnum(row, "avg_latency_us") / 1000.0 for row in selected]
        recalls = [fnum(row, "recall@10") for row in selected]
        x_min = min(0.0, min(avg_ms, default=0.0))
        x_max = max(10.5, max(avg_ms, default=10.0) * 1.04)
        y_min = min(97.6, min(recalls, default=98.0) - 0.2)
        y_max = max(100.1, max(recalls, default=100.0) + 0.2)
        sx = lambda x: left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
        sy = lambda y: top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h
        body = [
            svg_text(width / 2, 30, "v100 Supersector32K replay: selected dynamic points", size=20, weight="700", anchor="middle"),
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{sx(10.0):.1f}" y1="{top}" x2="{sx(10.0):.1f}" y2="{top + plot_h}" stroke="#dc2626" stroke-dasharray="7 5"/>',
            f'<line x1="{left}" y1="{sy(98.0):.1f}" x2="{left + plot_w}" y2="{sy(98.0):.1f}" stroke="#111827" stroke-dasharray="3 5"/>',
            svg_text(left + plot_w / 2, height - 24, "Avg latency (ms)", anchor="middle"),
            svg_text(18, top + plot_h / 2, "Recall@10 (%)", anchor="middle"),
        ]
        for value in [0, 2, 4, 6, 8, 10]:
            body.append(svg_text(sx(value), top + plot_h + 24, value, size=12, anchor="middle"))
        for value in [98, 99, 100]:
            body.append(svg_text(left - 10, sy(value) + 4, value, size=12, anchor="end"))
        for x, y in zip(avg_ms, recalls):
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#0f766e" fill-opacity="0.78"/>')
        path = fig_dir / "v100_dynamic_recall_latency.svg"
        write_svg(path, width, height, body)
        written.append(str(path.relative_to(ROOT)))

    if space_rows:
        width, height = 980, 520
        row = max(space_rows, key=lambda item: fnum(item, "strict_total_over_raw_x"))
        components = [
            ("disk index", fnum(row, "disk_index"), "#64748b"),
            ("pq codes", fnum(row, "pq_codes"), "#0f766e"),
            ("pq pivots", fnum(row, "pq_pivots"), "#14b8a6"),
            ("labels sidecar", fnum(row, "labels_sidecar"), "#f59e0b"),
            ("mem index", fnum(row, "mem_index"), "#94a3b8"),
            ("tags/meta", fnum(row, "disk_tags") + fnum(row, "hybrid_meta") + fnum(row, "mem_index_tags"), "#cbd5e1"),
        ]
        raw_mib = mib(fnum(row, "raw_vector_file_bytes"))
        total_mib = sum(mib(value) for _, value, _ in components)
        scale_max = max(total_mib, raw_mib, 1.0)
        bar_x, bar_y, bar_w, bar_h = 130, 95, 95, 320
        body = [
            svg_text(width / 2, 30, "v100 strict serving footprint and acceptance accounting", size=20, weight="700", anchor="middle"),
            svg_text(bar_x + bar_w / 2, 70, "Strict serving MiB", size=15, weight="700", anchor="middle"),
        ]
        y_cursor = bar_y + bar_h
        for name, value, color in components:
            h = mib(value) / scale_max * bar_h
            y_cursor -= h
            body.append(f'<rect x="{bar_x}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}"/>')
        raw_y = bar_y + bar_h - raw_mib / scale_max * bar_h
        body.append(f'<line x1="{bar_x - 30}" y1="{raw_y:.1f}" x2="{bar_x + bar_w + 30}" y2="{raw_y:.1f}" stroke="#111827" stroke-dasharray="6 4"/>')
        body.append(svg_text(bar_x + bar_w + 38, raw_y + 4, "raw vectors", size=12))
        body.append(svg_text(bar_x + bar_w / 2, bar_y + bar_h + 24, f"{total_mib:.1f} MiB", size=12, anchor="middle"))
        legend_x, legend_y = 280, 105
        for idx, (name, value, color) in enumerate(components):
            y = legend_y + idx * 30
            body.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{color}"/>')
            body.append(svg_text(legend_x + 24, y + 1, f"{name}: {mib(value):.1f} MiB", size=12))
        ratio_x0 = 650
        ratios = [("total/raw", fnum(row, "strict_total_over_raw_x"), "#64748b"), ("excess/raw", fnum(row, "strict_excess_over_raw_x"), "#0f766e")]
        for idx, (name, value, color) in enumerate(ratios):
            x = ratio_x0 + idx * 120
            h = value / 2.2 * 300
            y = 410 - h
            body.append(f'<rect x="{x}" y="{y:.1f}" width="70" height="{h:.1f}" fill="{color}"/>')
            body.append(svg_text(x + 35, y - 10, f"{value:.3f}x", size=12, anchor="middle"))
            body.append(svg_text(x + 35, 435, name, size=12, anchor="middle"))
        body.append(f'<line x1="{ratio_x0 - 35}" y1="{410 - (2.0 / 2.2 * 300):.1f}" x2="{ratio_x0 + 250}" y2="{410 - (2.0 / 2.2 * 300):.1f}" stroke="#dc2626" stroke-dasharray="6 4"/>')
        path = fig_dir / "v100_space_components.svg"
        write_svg(path, width, height, body)
        written.append(str(path.relative_to(ROOT)))

    if background:
        labels = ["baseline avg", "during avg", "baseline p95", "during p95"]
        values = [
            fnum(background, "baseline_max_avg_latency_ms"),
            fnum(background, "during_max_avg_latency_ms"),
            fnum(background, "baseline_max_p95_latency_ms"),
            fnum(background, "during_max_p95_latency_ms"),
        ]
        if any(value > 0 for value in values):
            width, height = 820, 470
            left, top, plot_w, plot_h = 80, 65, 690, 300
            scale_max = max(10.5, max(values) * 1.15)
            colors = ["#94a3b8", "#0f766e", "#cbd5e1", "#f59e0b"]
            body = [
                svg_text(width / 2, 30, "Foreground search during low-core background maintenance", size=20, weight="700", anchor="middle"),
                f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
                f'<line x1="{left}" y1="{top + plot_h - 10.0 / scale_max * plot_h:.1f}" x2="{left + plot_w}" y2="{top + plot_h - 10.0 / scale_max * plot_h:.1f}" stroke="#dc2626" stroke-dasharray="6 4"/>',
                svg_text(22, top + plot_h / 2, "Latency (ms)", anchor="middle"),
            ]
            for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
                x = left + 70 + idx * 155
                h = value / scale_max * plot_h
                y = top + plot_h - h
                body.append(f'<rect x="{x}" y="{y:.1f}" width="82" height="{h:.1f}" fill="{color}"/>')
                body.append(svg_text(x + 41, y - 10, f"{value:.2f}", size=12, anchor="middle"))
                body.append(svg_text(x + 41, top + plot_h + 24, label, size=12, anchor="middle"))
            path = fig_dir / "v100_background_interference.svg"
            write_svg(path, width, height, body)
            written.append(str(path.relative_to(ROOT)))

    return written


def plot_figures(out: Path, selected: list[dict[str, Any]], space_rows: list[dict[str, Any]], background: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        if selected:
            recalls = [fnum(row, "recall@10") for row in selected]
            avg_ms = [fnum(row, "avg_latency_us") / 1000.0 for row in selected]
            fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
            ax.scatter(avg_ms, recalls, s=28, alpha=0.78, color="#0f766e", edgecolor="white", linewidth=0.35)
            ax.axvline(10.0, color="#dc2626", linestyle="--", linewidth=1.1)
            ax.axhline(98.0, color="#111827", linestyle=":", linewidth=1.1)
            ax.set_xlabel("Avg latency (ms)")
            ax.set_ylabel("Recall@10 (%)")
            ax.set_title("v100 Supersector32K replay: selected dynamic points")
            ax.grid(alpha=0.25)
            path = fig_dir / "v100_dynamic_recall_latency.png"
            fig.savefig(path, dpi=FIGURE_DPI)
            plt.close(fig)
            written.append(str(path.relative_to(ROOT)))

        if space_rows:
            row = max(space_rows, key=lambda item: fnum(item, "strict_total_over_raw_x"))
            components = [
                ("disk index", fnum(row, "disk_index")),
                ("pq codes", fnum(row, "pq_codes")),
                ("pq pivots", fnum(row, "pq_pivots")),
                ("labels sidecar", fnum(row, "labels_sidecar")),
                ("mem index", fnum(row, "mem_index")),
                ("tags/meta", fnum(row, "disk_tags") + fnum(row, "hybrid_meta") + fnum(row, "mem_index_tags")),
            ]
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)
            bottom = 0.0
            for name, value in components:
                axes[0].bar(["strict"], [mib(value)], bottom=bottom, label=name)
                bottom += mib(value)
            raw_mib = mib(fnum(row, "raw_vector_file_bytes"))
            axes[0].axhline(raw_mib, color="#111827", linestyle="--", linewidth=1.0, label="raw vectors")
            axes[0].set_ylabel("MiB")
            axes[0].set_title("Strict serving footprint")
            axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            ratios = [fnum(row, "strict_total_over_raw_x"), fnum(row, "strict_excess_over_raw_x")]
            axes[1].bar(["total/raw", "excess/raw"], ratios, color=["#64748b", "#0f766e"], width=0.55)
            axes[1].axhline(2.0, color="#dc2626", linestyle="--", linewidth=1.0)
            axes[1].axhline(1.0, color="#111827", linestyle=":", linewidth=1.0)
            axes[1].set_ylabel("x raw")
            axes[1].set_title("Acceptance accounting")
            for idx, value in enumerate(ratios):
                axes[1].text(idx, value + 0.04, f"{value:.3f}x", ha="center")
            path = fig_dir / "v100_space_components.png"
            fig.savefig(path, dpi=FIGURE_DPI)
            plt.close(fig)
            written.append(str(path.relative_to(ROOT)))

        if background:
            labels = ["baseline avg", "during avg", "baseline p95", "during p95"]
            values = [
                fnum(background, "baseline_max_avg_latency_ms"),
                fnum(background, "during_max_avg_latency_ms"),
                fnum(background, "baseline_max_p95_latency_ms"),
                fnum(background, "during_max_p95_latency_ms"),
            ]
            if any(value > 0 for value in values):
                fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
                ax.bar(labels, values, color=["#94a3b8", "#0f766e", "#cbd5e1", "#f59e0b"], width=0.58)
                ax.axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.0)
                ax.set_ylabel("Latency (ms)")
                ax.set_title("Foreground search during low-core background maintenance")
                ax.grid(axis="y", alpha=0.25)
                for idx, value in enumerate(values):
                    ax.text(idx, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)
                path = fig_dir / "v100_background_interference.png"
                fig.savefig(path, dpi=FIGURE_DPI)
                plt.close(fig)
                written.append(str(path.relative_to(ROOT)))

        return written
    except Exception:
        return plot_figures_svg(out, selected, space_rows, background)


def render_reports(
    out: Path,
    selected: list[dict[str, Any]],
    pq_compare: list[dict[str, Any]],
    space_rows: list[dict[str, Any]],
    background: dict[str, Any],
    explicit_metrics: dict[str, Any],
    read_metrics: dict[str, Any],
    layout_metrics: dict[str, Any],
    figures: list[str],
    expected_selected: int,
    expected_pq_rows: int,
    expected_cycles: int,
    expected_query_count: int,
) -> None:
    sm = selected_metrics(selected)
    xm = space_metrics(space_rows)
    expected_tuples = expected_dynamic_tuples(expected_cycles)
    present_tuples = {
        tuple(parts)
        for parts in (
            (int(item.split("|")[0]), item.split("|")[1], item.split("|")[2], item.split("|")[3])
            for item in sm["dynamic_tuples"]
        )
    }
    missing_tuples = sorted(expected_tuples - present_tuples)
    extra_tuples = sorted(present_tuples - expected_tuples)
    matched = [row for row in pq_compare if row.get("matched_reference_status") == "matched"]
    unmatched = [row for row in pq_compare if row.get("matched_reference_status") and row.get("matched_reference_status") != "matched"]
    expected_pq_tuples = {
        (cycle, selector, bucket)
        for cycle in range(1, expected_cycles + 1)
        for selector in SELECTORS
        for bucket in BUCKETS
    }
    pq_tuples = {
        (int(row.get("cycle_idx") or 0), str(row.get("selector_type") or ""), str(row.get("bucket") or ""))
        for row in pq_compare
    }
    pq_missing_tuples = sorted(expected_pq_tuples - pq_tuples)
    pq_extra_tuples = sorted(pq_tuples - expected_pq_tuples)
    pq_case_ids = [str(row.get("case_id") or "") for row in pq_compare]
    matched_case_ids = [str(row.get("case_id") or "") for row in matched]
    pq_unique_case_count = len({case_id for case_id in pq_case_ids if case_id})
    pq_missing_case_id_count = sum(1 for case_id in pq_case_ids if not case_id)
    pq_duplicate_case_count = len([case_id for case_id in pq_case_ids if case_id]) - pq_unique_case_count
    matched_unique_case_count = len({case_id for case_id in matched_case_ids if case_id})
    replay_prefixes = selected_v3_prefixes(selected)
    space_prefixes = set(xm["v3_prefixes"])
    layout_prefixes = set(layout_metrics["v3_prefixes"])
    primary_pass = (
        sm["selected_count"] == expected_selected
        and sm["unique_case_count"] == expected_selected
        and sm["missing_case_id_count"] == 0
        and sm["duplicate_case_count"] == 0
        and sm["dynamic_tuple_count"] == expected_selected
        and sm["missing_dynamic_tuple_count"] == 0
        and not missing_tuples
        and not extra_tuples
        and sm["status_ok_count"] == sm["selected_count"]
        and sm["prefix_present_count"] == sm["selected_count"]
        and sm["unique_prefix_count"] > 0
        and sm["query_count_present_count"] == sm["selected_count"]
        and sm["min_query_count"] >= expected_query_count
        and sm["recall_present_count"] == sm["selected_count"]
        and sm["avg_latency_present_count"] == sm["selected_count"]
        and sm["p95_latency_present_count"] == sm["selected_count"]
        and sm["recall_pass_count"] == sm["selected_count"]
        and sm["avg_lt_10ms_count"] == sm["selected_count"]
        and sm["p95_lt_10ms_count"] == sm["selected_count"]
    )
    space_pass = (
        xm["space_rows"] > 0
        and bool(replay_prefixes)
        and xm["space_rows"] == xm["valid_ratio_rows"]
        and space_prefixes == replay_prefixes
        and xm["max_strict_total_over_raw_x"] < 2.0
    )
    background_pass = primary_pass and background.get("pass") is True
    pq_pass = (
        len(pq_compare) == expected_pq_rows
        and len(matched) == expected_pq_rows
        and pq_unique_case_count == expected_pq_rows
        and matched_unique_case_count == expected_pq_rows
        and pq_missing_case_id_count == 0
        and pq_duplicate_case_count == 0
        and not pq_missing_tuples
        and not pq_extra_tuples
        and not unmatched
    )
    read_pass = read_metrics["rows"] == expected_selected and read_metrics["violation_count"] == 0
    layout_pass = (
        layout_metrics["repack_rows"] > 0
        and bool(replay_prefixes)
        and layout_metrics["violation_count"] == 0
        and layout_prefixes == replay_prefixes
    )

    claims = {
        "format": "pipeann.aris.v100_goal_claim_registry.v1",
        "claims": [
            {
                "id": "C_DYNAMIC_SELECTED_RECALL_AVG_P95_LATENCY",
                "status": "PASS" if primary_pass else "NEEDS_EVIDENCE",
                "metrics": sm,
                "coverage": {
                    "expected_cycles": expected_cycles,
                    "missing_tuples": ["|".join(map(str, item)) for item in missing_tuples[:20]],
                    "extra_tuples": ["|".join(map(str, item)) for item in extra_tuples[:20]],
                    "missing_tuple_count": len(missing_tuples),
                    "extra_tuple_count": len(extra_tuples),
                },
                "caveat": "Route/L/beamwidth retuning is allowed by the selected serving-configuration acceptance口径; this does not prove fixed-parameter graph quality under identical search parameters.",
            },
            {
                "id": "C_PQ_DRIFT_MATCHED_REFERENCE",
                "status": "PASS" if pq_pass else "NEEDS_EVIDENCE",
                "metrics": {
                    "matched": len(matched),
                    "total": len(pq_compare),
                    "expected_total": expected_pq_rows,
                    "unique_case_count": pq_unique_case_count,
                    "matched_unique_case_count": matched_unique_case_count,
                    "missing_case_id_count": pq_missing_case_id_count,
                    "duplicate_case_count": pq_duplicate_case_count,
                    "missing_tuple_count": len(pq_missing_tuples),
                    "extra_tuple_count": len(pq_extra_tuples),
                    "missing_tuples": ["|".join(map(str, item)) for item in pq_missing_tuples[:20]],
                    "extra_tuples": ["|".join(map(str, item)) for item in pq_extra_tuples[:20]],
                    "unmatched_cases": [row.get("case_id") for row in unmatched],
                },
            },
            {
                "id": "C_V3_READ_GRANULARITY_4KB",
                "status": "PASS" if read_pass and layout_pass else "NEEDS_EVIDENCE",
                "metrics": {
                    "query_stats": read_metrics,
                    "repack_layout": layout_metrics,
                    "selected_v3_prefixes": sorted(replay_prefixes),
                },
            },
            {
                "id": "C_SPACE_STRICT_TOTAL_LT_2X_RAW",
                "status": "PASS" if space_pass else "NEEDS_EVIDENCE",
                "metrics": {**xm, "selected_v3_prefixes": sorted(replay_prefixes)},
                "caveat": "The user revised the acceptance口径 to total serving footprint below 2x raw vectors.",
            },
            {
                "id": "C_BACKGROUND_MAINTENANCE_NO_FRONTEND_LATENCY_BREAK",
                "status": "PASS" if background_pass else "NEEDS_EVIDENCE",
                "metrics": background,
            },
            {
                "id": "C_EXPLICIT_MATERIALIZE_AND_DELETE_MERGE_EVIDENCE",
                "status": "PASS" if explicit_metrics.get("pass") else "NEEDS_EVIDENCE",
                "metrics": explicit_metrics,
            },
        ],
    }
    write_json(out / "optimized_claim_registry.json", claims)

    best_space = min(space_rows, key=lambda row: fnum(row, "strict_total_over_raw_x", math.inf), default={})
    worst_space = max(space_rows, key=lambda row: fnum(row, "strict_total_over_raw_x", 0.0), default={})
    (out / "index_space_audit.md").write_text(
        "# V100 Index Space Audit\n\n"
        f"- Space rows audited: `{len(space_rows)}`\n"
        f"- Best strict total/raw: `{fnum(best_space, 'strict_total_over_raw_x'):.6f}x`\n"
        f"- Worst strict total/raw: `{fnum(worst_space, 'strict_total_over_raw_x'):.6f}x`\n"
        f"- Worst strict excess/raw: `{fnum(worst_space, 'strict_excess_over_raw_x'):.6f}x`\n"
        "- Strict口径 counts active v3 disk index, PQ codes/pivots, tag/meta files, and label sidecar.\n"
        "- Engineering口径 reports excess over raw separately; transient v1 source copies and repack workspace are excluded.\n"
        "- The revised acceptance target is strict total serving footprint `<2x` raw vector bytes.\n",
        encoding="utf-8",
    )
    (out / "label_sidecar_layout_audit.md").write_text(
        "# V100 Label Sidecar Layout Audit\n\n"
        f"- Repack rows audited: `{layout_metrics['repack_rows']}`\n"
        f"- Rows with `label_size=0`: `{layout_metrics['label_size_zero_rows']}`\n"
        f"- Layout violations: `{layout_metrics['violation_count']}`\n\n"
        "The Supersector32K replay evidence must preserve label storage in `_labels.densebit` sidecars. "
        "This audit marks the claim pass only when every repacked main disk index reports `label_size=0`; "
        "label sidecars, tag maps, and metadata remain counted in strict serving footprint.\n",
        encoding="utf-8",
    )

    beamwidth_dist = ", ".join(f"{key}={value}" for key, value in sm["beamwidth_distribution"].items())
    beamwidth_dist_compact = beamwidth_dist.replace(", ", ";")
    serving_label = str(sm["serving_configuration"]).removeprefix("selected ")

    ppt_rows = [
        {"metric": "selected_rows", "value": sm["selected_count"], "note": f"expected {expected_selected}"},
        {
            "metric": "selected_serving_configuration",
            "value": sm["serving_configuration"],
            "note": "selected serving config acceptance口径",
        },
        {
            "metric": "selected_beamwidth_distribution",
            "value": beamwidth_dist_compact,
            "note": "200 selected rows",
        },
        {"metric": "min_recall@10", "value": f"{sm['min_recall']:.6f}", "note": "target >=98"},
        {"metric": "max_avg_latency_ms", "value": f"{sm['max_avg_latency_ms']:.6f}", "note": "target <10"},
        {"metric": "max_p95_latency_ms", "value": f"{sm['max_p95_latency_ms']:.6f}", "note": "target <10"},
        {"metric": "pq_matched", "value": f"{len(matched)}/{len(pq_compare)}", "note": "matched-reference status"},
        {"metric": "read_4kb_violations", "value": read_metrics["violation_count"], "note": "must be 0"},
        {
            "metric": "max_strict_total_over_raw_x",
            "value": f"{xm['max_strict_total_over_raw_x']:.6f}",
            "note": "target <2x",
        },
        {
            "metric": "background_during_max_avg_latency_ms",
            "value": background.get("during_max_avg_latency_ms", ""),
            "note": "low-core maintenance interference",
        },
    ]
    write_csv(out / "ppt_ready_metrics.csv", ppt_rows)
    (out / "ppt_ready_conclusion_summary.md").write_text(
        "# V100 PPT-Ready Conclusion Summary\n\n"
        f"- Dynamic selected rows: `{sm['selected_count']}`; recall pass `{sm['recall_pass_count']}`, avg<10ms pass `{sm['avg_lt_10ms_count']}`, p95<10ms pass `{sm['p95_lt_10ms_count']}`.\n"
        f"- Serving configuration口径: {sm['serving_configuration']}; beamwidth distribution {beamwidth_dist}.\n"
        f"- Worst avg latency: `{sm['max_avg_latency_ms']:.3f} ms`; worst p95 latency: `{sm['max_p95_latency_ms']:.3f} ms`.\n"
        f"- PQ matched-reference: `{len(matched)}/{len(pq_compare)}` matched; unmatched cases: `{[row.get('case_id') for row in unmatched]}`.\n"
        f"- 4KB read invariant violations: `{read_metrics['violation_count']}`; layout violations: `{layout_metrics['violation_count']}`.\n"
        f"- Space: worst strict total/raw `{xm['max_strict_total_over_raw_x']:.6f}x`, worst strict excess/raw `{xm['max_strict_excess_over_raw_x']:.6f}x`.\n"
        f"- Background maintenance: `{background.get('claim_status', 'missing')}`; during max avg `{background.get('during_max_avg_latency_ms', '')}` ms.\n"
        f"- Figures: `{figures}`\n",
        encoding="utf-8",
    )
    (out / "aris_final_review.md").write_text(
        "# V100 ARIS Final Review\n\n"
        f"Overall status: `{'PASS' if primary_pass and pq_pass and read_pass and layout_pass and space_pass and background_pass and explicit_metrics.get('pass') else 'NEEDS_EVIDENCE'}`.\n\n"
        "## Evidence Checks\n"
        f"- Dynamic selected recall/avg/p95 latency: {claims['claims'][0]['status']}; selected serving config {serving_label}; beamwidth distribution {beamwidth_dist}; {sm['recall_pass_count']}/{sm['selected_count']} recall, avg<10ms, and p95<10ms.\n"
        f"- PQ drift matched-reference: `{claims['claims'][1]['status']}` with `{len(matched)}/{len(pq_compare)}` matched; expected `{expected_pq_rows}`.\n"
        f"- 4KB read/layout invariant: `{claims['claims'][2]['status']}` with `{read_metrics}` and `{layout_metrics}`.\n"
        f"- Index space `<2x` strict total/raw: `{claims['claims'][3]['status']}` with `{xm}`.\n"
        f"- Background maintenance interference: `{claims['claims'][4]['status']}`.\n"
        f"- Explicit materialize/delete/merge evidence: `{claims['claims'][5]['status']}` with `{explicit_metrics}`.\n\n"
        "## Claim Wording Guardrails\n"
        "- Use the selected route/L/beamwidth口径 (selected serving configuration) for dynamic update recall; do not claim fixed-parameter graph quality under identical search parameters unless a fixed-parameter experiment is cited.\n"
        "- Report total serving footprint and excess-over-raw separately.\n"
        "- Keep the 4KB random-read primitive wording: straddling records may issue two 4KB reads, not one 32KB read.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--v3-replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--explicitmat-dir", type=Path, default=None)
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--pq-replay-dir", type=Path, default=None)
    parser.add_argument("--expected-selected", type=int, default=200)
    parser.add_argument("--expected-pq-rows", type=int, default=100)
    parser.add_argument("--expected-cycles", type=int, default=5)
    parser.add_argument("--expected-query-count", type=int, default=1000)
    parser.add_argument("--expected-delete-count", type=int, default=600000)
    parser.add_argument("--expected-zero-insert-count", type=int, default=1000000)
    parser.add_argument("--min-delete-fraction", type=float, default=0.6)
    parser.add_argument("--expected-background-foreground-rows", type=int, default=80)
    parser.add_argument("--expected-background-query-limit", type=int, default=1000)
    parser.add_argument("--expected-background-case-count", type=int, default=200)
    parser.add_argument("--ppt-figures-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_dir = resolve(args.baseline_dir)
    v3_replay_dir = resolve(args.v3_replay_dir)
    out = resolve(args.out_dir)
    explicitmat_dir = resolve(args.explicitmat_dir) if args.explicitmat_dir else None
    background_dir = resolve(args.background_dir) if args.background_dir else None
    pq_replay_dir = resolve(args.pq_replay_dir) if args.pq_replay_dir else None

    selected = load_selected(v3_replay_dir)
    space_rows = read_jsonl(v3_replay_dir / "index_space_audit.jsonl")
    repack_rows = read_jsonl(v3_replay_dir / "raw" / "repack_super32k.jsonl")
    background = background_evidence_metrics(
        background_dir,
        v3_replay_dir,
        selected,
        selected_v3_prefixes(selected),
        expected_foreground_rows=args.expected_background_foreground_rows,
        expected_query_limit=args.expected_background_query_limit,
        expected_case_count=args.expected_background_case_count,
    )
    explicit_metrics = explicit_evidence_metrics(
        explicitmat_dir,
        baseline_dir,
        args.expected_cycles,
        args.expected_delete_count,
        args.min_delete_fraction,
        args.expected_zero_insert_count,
    )
    read_metrics = read_invariant_metrics(selected)
    layout_metrics = repack_layout_metrics(repack_rows)
    pq_compare = build_pq_compare(baseline_dir, selected, pq_replay_dir)

    out.mkdir(parents=True, exist_ok=True)
    selected_by_row_index = {
        int(row["row_index"]): row
        for row in selected
        if row.get("row_index") not in (None, "")
    }
    for stem in ["targeted_latency_profile", "optimized_dynamic_update_results"]:
        rows = [payload(row) for row in read_jsonl(v3_replay_dir / f"{stem}.jsonl")]
        rows = annotate_serving_rows(rows, selected_by_row_index)
        write_jsonl(out / f"{stem}.jsonl", rows)
        write_csv(out / f"{stem}.csv", rows)
    for rel in [
        "index_space_audit.jsonl",
        "index_space_audit.csv",
    ]:
        copy_small_artifact(v3_replay_dir / rel, out / rel)
    write_jsonl(out / "pq_drift_strategy_compare.jsonl", pq_compare)
    write_csv(out / "pq_drift_strategy_compare.csv", pq_compare)
    figures = plot_figures(out, selected, space_rows, background)
    if args.ppt_figures_dir:
        ppt_dir = resolve(args.ppt_figures_dir)
        ppt_dir.mkdir(parents=True, exist_ok=True)
        for rel in figures:
            src = ROOT / rel
            shutil.copy2(src, ppt_dir / src.name)
    render_reports(
        out,
        selected,
        pq_compare,
        space_rows,
        background,
        explicit_metrics,
        read_metrics,
        layout_metrics,
        figures,
        args.expected_selected,
        args.expected_pq_rows,
        args.expected_cycles,
        args.expected_query_count,
    )
    print(json.dumps({"out_dir": str(out), "selected_rows": len(selected), "figures": figures}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
