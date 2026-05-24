#!/usr/bin/env python3
"""Generate PPT-ready figures from Supersector32K ARIS evidence.

The script is evidence-only: it reads committed/raw CSV+JSONL artifacts and
writes lightweight figures, summaries, and a manifest for the graduation deck.
Large repacked indexes remain outside git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPER = ROOT / "experiments" / "supersector32k_packing_aris_20260523"
PREV = ROOT / "experiments" / "optimized_dynamic_update_pq_drift_aris_20260523"
MAIN = ROOT / "experiments" / "pq_drift_1m_aris_main_20260522"
OLD = ROOT / "experiments" / "r116_suite_pq16_aris_20260520_072453"
OUT = ROOT / "experiments" / "ppt_supersector32k_refresh_20260524"
FIGURES = OUT / "figures"
PPT_FIGURES = ROOT / "PPT" / "graduation-ppt" / "figures"
ALLOW_OVERWRITE_PPT_FIGURES = False

BUCKET_ORDER = ["u1e-03", "u3e-03", "u1e-02", "u5e-02", "u1e-01", "u25", "u30", "u50", "u75", "u100"]
BUCKET_LABELS = {
    "u1e-03": "0.1%",
    "u3e-03": "0.3%",
    "u1e-02": "1%",
    "u5e-02": "5%",
    "u1e-01": "10%",
    "u25": "25%",
    "u30": "30%",
    "u50": "50%",
    "u75": "75%",
    "u100": "100%",
}
ROUTE_COLORS = {
    "prefilter": "#0f766e",
    "graph": "#7c3aed",
    "mixed": "#ea580c",
    "auto": "#64748b",
    "": "#64748b",
}
EXPECTED_VARIANTS = {"retrain_each_cycle", "no_retrain_across_cycles"}
EXPECTED_SELECTOR_TYPES = {"intersect", "range"}
EXPECTED_DYNAMIC_ROUTES = set(ROUTE_COLORS)
EXPECTED_CYCLES = {1, 2, 3, 4, 5}
SELECTOR_MARKERS = {"intersect": "o", "range": "s"}
PPT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def require_fnum(row: dict[str, Any], key: str, context: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"missing required numeric field {key} in {context}: {row}")
    return float(value)


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def repack_read_page_bytes(row: dict[str, Any]) -> int:
    value = row.get("layout_read_page_bytes", row.get("read_page_bytes"))
    return 0 if value in (None, "") else int(value)


def shorten_case(case_id: str) -> str:
    text = (
        case_id.replace("cycle", "c")
        .replace("_no_retrain_across_cycles_", "\nnoRT ")
        .replace("_retrain_each_cycle_", "\nRT ")
        .replace("intersect_", "int ")
        .replace("range_", "rng ")
    )
    return text


def save_figure(fig: plt.Figure, name: str, copy_to_ppt: bool = True) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for suffix in ("png", "pdf"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, dpi=220)
        written.append(str(path.relative_to(ROOT)))
    if copy_to_ppt:
        PPT_FIGURES.mkdir(parents=True, exist_ok=True)
        dest = PPT_FIGURES / f"{name}.png"
        if dest.exists() and not ALLOW_OVERWRITE_PPT_FIGURES:
            raise FileExistsError(
                f"{dest.relative_to(ROOT)} already exists; rerun with --allow-overwrite-ppt-figures "
                "only after confirming it is generated output"
            )
        shutil.copy2(FIGURES / f"{name}.png", dest)
        written.append(str(dest.relative_to(ROOT)))
    plt.close(fig)
    return written


def require_inputs() -> None:
    required = [
        SUPER / "optimized_dynamic_update_results.jsonl",
        SUPER / "targeted_latency_profile.jsonl",
        SUPER / "pq_drift_strategy_compare.jsonl",
        SUPER / "index_space_audit.jsonl",
        SUPER / "raw" / "repack_super32k.jsonl",
        PREV / "raw" / "phase3_pq_unmatched_targeted_rerun.jsonl",
        MAIN / "raw" / "phaseC_penalty.jsonl",
        MAIN / "raw" / "phaseC_delete_steps.jsonl",
        MAIN / "raw" / "phaseC_no_retrain_cycles.jsonl",
        MAIN / "raw" / "phaseD_pq_core_sweep.jsonl",
        ROOT / "experiments" / "flat_small_search" / "table.csv",
        ROOT / "experiments" / "label_space_ratio" / "table.csv",
        OLD / "exp3_search_during_insert" / "table.csv",
        OLD / "pq16_pq_residency_compare.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Supersector32K evidence: " + ", ".join(missing))


def load_selected() -> list[dict[str, Any]]:
    rows = read_jsonl(SUPER / "optimized_dynamic_update_results.jsonl")
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 optimized dynamic selected rows, found {len(rows)}")
    case_ids = {row.get("case_id") for row in rows}
    if None in case_ids or "" in case_ids or len(case_ids) != 200:
        raise RuntimeError(f"expected 200 non-empty unique case_id values, found {len(case_ids)}")
    for idx, row in enumerate(rows):
        context = f"optimized_dynamic_update_results row {idx}"
        fill_case_fields(row, context)
        for field in ["variant", "selector_type", "bucket", "route", "actual_route", "cycle_idx"]:
            if row.get(field) in (None, ""):
                raise RuntimeError(f"missing required categorical field {field} in {context}: {row}")
        if row["variant"] not in EXPECTED_VARIANTS:
            raise RuntimeError(f"unexpected variant in {context}: {row['variant']}")
        if row["selector_type"] not in EXPECTED_SELECTOR_TYPES:
            raise RuntimeError(f"unexpected selector_type in {context}: {row['selector_type']}")
        if row["bucket"] not in BUCKET_ORDER:
            raise RuntimeError(f"unexpected bucket in {context}: {row['bucket']}")
        if row["route"] not in EXPECTED_DYNAMIC_ROUTES:
            raise RuntimeError(f"unexpected route in {context}: {row['route']}")
        if row["actual_route"] not in EXPECTED_DYNAMIC_ROUTES:
            raise RuntimeError(f"unexpected actual_route in {context}: {row['actual_route']}")
        row["avg_latency_ms"] = require_fnum(row, "avg_latency_us", context) / 1000.0
        row["p95_latency_ms"] = require_fnum(row, "p95_latency_us", context) / 1000.0
        row["recall"] = require_fnum(row, "recall@10", context)
        row["cycle_idx_i"] = int(row["cycle_idx"])
        if row["cycle_idx_i"] not in EXPECTED_CYCLES:
            raise RuntimeError(f"unexpected cycle_idx in {context}: {row['cycle_idx']}")
    return rows


def load_v3_pq_l420_row() -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(SUPER / "pq_drift_strategy_compare.jsonl")
        if row.get("case_id") == "cycle05_no_retrain_across_cycles_range_u30"
        and int(row.get("search_l") or row.get("chosen_L") or 0) == 420
        and row.get("layout") == "supersector32k"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one v3 L420 PQ row, found {len(rows)}")
    return rows[0]


CASE_ID_RE = re.compile(r"^cycle(?P<cycle_idx>\d{2})_(?P<variant>retrain_each_cycle|no_retrain_across_cycles)_(?P<selector_type>intersect|range)_(?P<bucket>.+)$")


def fill_case_fields(row: dict[str, Any], context: str) -> None:
    case_id = row.get("case_id")
    if not case_id:
        raise RuntimeError(f"missing case_id in {context}: {row}")
    match = CASE_ID_RE.match(str(case_id))
    if not match:
        raise RuntimeError(f"case_id does not match expected dynamic format in {context}: {case_id}")
    derived = match.groupdict()
    derived["cycle_idx"] = str(int(derived["cycle_idx"]))
    for field in ["variant", "selector_type", "bucket"]:
        if row.get(field) in (None, ""):
            row[field] = derived[field]
        elif str(row[field]) != derived[field]:
            raise RuntimeError(f"{field} conflicts with case_id in {context}: {row[field]} != {derived[field]}")
    if row.get("cycle_idx") in (None, ""):
        row["cycle_idx"] = int(derived["cycle_idx"])
    elif int(row["cycle_idx"]) != int(derived["cycle_idx"]):
        raise RuntimeError(f"cycle_idx conflicts with case_id in {context}: {row['cycle_idx']} != {derived['cycle_idx']}")


def summarize_metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    space = read_jsonl(SUPER / "index_space_audit.jsonl")[-1]
    pq = load_v3_pq_l420_row()
    profile = read_jsonl(SUPER / "targeted_latency_profile.jsonl")
    repack = read_jsonl(SUPER / "raw" / "repack_super32k.jsonl")
    current_repack = [
        row
        for row in repack
        if row.get("layout_version") == 3
        and row.get("layout_variant") == "page_aware_slots"
        and repack_read_page_bytes(row) == 4096
    ]
    if not current_repack:
        raise RuntimeError("missing current v3 page-aware 4KB repack evidence")
    replacements = [row for row in selected if row.get("result_source") == "targeted_replacement_super32k_v3"]
    return {
        "selected_rows": len(selected),
        "unique_cases": len({row.get("case_id") for row in selected}),
        "min_recall": min((row["recall"] for row in selected), default=0.0),
        "max_avg_latency_ms": max((row["avg_latency_ms"] for row in selected), default=0.0),
        "max_p95_latency_ms": max((row["p95_latency_ms"] for row in selected), default=0.0),
        "avg_latency_ge_10ms": sum(row["avg_latency_ms"] >= 10.0 for row in selected),
        "p95_latency_ge_10ms": sum(row["p95_latency_ms"] >= 10.0 for row in selected),
        "recall_lt_98": sum(row["recall"] < 98.0 for row in selected),
        "replacement_cases": [row.get("case_id") for row in replacements],
        "replacement_count": len(replacements),
        "routes": dict(Counter(str(row.get("route") or row.get("configured_route") or "") for row in selected)),
        "actual_routes": dict(Counter(str(row.get("actual_route") or "") for row in selected)),
        "space": space,
        "pq_l420": pq,
        "targeted_profile_rows": len(profile),
        "repack_rows": len(repack),
        "current_v3_repack_rows": len(current_repack),
        "repack_max_elapsed_s": max((fnum(row, "repack_elapsed_s") for row in current_repack), default=0.0),
        "read_page_bytes": sorted({repack_read_page_bytes(row) for row in current_repack}),
        "straddling_slots_per_block": sorted({row.get("straddling_slots_per_block") for row in current_repack}),
        "avg_4k_pages_per_record": sorted({row.get("avg_4k_pages_per_record") for row in current_repack}),
    }


def plot_dynamic_scatter(selected: list[dict[str, Any]]) -> list[str]:
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    plotted_count = 0
    for selector, marker in SELECTOR_MARKERS.items():
        for route, color in ROUTE_COLORS.items():
            pts = [
                row
                for row in selected
                if str(row.get("selector_type")) == selector
                and str(row.get("actual_route") or row.get("route") or "") == route
            ]
            if not pts:
                continue
            plotted_count += len(pts)
            ax.scatter(
                [row["avg_latency_ms"] for row in pts],
                [row["recall"] for row in pts],
                s=36,
                marker=marker,
                color=color,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.45,
                label=f"{selector} / {route}",
            )
    if plotted_count != len(selected):
        raise RuntimeError(f"dynamic scatter plotted {plotted_count} rows, expected {len(selected)}")
    ax.axvline(10.0, color="#dc2626", linestyle="--", linewidth=1.2, label="10 ms avg budget")
    ax.axhline(98.0, color="#111827", linestyle=":", linewidth=1.1, label="98% recall")
    ax.set_xlabel("Avg latency (ms), v3 packed serving snapshot")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title("Supersector32K v3: all 200 selected dynamic-update points")
    ax.set_xlim(left=0, right=max(10.8, max(row["avg_latency_ms"] for row in selected) * 1.08))
    ax.set_ylim(97.75, 100.15)
    ax.grid(alpha=0.24)
    ax.legend(ncol=3, loc="lower left")
    worst = max(selected, key=lambda row: row["avg_latency_ms"])
    ax.annotate(
        f"max avg {worst['avg_latency_ms']:.3f} ms",
        xy=(worst["avg_latency_ms"], worst["recall"]),
        xytext=(-120, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=10,
    )
    return save_figure(fig, "ppt_v3_dynamic_recall_latency")


def plot_cycle_quality(selected: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row.get("variant")), int(row.get("cycle_idx_i") or 0))].append(row)
    variants = ["retrain_each_cycle", "no_retrain_across_cycles"]
    labels = {"retrain_each_cycle": "retrain", "no_retrain_across_cycles": "no-retrain"}
    colors = {"retrain_each_cycle": "#2563eb", "no_retrain_across_cycles": "#f59e0b"}
    cycles = sorted({cycle for _, cycle in grouped})
    if set(cycles) != EXPECTED_CYCLES:
        raise RuntimeError(f"cycle-quality plot expected cycles {sorted(EXPECTED_CYCLES)}, found {cycles}")
    for variant in variants:
        for cycle in cycles:
            count = len(grouped.get((variant, cycle), []))
            if count != 20:
                raise RuntimeError(f"expected 20 rows for {variant} cycle {cycle}, found {count}")
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    for variant in variants:
        min_recall = []
        max_avg = []
        for cycle in cycles:
            rows = grouped.get((variant, cycle), [])
            min_recall.append(min((row["recall"] for row in rows), default=np.nan))
            max_avg.append(max((row["avg_latency_ms"] for row in rows), default=np.nan))
        axes[0].plot(cycles, min_recall, marker="o", linewidth=2.2, color=colors[variant], label=labels[variant])
        axes[1].plot(cycles, max_avg, marker="o", linewidth=2.2, color=colors[variant], label=labels[variant])
    axes[0].axhline(98.0, color="#111827", linestyle="--", linewidth=1.1)
    axes[1].axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.1)
    axes[0].set_title("Worst recall per cycle")
    axes[1].set_title("Worst avg latency per cycle")
    axes[0].set_ylabel("Min recall@10 (%)")
    axes[1].set_ylabel("Max avg latency (ms)")
    for ax in axes:
        ax.set_xlabel("Delete/insert cycle")
        ax.set_xticks(cycles)
        ax.grid(axis="y", alpha=0.26)
        ax.legend(loc="best")
    return save_figure(fig, "ppt_v3_cycle_quality")


def plot_targeted_replacements(selected: list[dict[str, Any]]) -> list[str]:
    replacements = [row for row in selected if row.get("result_source") == "targeted_replacement_super32k_v3"]
    replacements = sorted(replacements, key=lambda row: str(row.get("case_id")))
    if len(replacements) != 7:
        raise RuntimeError(f"expected 7 targeted v3 replacement rows, found {len(replacements)}")
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(9.8, 5.1), constrained_layout=True)
    x = np.arange(len(replacements))
    width = 0.22
    before_avg = [
        require_fnum(row, "baseline_full_avg_latency_us", f"targeted replacement {row.get('case_id')}") / 1000.0
        for row in replacements
    ]
    after_avg = [row["avg_latency_ms"] for row in replacements]
    after_p95 = [row["p95_latency_ms"] for row in replacements]
    ax.bar(x - width, before_avg, width, label="before avg", color="#94a3b8")
    ax.bar(x, after_avg, width, label="after avg", color="#0f766e")
    ax.bar(x + width, after_p95, width, label="after p95", color="#f59e0b")
    ax.axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.1, label="10 ms")
    ax.set_xticks(x, [shorten_case(str(row.get("case_id"))) for row in replacements], rotation=35, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Targeted v3 fixes for the 7 previous avg-latency failures")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(ncol=4, loc="upper right")
    return save_figure(fig, "ppt_v3_targeted_latency_before_after")


def plot_pq_strategy() -> list[str]:
    penalty_rows = read_jsonl(MAIN / "raw" / "phaseC_penalty.jsonl")
    target_penalty = [
        row
        for row in penalty_rows
        if row.get("cycle_idx") == 5 and row.get("selector_type") == "range" and row.get("bucket") == "u30"
    ]
    if len(target_penalty) != 1:
        raise RuntimeError(f"expected exactly one phaseC cycle5 range-u30 PQ penalty row, found {len(target_penalty)}")
    ref = target_penalty[0]
    v3 = load_v3_pq_l420_row()
    prev_candidates = read_jsonl(PREV / "raw" / "phase3_pq_unmatched_targeted_rerun.jsonl")
    prev_l420_rows = [
        row
        for row in prev_candidates
        if row.get("case_id") == "cycle05_no_retrain_across_cycles_range_u30"
        and int(row.get("rerun_L") or row.get("search_l") or 0) == 420
        and row.get("route") == "prefilter"
    ]
    prev_l420 = prev_l420_rows[0] if len(prev_l420_rows) == 1 else None
    if prev_l420 is None:
        raise RuntimeError(f"expected exactly one previous phase3 range-u30 L420 targeted PQ row, found {len(prev_l420_rows)}")
    for field in ["reference_recall", "reference_avg_latency_ms", "no_retrain_recall", "no_retrain_avg_latency_ms"]:
        if ref.get(field) in (None, ""):
            raise RuntimeError(f"missing required PQ penalty field {field}")
    for field in ["recall@10", "avg_latency_us"]:
        if prev_l420.get(field) in (None, ""):
            raise RuntimeError(f"missing required previous L420 field {field}")
        if v3.get(field) in (None, ""):
            raise RuntimeError(f"missing required v3 L420 field {field}")
    bars = [
        ("reference", fnum(ref, "reference_recall"), fnum(ref, "reference_avg_latency_ms")),
        ("old no-retrain", fnum(ref, "no_retrain_recall"), fnum(ref, "no_retrain_avg_latency_ms")),
        ("prev L420", fnum(prev_l420, "recall@10"), fnum(prev_l420, "avg_latency_us") / 1000.0),
        ("v3 L420", fnum(v3, "recall@10"), fnum(v3, "avg_latency_us") / 1000.0),
    ]
    plt.rcParams.update(PPT_RC)
    fig, ax1 = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    x = np.arange(len(bars))
    recall = [item[1] for item in bars]
    latency = [item[2] for item in bars]
    colors = ["#2563eb", "#94a3b8", "#16a34a", "#0f766e"]
    ax1.bar(x, recall, color=colors, width=0.54)
    ax1.axhline(99.41, color="#dc2626", linestyle="--", linewidth=1.1, label="matched target 99.41")
    ax1.set_ylabel("Recall@10 (%)")
    ax1.set_ylim(98.5, 99.75)
    ax1.set_xticks(x, [item[0] for item in bars])
    ax1.set_title("PQ drift unmatched point: range-u30 cycle5")
    ax1.grid(axis="y", alpha=0.25)
    for xi, value in zip(x, recall):
        ax1.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, latency, color="#111827", marker="o", linewidth=1.8, label="avg latency")
    ax2.set_ylabel("Avg latency (ms)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    return save_figure(fig, "ppt_v3_pq_drift_strategy")


def plot_space_components() -> list[str]:
    row = read_jsonl(SUPER / "index_space_audit.jsonl")[-1]
    raw = require_fnum(row, "raw_vector_file_bytes", "index_space_audit")
    components = [
        ("disk index", require_fnum(row, "disk_index", "index_space_audit")),
        ("pq codes", require_fnum(row, "pq_codes", "index_space_audit")),
        ("pq pivots", require_fnum(row, "pq_pivots", "index_space_audit")),
        ("labels sidecar", require_fnum(row, "labels_sidecar", "index_space_audit")),
        (
            "tags/meta",
            require_fnum(row, "disk_tags", "index_space_audit")
            + require_fnum(row, "hybrid_meta", "index_space_audit"),
        ),
    ]
    strict = require_fnum(row, "strict_serving_bytes", "index_space_audit")
    excess = strict - raw
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    left = axes[0]
    running = 0.0
    for name, bytes_value in components:
        value = mib(bytes_value)
        left.bar(["strict serving"], [value], bottom=running, label=f"{name} {value:.1f} MiB")
        running += value
    raw_mib = mib(raw)
    left.axhline(raw_mib, color="#111827", linestyle="--", linewidth=1.1, label=f"raw vectors {raw_mib:.1f} MiB")
    left.set_ylabel("MiB")
    left.set_title("Strict serving footprint")
    left.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    right = axes[1]
    ratios = [
        require_fnum(row, "strict_total_over_raw_x", "index_space_audit"),
        require_fnum(row, "strict_excess_over_raw_x", "index_space_audit"),
    ]
    right.bar(["total/raw", "excess/raw"], ratios, color=["#64748b", "#0f766e"], width=0.55)
    right.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.1)
    right.set_ylim(0, max(2.1, max(ratios) * 1.15))
    right.set_ylabel("x raw vector bytes")
    right.set_title("Acceptance accounting")
    for xi, value in enumerate(ratios):
        right.text(xi, value + 0.04, f"{value:.3f}x", ha="center", fontsize=10)
    return save_figure(fig, "ppt_v3_space_components")


def plot_read_granularity() -> list[str]:
    repack = read_jsonl(SUPER / "raw" / "repack_super32k.jsonl")
    current = [
        row
        for row in repack
        if row.get("layout_version") == 3
        and row.get("layout_variant") == "page_aware_slots"
        and repack_read_page_bytes(row) == 4096
    ]
    if not current:
        raise RuntimeError("missing v3 page-aware 4KB repack row for read-granularity plot")
    row = current[-1]
    for field in ["layout_nodes_per_block", "straddling_slots_per_block", "avg_4k_pages_per_record"]:
        if row.get(field) in (None, ""):
            raise RuntimeError(f"missing required repack field {field}")
    nodes = fnum(row, "layout_nodes_per_block")
    straddling = fnum(row, "straddling_slots_per_block")
    one_page = max(nodes - straddling, 0)
    avg_pages = fnum(row, "avg_4k_pages_per_record")
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), constrained_layout=True)
    axes[0].bar(["1 page", "2 pages"], [one_page, straddling], color=["#0f766e", "#f59e0b"], width=0.55)
    axes[0].set_ylabel("Slots per 32KB block")
    axes[0].set_title("Page-aware slots")
    axes[0].text(0, one_page + 0.6, f"{one_page:.0f}", ha="center")
    axes[0].text(1, straddling + 0.6, f"{straddling:.0f}", ha="center")
    axes[1].bar(["read page", "avg pages/node"], [4096, avg_pages], color=["#2563eb", "#7c3aed"], width=0.55)
    axes[1].set_yscale("log")
    axes[1].set_title("4KB random-read primitive retained")
    axes[1].set_ylabel("bytes / page count (log scale)")
    axes[1].text(0, 4096 * 1.16, "4096 B", ha="center")
    axes[1].text(1, avg_pages * 1.16, f"{avg_pages:.3f}", ha="center")
    return save_figure(fig, "ppt_v3_read_granularity")


def plot_maintenance_window() -> list[str]:
    deletes = read_jsonl(MAIN / "raw" / "phaseC_delete_steps.jsonl")
    inserts = read_jsonl(MAIN / "raw" / "phaseC_no_retrain_cycles.jsonl")
    core = read_jsonl(MAIN / "raw" / "phaseD_pq_core_sweep.jsonl")
    if len(deletes) != 5:
        raise RuntimeError(f"expected 5 delete-step rows, found {len(deletes)}")
    if len(inserts) != 5:
        raise RuntimeError(f"expected 5 no-retrain insert-cycle rows, found {len(inserts)}")
    delete_elapsed_values = [require_fnum(row, "delete_elapsed_s", "phaseC_delete_steps") for row in deletes]
    merge_elapsed_values = [
        require_fnum(row, "merge_elapsed_s", "phaseC delete/insert merge")
        for row in [*deletes, *inserts]
    ]
    core16 = next((row for row in core if row.get("core_count") == 16), None)
    if core16 is None:
        raise RuntimeError("missing phaseD 16-core PQ sweep row")
    pq_train_recode_s = (
        require_fnum(core16, "pq_train_wall_s", "phaseD core16")
        + require_fnum(core16, "pq_recode_wall_s", "phaseD core16")
    )
    full_rebuild_s = require_fnum(core16, "build_wall_s", "phaseD core16")
    repack = read_jsonl(SUPER / "raw" / "repack_super32k.jsonl")
    repack_elapsed_values = [
        fnum(row, "repack_elapsed_s")
        for row in repack
        if row.get("layout_version") == 3
        and row.get("layout_variant") == "page_aware_slots"
        and repack_read_page_bytes(row) == 4096
        and row.get("repack_elapsed_s") not in (None, "")
    ]
    if not repack_elapsed_values:
        raise RuntimeError("missing v3 repack_elapsed_s evidence for maintenance-window plot")
    repack_elapsed = max(repack_elapsed_values)
    bars = [
        ("delete mark\nAPI max", max(delete_elapsed_values)),
        ("merge\nmax", max(merge_elapsed_values)),
        ("PQ train\n+ recode", pq_train_recode_s),
        ("full rebuild\n16 cores", full_rebuild_s),
        ("v3 repack\nbackground", repack_elapsed),
    ]
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    x = np.arange(len(bars))
    values = [item[1] for item in bars]
    colors = ["#0f766e", "#16a34a", "#2563eb", "#dc2626", "#7c3aed"]
    ax.bar(x, values, color=colors, width=0.58)
    ax.axhline(180.0, color="#111827", linestyle="--", linewidth=1.1, label="3 min window")
    ax.set_xticks(x, [item[0] for item in bars])
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Mixed evidence: reused update timings plus v3 background repack")
    ax.grid(axis="y", alpha=0.26)
    ax.legend(loc="upper left")
    for xi, value in zip(x, values):
        ax.text(xi, value + max(values) * 0.025, f"{value:.1f}s", ha="center", fontsize=9)
    return save_figure(fig, "ppt_mixed_maintenance_window")


def plot_small_flat() -> list[str]:
    rows = read_csv(ROOT / "experiments" / "flat_small_search" / "table.csv")
    if not rows:
        raise RuntimeError("missing flat-small-search rows")
    points = [int(row["points"]) for row in rows]
    latency_ms = [float(row["avg_latency_us"]) / 1000.0 for row in rows]
    rss_mib = [float(row["process_max_rss_kb"]) / 1024.0 for row in rows]
    plt.rcParams.update(PPT_RC)
    fig, ax1 = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
    ax1.plot(points, latency_ms, marker="o", color="#2563eb", linewidth=2.2, label="avg latency")
    ax1.axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Live points before disk graph threshold")
    ax1.set_ylabel("Avg latency (ms)")
    ax2 = ax1.twinx()
    ax2.plot(points, rss_mib, marker="s", color="#0f766e", linewidth=2.2, label="max RSS")
    ax2.axhline(30.0, color="#111827", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Process max RSS (MiB)")
    ax1.set_title("Reused evidence: flat exact-search stage before disk threshold")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    return save_figure(fig, "ppt_reused_small_flat_rss_latency_qps")


def plot_label_sidecar() -> list[str]:
    rows = read_csv(ROOT / "experiments" / "label_space_ratio" / "table.csv")
    sift = next((row for row in rows if row["dataset"] == "SIFT1M/r116"), None)
    if sift is None:
        raise RuntimeError("missing SIFT1M/r116 label-space row")
    values = [float(sift["original_mib"]), float(sift["processed_mib"]), 0.0]
    labels = ["source spmat", "sidecar densebit", "main index labels"]
    colors = ["#64748b", "#0f766e", "#dc2626"]
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.bar(labels, values, color=colors, width=0.56)
    ax.set_ylabel("MiB")
    ax.set_title("Reused evidence: labels stay outside the main disk index")
    ax.grid(axis="y", alpha=0.24)
    for xi, value in enumerate(values):
        ax.text(xi, max(value, 0.02) + 0.25, f"{value:.2f} MiB", ha="center", fontsize=10)
    ax.text(0.5, 0.82, f"sidecar = {float(sift['processed_over_original_percent']):.2f}% of spmat", transform=ax.transAxes, ha="center")
    return save_figure(fig, "ppt_reused_label_sidecar_space")


def plot_insert_during_search() -> list[str]:
    rows = read_csv(OLD / "exp3_search_during_insert" / "table.csv")
    if not rows:
        raise RuntimeError("missing insert-during-search rows")
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["insert_threads"]))].append(float(row["avg_latency_us"]) / 1000.0)
    xs = sorted(grouped)
    p50 = [float(np.percentile(grouped[x], 50)) for x in xs]
    p95 = [float(np.percentile(grouped[x], 95)) for x in xs]
    maxv = [max(grouped[x]) for x in xs]
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(xs, p50, marker="o", linewidth=2.0, label="p50 workload avg", color="#2563eb")
    ax.plot(xs, p95, marker="s", linewidth=2.0, label="p95 workload avg", color="#f59e0b")
    ax.plot(xs, maxv, marker="^", linewidth=2.0, label="max workload avg", color="#dc2626")
    ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms")
    ax.set_xlabel("Concurrent insert threads")
    ax.set_ylabel("Avg latency across selectivity workloads (ms)")
    ax.set_title("Reused evidence: insert path remains v1; v3 is serving snapshot")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="upper left")
    return save_figure(fig, "ppt_reused_insert_during_search")


def plot_pq_memory_rss() -> list[str]:
    rows = read_csv(OLD / "pq16_pq_residency_compare.csv")
    rows = [row for row in rows if row.get("mode") == "pq_memory"]
    if not rows:
        raise RuntimeError("missing PQ-memory residency rows")
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for selector, color, marker in [("intersect", "#2563eb", "o"), ("range", "#dc2626", "s")]:
        by_bucket = {row["bucket"]: row for row in rows if row["selector_type"] == selector}
        ordered = [by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in by_bucket]
        x = np.arange(len(ordered))
        y = [float(row["adjusted_rss_mib"]) for row in ordered]
        ax.plot(x, y, marker=marker, color=color, linewidth=2.2, label=selector)
    ax.axhline(30.0, color="#111827", linestyle="--", linewidth=1.1, label="30 MiB")
    ax.set_xticks(np.arange(len(BUCKET_ORDER)), [BUCKET_LABELS[b] for b in BUCKET_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Adjusted RSS (MiB)")
    ax.set_xlabel("Selectivity")
    ax.set_title("Reused evidence: PQ16 resident-code query memory")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    return save_figure(fig, "ppt_reused_pq_memory_rss")


def write_summary(metrics: dict[str, Any], figure_paths: list[str]) -> None:
    rows = [
        {"metric": "selected_rows", "value": metrics["selected_rows"]},
        {"metric": "min_recall@10", "value": f"{metrics['min_recall']:.6f}"},
        {"metric": "max_avg_latency_ms", "value": f"{metrics['max_avg_latency_ms']:.6f}"},
        {"metric": "max_p95_latency_ms", "value": f"{metrics['max_p95_latency_ms']:.6f}"},
        {"metric": "avg_latency_ge_10ms", "value": metrics["avg_latency_ge_10ms"]},
        {"metric": "p95_latency_ge_10ms", "value": metrics["p95_latency_ge_10ms"]},
        {"metric": "strict_total_over_raw_x", "value": f"{metrics['space']['strict_total_over_raw_x']:.6f}"},
        {"metric": "strict_excess_over_raw_x", "value": f"{metrics['space']['strict_excess_over_raw_x']:.6f}"},
        {"metric": "pq_l420_recall@10", "value": f"{metrics['pq_l420']['recall@10']:.6f}"},
        {"metric": "read_page_bytes", "value": metrics["read_page_bytes"]},
        {"metric": "straddling_slots_per_block", "value": metrics["straddling_slots_per_block"]},
        {"metric": "avg_4k_pages_per_record", "value": metrics["avg_4k_pages_per_record"]},
    ]
    write_csv(OUT / "ppt_refresh_metrics.csv", rows)
    write_json(OUT / "ppt_refresh_claim_registry.json", metrics)
    write_json(OUT / "figure_manifest.json", {"figures": figure_paths})
    summary = [
        "# PPT Supersector32K Refresh Summary",
        "",
        f"- Dynamic selected rows: {metrics['selected_rows']} / unique cases {metrics['unique_cases']}.",
        f"- Recall/latency gate: min recall {metrics['min_recall']:.3f}, max avg latency {metrics['max_avg_latency_ms']:.3f} ms.",
        f"- P95 caveat: {metrics['p95_latency_ge_10ms']} selected points have p95 latency >= 10 ms; max p95 {metrics['max_p95_latency_ms']:.3f} ms.",
        f"- Space strict total/raw: {metrics['space']['strict_total_over_raw_x']:.6f}x.",
        f"- Space strict excess/raw: {metrics['space']['strict_excess_over_raw_x']:.6f}x.",
        f"- PQ drift L420 recall: {metrics['pq_l420']['recall@10']:.3f}.",
        "- The insert-during-search and flat-threshold figures use the existing update-path evidence; v3 is a serving snapshot produced after merge/repack.",
        "",
        "## Figures",
    ]
    summary.extend(f"- `{path}`" for path in figure_paths)
    (OUT / "ppt_refresh_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def generate() -> dict[str, Any]:
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = load_selected()
    metrics = summarize_metrics(selected)
    figures: list[str] = []
    figures += plot_dynamic_scatter(selected)
    figures += plot_cycle_quality(selected)
    figures += plot_targeted_replacements(selected)
    figures += plot_pq_strategy()
    figures += plot_space_components()
    figures += plot_read_granularity()
    figures += plot_maintenance_window()
    figures += plot_small_flat()
    figures += plot_label_sidecar()
    figures += plot_insert_during_search()
    figures += plot_pq_memory_rss()
    write_summary(metrics, figures)
    return {"metrics": metrics, "figures": figures, "out": str(OUT.relative_to(ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    parser.add_argument(
        "--allow-overwrite-ppt-figures",
        action="store_true",
        help="allow overwriting generated figure files under PPT/graduation-ppt/figures",
    )
    args = parser.parse_args()
    global ALLOW_OVERWRITE_PPT_FIGURES
    ALLOW_OVERWRITE_PPT_FIGURES = args.allow_overwrite_ppt_figures
    result = generate()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"wrote {len(result['figures'])} figure artifacts under {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
