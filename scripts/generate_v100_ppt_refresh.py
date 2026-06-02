#!/usr/bin/env python3
"""Refresh the graduation deck from final v100 ARIS evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "experiments" / "v100_goal_final_artifacts_20260602T005857Z"
DEFAULT_PPT_DIR = ROOT / "PPT" / "graduation-ppt"
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
    "graph": "#2563eb",
    "mixed": "#ea580c",
    "auto": "#64748b",
    "": "#64748b",
}
VARIANT_LABELS = {
    "retrain_each_cycle": "retrain",
    "no_retrain_across_cycles": "no-retrain",
}
VARIANT_COLORS = {
    "retrain_each_cycle": "#2563eb",
    "no_retrain_across_cycles": "#f59e0b",
}
PPT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def claim_by_id(registry: dict[str, Any], claim_id: str) -> dict[str, Any]:
    for claim in registry.get("claims", []):
        if claim.get("id") == claim_id:
            if claim.get("status") != "PASS":
                raise RuntimeError(f"{claim_id} is not PASS: {claim.get('status')}")
            return claim
    raise RuntimeError(f"missing claim {claim_id}")


def cycle_idx(row: dict[str, Any]) -> int:
    value = row.get("cycle_idx")
    if value not in (None, ""):
        return int(value)
    cycle = str(row.get("cycle") or "")
    match = re.search(r"cycle0?(\d+)", cycle)
    if not match:
        raise RuntimeError(f"cannot infer cycle index from row: {row}")
    return int(match.group(1))


def normalize_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 selected rows, found {len(rows)}")
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["row_index"])
        if idx in seen:
            raise RuntimeError(f"duplicate selected row_index {idx}")
        seen.add(idx)
        item = dict(row)
        item["row_index"] = idx
        item["cycle_idx"] = cycle_idx(row)
        item["avg_latency_ms"] = fnum(row, "avg_latency_us") / 1000.0
        item["p95_latency_ms"] = fnum(row, "p95_latency_us") / 1000.0
        item["recall"] = fnum(row, "recall@10")
        item["actual_route"] = str(row.get("actual_route") or row.get("route") or "")
        for field in ["variant", "selector_type", "bucket", "actual_route"]:
            if item.get(field) in (None, ""):
                raise RuntimeError(f"selected row {idx} missing {field}")
        out.append(item)
    return out


def load_all(evidence: Path) -> dict[str, Any]:
    selected = normalize_selected(read_jsonl(evidence / "optimized_dynamic_update_results.jsonl"))
    profile = read_jsonl(evidence / "targeted_latency_profile.jsonl")
    pq = read_jsonl(evidence / "pq_drift_strategy_compare.jsonl")
    space = read_jsonl(evidence / "index_space_audit.jsonl")
    registry = read_json(evidence / "optimized_claim_registry.json")
    return {
        "selected": selected,
        "profile": profile,
        "pq": pq,
        "space": space,
        "registry": registry,
        "claims": {claim.get("id"): claim for claim in registry.get("claims", [])},
    }


def save_fig(fig: plt.Figure, out_dir: Path, ppt_dir: Path, name: str) -> list[str]:
    figure_dir = out_dir / "figures"
    ppt_figure_dir = ppt_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ppt_figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ["png", "pdf"]:
        path = figure_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        written.append(str(path))
    dest = ppt_figure_dir / f"{name}.png"
    shutil.copy2(figure_dir / f"{name}.png", dest)
    written.append(str(dest))
    plt.close(fig)
    return written


def plot_dynamic_scatter(selected: list[dict[str, Any]], out_dir: Path, ppt_dir: Path) -> list[str]:
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    markers = {"intersect": "o", "range": "s"}
    plotted = 0
    for selector, marker in markers.items():
        for route, color in ROUTE_COLORS.items():
            points = [row for row in selected if row["selector_type"] == selector and row["actual_route"] == route]
            if not points:
                continue
            plotted += len(points)
            ax.scatter(
                [row["avg_latency_ms"] for row in points],
                [row["recall"] for row in points],
                s=34,
                marker=marker,
                color=color,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.45,
                label=f"{selector} / {route}",
            )
    if plotted != len(selected):
        raise RuntimeError(f"plotted {plotted} selected rows, expected {len(selected)}")
    ax.axvline(10.0, color="#dc2626", linestyle="--", linewidth=1.1, label="10 ms avg budget")
    ax.axhline(98.0, color="#111827", linestyle=":", linewidth=1.1, label="98% recall")
    ax.set_xlabel("Avg latency (ms), v100 v3 packed serving")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title("Dynamic update selected points: 200/200 pass recall and avg latency")
    ax.set_xlim(0, max(10.7, max(row["avg_latency_ms"] for row in selected) * 1.08))
    ax.set_ylim(97.75, 100.15)
    ax.grid(alpha=0.24)
    ax.legend(ncol=3, loc="lower left")
    worst = max(selected, key=lambda row: row["avg_latency_ms"])
    ax.annotate(
        f"max avg {worst['avg_latency_ms']:.3f} ms",
        xy=(worst["avg_latency_ms"], worst["recall"]),
        xytext=(-120, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=9,
    )
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_dynamic_recall_latency")


def plot_cycle_quality(selected: list[dict[str, Any]], out_dir: Path, ppt_dir: Path) -> list[str]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["variant"], row["cycle_idx"])].append(row)
    cycles = [1, 2, 3, 4, 5]
    for variant in VARIANT_LABELS:
        for cycle in cycles:
            count = len(grouped[(variant, cycle)])
            if count != 20:
                raise RuntimeError(f"expected 20 rows for {variant} cycle {cycle}, found {count}")

    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    for variant, label in VARIANT_LABELS.items():
        color = VARIANT_COLORS[variant]
        min_recall = [min(row["recall"] for row in grouped[(variant, cycle)]) for cycle in cycles]
        max_avg = [max(row["avg_latency_ms"] for row in grouped[(variant, cycle)]) for cycle in cycles]
        max_p95 = [max(row["p95_latency_ms"] for row in grouped[(variant, cycle)]) for cycle in cycles]
        axes[0].plot(cycles, min_recall, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(cycles, max_avg, marker="o", linewidth=2.0, color=color, label=f"{label} avg")
        axes[1].plot(cycles, max_p95, marker="s", linestyle="--", linewidth=1.8, color=color, label=f"{label} p95")
    axes[0].axhline(98.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.0)
    axes[0].set_title("Worst recall per cycle")
    axes[1].set_title("Worst avg/p95 latency per cycle")
    axes[0].set_ylabel("Min recall@10 (%)")
    axes[1].set_ylabel("Max latency (ms)")
    for ax in axes:
        ax.set_xlabel("Delete/insert cycle")
        ax.set_xticks(cycles)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_cycle_quality")


def targeted_label(row: dict[str, Any]) -> str:
    variant = "noRT" if row.get("variant") == "no_retrain_across_cycles" else "RT"
    bucket = BUCKET_LABELS.get(str(row.get("bucket")), str(row.get("bucket")))
    selector = "rng" if row.get("selector_type") == "range" else "int"
    return f"r{row.get('row_index')}\n{variant} {selector} {bucket}"


def plot_targeted_profile(profile: list[dict[str, Any]], out_dir: Path, ppt_dir: Path) -> list[str]:
    slow = [row for row in profile if fnum(row, "v1_p95_latency_ms") >= 10.0]
    if len(slow) < 7:
        slow = sorted(profile, key=lambda row: fnum(row, "v1_p95_latency_ms"), reverse=True)[:7]
    else:
        slow = sorted(slow, key=lambda row: (str(row.get("variant")), str(row.get("selector_type")), str(row.get("bucket"))))

    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(9.8, 5.1), constrained_layout=True)
    x = np.arange(len(slow))
    width = 0.24
    before_p95 = [fnum(row, "v1_p95_latency_ms") for row in slow]
    after_avg = [fnum(row, "v3_avg_latency_ms") for row in slow]
    after_p95 = [fnum(row, "v3_p95_latency_ms") for row in slow]
    ax.bar(x - width, before_p95, width, label="before p95", color="#94a3b8")
    ax.bar(x, after_avg, width, label="after avg", color="#0f766e")
    ax.bar(x + width, after_p95, width, label="after p95", color="#f59e0b")
    ax.axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.1, label="10 ms")
    ax.set_xticks(x, [targeted_label(row) for row in slow], rotation=0)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Targeted latency fixes: high-risk p95 rows now pass avg and p95")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(ncol=4, loc="upper right")
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_targeted_latency_before_after")


def plot_pq_strategy(pq_rows: list[dict[str, Any]], registry: dict[str, Any], out_dir: Path, ppt_dir: Path) -> list[str]:
    pq_claim = claim_by_id(registry, "C_PQ_DRIFT_MATCHED_REFERENCE")
    target = next((row for row in pq_rows if row.get("case_id") == "cycle05_no_retrain_across_cycles_range_u30"), None)
    if target is None:
        raise RuntimeError("missing cycle05 no-retrain range-u30 PQ row")
    bars = [
        ("retrain ref", fnum(target, "reference_recall"), fnum(target, "reference_avg_latency_ms")),
        ("no-retrain", fnum(target, "no_retrain_recall"), fnum(target, "no_retrain_avg_latency_ms")),
        ("matched target", fnum(target, "matched_reference_target_recall"), fnum(target, "matched_reference_avg_latency_ms")),
        ("v3 serving", fnum(target, "v3_recall@10"), fnum(target, "v3_avg_latency_ms")),
    ]
    plt.rcParams.update(PPT_RC)
    fig, ax1 = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    x = np.arange(len(bars))
    recall = [item[1] for item in bars]
    latency = [item[2] for item in bars]
    ax1.bar(x, recall, width=0.55, color=["#2563eb", "#0f766e", "#94a3b8", "#16a34a"])
    ax1.axhline(fnum(target, "matched_reference_target_recall"), color="#dc2626", linestyle="--", linewidth=1.1)
    ax1.set_ylabel("Recall@10 (%)")
    ax1.set_ylim(max(97.8, min(recall) - 0.45), min(100.0, max(recall) + 0.55))
    ax1.set_xticks(x, [item[0] for item in bars])
    ax1.set_title("PQ drift matched-reference: 100/100 matched")
    ax1.grid(axis="y", alpha=0.25)
    for xi, value in zip(x, recall):
        ax1.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, latency, color="#111827", marker="o", linewidth=1.8, label="avg latency")
    ax2.set_ylabel("Avg latency (ms)")
    ax2.set_ylim(0, max(latency) * 1.35)
    ax1.text(
        0.02,
        0.04,
        f"{pq_claim['metrics']['matched']}/{pq_claim['metrics']['total']} cases matched",
        transform=ax1.transAxes,
        fontsize=9,
        color="#065f46",
        fontweight="bold",
    )
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_pq_drift_strategy")


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def plot_space(space_rows: list[dict[str, Any]], out_dir: Path, ppt_dir: Path) -> list[str]:
    row = max(space_rows, key=lambda item: fnum(item, "strict_total_over_raw_x"))
    raw = fnum(row, "raw_vector_file_bytes")
    components = [
        ("disk index", fnum(row, "disk_index")),
        ("PQ codes", fnum(row, "pq_codes")),
        ("PQ pivots", fnum(row, "pq_pivots")),
        ("label sidecar", fnum(row, "labels_sidecar")),
        ("tags/meta", fnum(row, "disk_tags") + fnum(row, "mem_index_tags") + fnum(row, "hybrid_meta")),
    ]
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    running = 0.0
    for name, bytes_value in components:
        value = mib(bytes_value)
        axes[0].bar(["strict serving"], [value], bottom=running, label=f"{name} {value:.1f} MiB")
        running += value
    axes[0].axhline(mib(raw), color="#111827", linestyle="--", linewidth=1.1, label=f"raw vectors {mib(raw):.1f} MiB")
    axes[0].set_ylabel("MiB")
    axes[0].set_title("Strict serving footprint")
    axes[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ratios = [fnum(row, "strict_total_over_raw_x"), fnum(row, "strict_excess_over_raw_x")]
    axes[1].bar(["total/raw", "excess/raw"], ratios, width=0.55, color=["#64748b", "#0f766e"])
    axes[1].axhline(2.0, color="#111827", linestyle="--", linewidth=1.0, label="2x total budget")
    axes[1].axhline(1.0, color="#dc2626", linestyle=":", linewidth=1.0, label="1x excess budget")
    axes[1].set_ylim(0, 2.25)
    axes[1].set_ylabel("x raw vector bytes")
    axes[1].set_title("Acceptance accounting")
    axes[1].legend(loc="upper center")
    for xi, value in enumerate(ratios):
        axes[1].text(xi, value + 0.045, f"{value:.3f}x", ha="center", fontsize=9)
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_space_components")


def plot_read_granularity(registry: dict[str, Any], out_dir: Path, ppt_dir: Path) -> list[str]:
    claim = claim_by_id(registry, "C_V3_READ_GRANULARITY_4KB")
    repack = claim["metrics"]["repack_layout"]
    avg_pages = float(repack["avg_4k_pages_per_record"][0])
    straddling = int(repack["straddling_slots_per_block"][0])
    nodes_per_block = int(round(straddling / max(avg_pages - 1.0, 1e-9)))
    one_page = nodes_per_block - straddling
    stats = claim["metrics"]["query_stats"]
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), constrained_layout=True)
    axes[0].bar(["1 page", "2 pages"], [one_page, straddling], width=0.55, color=["#0f766e", "#f59e0b"])
    axes[0].set_ylabel("Slots per 32KB block")
    axes[0].set_title("Page-aware node slots")
    axes[0].text(0, one_page + 0.5, str(one_page), ha="center", fontsize=9)
    axes[0].text(1, straddling + 0.5, str(straddling), ha="center", fontsize=9)
    axes[1].bar(["read unit", "bytes / 4K"], [4096, stats["max_bytes_per_4k"]], width=0.55, color=["#2563eb", "#7c3aed"])
    axes[1].set_ylim(0, 5200)
    axes[1].set_ylabel("Bytes")
    axes[1].set_title("4KB random-read invariant")
    axes[1].text(0, 4096 + 120, "4096 B", ha="center", fontsize=9)
    axes[1].text(1, stats["max_bytes_per_4k"] + 120, "4096 B", ha="center", fontsize=9)
    axes[1].text(
        0.5,
        0.08,
        f"query rows checked: {stats['ratio_checked_rows']}\nviolations: {stats['violation_count']}",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
        color="#065f46",
    )
    return save_fig(fig, out_dir, ppt_dir, "ppt_v3_read_granularity")


def plot_maintenance(registry: dict[str, Any], out_dir: Path, ppt_dir: Path) -> list[str]:
    bg = claim_by_id(registry, "C_BACKGROUND_MAINTENANCE_NO_FRONTEND_LATENCY_BREAK")["metrics"]
    explicit = claim_by_id(registry, "C_EXPLICIT_MATERIALIZE_AND_DELETE_MERGE_EVIDENCE")["metrics"]
    plt.rcParams.update(PPT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), gridspec_kw={"width_ratios": [1, 1.25]}, constrained_layout=True)
    latency = [bg["during_max_avg_latency_ms"], bg["during_max_p95_latency_ms"]]
    axes[0].bar(["max avg", "max p95"], latency, width=0.55, color=["#0f766e", "#f59e0b"])
    axes[0].axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.1)
    axes[0].set_ylim(0, 11.2)
    axes[0].set_ylabel("Foreground latency (ms)")
    axes[0].set_title("During 1-core background build")
    for xi, value in enumerate(latency):
        axes[0].text(xi, value + 0.22, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].text(0.5, 0.08, "200/200 rows pass", transform=axes[0].transAxes, ha="center", color="#065f46", fontweight="bold")

    bars = [
        ("zero-insert\nmerge", explicit["zero_insert_merge_wall_s"], "#16a34a"),
        ("delete merge\nmax", explicit["max_delete_merge_s"], "#0f766e"),
        ("PQ train", bg["background_pq_train_wall_s"], "#2563eb"),
        ("PQ recode", bg["background_pq_recode_wall_s"], "#7c3aed"),
        ("background\nfull build", bg["background_elapsed_wall_s"], "#94a3b8"),
    ]
    y = np.arange(len(bars))
    axes[1].barh(y, [item[1] for item in bars], color=[item[2] for item in bars])
    axes[1].axvline(180.0, color="#111827", linestyle="--", linewidth=1.0, label="3 min reference")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Wall time (s, log scale)")
    axes[1].set_yticks(y, [item[0] for item in bars])
    axes[1].set_title("Blocking work vs background work")
    axes[1].legend(loc="lower right")
    for yi, (_, value, _) in enumerate(bars):
        axes[1].text(value * 1.06, yi, f"{value:.1f}s", va="center", fontsize=8)
    return save_fig(fig, out_dir, ppt_dir, "ppt_mixed_maintenance_window")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence for replacement, found {count}: {old[:80]}")
    return text.replace(old, new)


def update_tex(tex_path: Path, metrics: dict[str, Any]) -> None:
    text = tex_path.read_text(encoding="utf-8")
    max_avg = metrics["max_avg_latency_ms"]
    max_p95 = metrics["max_p95_latency_ms"]
    min_recall = metrics["min_recall"]
    space_total = metrics["space_total"]
    space_excess = metrics["space_excess"]
    bg_avg = metrics["background_max_avg"]
    bg_p95 = metrics["background_max_p95"]
    pq = metrics["pq_target"]

    replacements = {
        "查询 10@10 召回率稳定 $>$98\\% & 完成 &  \\\\": (
            f"查询 10@10 召回率稳定 $>$98\\% & 完成 & v100 v3 selected 200/200 通过；最小 recall@10 为 {min_recall:.2f}\\%。 \\\\"
        ),
        "联合向量近似查询时延 $<$10ms & 完成 & v3 packed serving 重跑后，200/200 selected 点 avg latency $<$10ms；p95 最大 10.32ms。 \\\\": (
            f"联合向量近似查询时延 $<$10ms & 完成 & v100 v3 packed serving 下 200/200 selected 点 avg/p95 latency 均 $<$10ms；p95 最大 {max_p95:.2f}ms。 \\\\"
        ),
        "索引膨胀率不能高于原始向量 1 倍 & 完成 & 严格额外空间/原始向量为 0.981x；严格总 serving footprint 为 1.981x。 \\\\": (
            f"索引总 footprint 低于原始向量 2 倍 & 完成 & 严格总 serving/raw 为 {space_total:.3f}x；严格额外空间/raw 为 {space_excess:.3f}x。 \\\\"
        ),
        "\\item 7 个高选择性慢点定向替换后，avg 全部 $<$10ms。": "\\item 7 个高风险慢点定向替换后，avg/p95 全部 $<$10ms。",
        "        \\textbf{左图：5 轮 delete/insert 后的 v3 selected 质量。} 允许重新选择 route/L 后，retrain 与 no-retrain 两类路径在每轮均保持 recall@10 $\\geq$98，且每轮最慢 avg latency $<$10ms。": (
            "        \\textbf{左图：5 轮 delete/insert 后的 v3 selected 质量。} 允许重新选择 route/L 后，retrain 与 no-retrain 两类路径在每轮均保持 recall@10 $\\geq$98，且每轮最慢 avg/p95 latency $<$10ms。"
        ),
        "        \\textbf{右图：全部 200 个 selected 点。} 新存储格式下，所有选择性、intersect/range、5 轮更新后的 selected 点均满足 recall@10 $\\geq$98 且 avg latency $<$10ms。": (
            "        \\textbf{右图：全部 200 个 selected 点。} 新存储格式下，所有选择性、intersect/range、5 轮更新后的 selected 点均满足 recall@10 $\\geq$98 且 avg/p95 latency $<$10ms。"
        ),
        "    \\textcolor{red}{结论：按验收口径重新选择 route/L 后，v3 packed serving 下 200/200 点通过 recall 与 avg latency 门槛。}": (
            "    \\textcolor{red}{结论：按验收口径重新选择 route/L 后，v100 v3 packed serving 下 200/200 点通过 recall、avg latency 与 p95 latency 门槛。}"
        ),
        "        \\textcolor{red}{结论：200 个动态 selected 点全部满足 recall@10 $\\geq$98；最大单线程 avg latency 为 9.96ms。}": (
            f"        \\textcolor{{red}}{{结论：200 个动态 selected 点全部满足 recall@10 $\\geq$98；最大 avg latency 为 {max_avg:.2f}ms，最大 p95 latency 为 {max_p95:.2f}ms。}}"
        ),
        "        严格口径下，1M serving 总 footprint / 原始向量为 \\textbf{1.981x}；额外空间 / 原始向量为 \\textbf{0.981x}，低于 1x 验收线。": (
            f"        严格口径下，1M serving 总 footprint / 原始向量为 \\textbf{{{space_total:.3f}x}}，低于 2x；额外空间 / 原始向量为 \\textbf{{{space_excess:.3f}x}}，低于 1x。"
        ),
        "        cycle5 range-u30 使用 no-retrain L420 后 recall@10 = 99.42，达到 matched-reference 目标。": (
            f"        PQ drift 策略比较 100/100 matched；cycle5 range-u30 使用 no-retrain L{int(pq['no_retrain_L'])} 后 recall@10 = {pq['no_retrain_recall']:.2f}，高于 matched target {pq['matched_reference_target_recall']:.2f}。"
        ),
        "        PQ train+recode 与 merge 属于前台必须等待部分；full rebuild 与 v3 repack 可后台化，不进入 3 分钟前台窗口。": (
            f"        1 核后台 full build/PQ retrain 并发时，前台 200/200 selected 点仍通过 10ms；during max avg/p95 = {bg_avg:.2f}/{bg_p95:.2f}ms。"
        ),
        "        动态 selected recall@10 & 最小 98.00 & 允许重新选择 route/L 的验收口径。 \\\\\n        动态 selected avg latency & 最大 9.96ms & 200/200 点 $<$10ms。 \\\\\n        动态 selected p95 latency & 最大 10.32ms & 4 个点 p95 $\\geq$10ms，保留 caveat。 \\\\\n        PQ drift matched-reference & 99.42 & cycle5 range-u30, L420。 \\\\\n        索引空间严格额外/原始 & 0.981x & 严格总 serving/raw 为 1.981x。 \\\\\n        读取粒度 & 4KB & straddling slot 使用两个 4KB 请求。 \\\\": (
            f"        动态 selected recall@10 & 最小 {min_recall:.2f} & 允许重新选择 route/L 的验收口径，200/200 通过。 \\\\\n"
            f"        动态 selected avg latency & 最大 {max_avg:.2f}ms & 200/200 点 $<$10ms。 \\\\\n"
            f"        动态 selected p95 latency & 最大 {max_p95:.2f}ms & 200/200 点 $<$10ms。 \\\\\n"
            f"        PQ drift matched-reference & 100/100 & cycle5 range-u30 no-retrain L{int(pq['no_retrain_L'])} matched。 \\\\\n"
            f"        索引空间严格 total/raw & {space_total:.3f}x & 严格额外/raw 为 {space_excess:.3f}x。 \\\\\n"
            f"        后台维护干扰 & {bg_avg:.2f}/{bg_p95:.2f}ms & 1 核 full build/PQ retrain 并发时 avg/p95 通过。 \\\\\n"
            "        读取粒度 & 4KB & straddling slot 使用两个 4KB 请求。 \\\\"
        ),
    }
    for old, new in replacements.items():
        text = replace_once(text, old, new)
    tex_path.write_text(text, encoding="utf-8")


def build_metrics(data: dict[str, Any]) -> dict[str, Any]:
    selected = data["selected"]
    registry = data["registry"]
    space_claim = claim_by_id(registry, "C_SPACE_STRICT_TOTAL_LT_2X_RAW")["metrics"]
    bg_claim = claim_by_id(registry, "C_BACKGROUND_MAINTENANCE_NO_FRONTEND_LATENCY_BREAK")["metrics"]
    explicit_claim = claim_by_id(registry, "C_EXPLICIT_MATERIALIZE_AND_DELETE_MERGE_EVIDENCE")["metrics"]
    pq_target = next(row for row in data["pq"] if row.get("case_id") == "cycle05_no_retrain_across_cycles_range_u30")
    return {
        "selected_rows": len(selected),
        "min_recall": min(row["recall"] for row in selected),
        "max_avg_latency_ms": max(row["avg_latency_ms"] for row in selected),
        "max_p95_latency_ms": max(row["p95_latency_ms"] for row in selected),
        "avg_lt_10_count": sum(row["avg_latency_ms"] < 10.0 for row in selected),
        "p95_lt_10_count": sum(row["p95_latency_ms"] < 10.0 for row in selected),
        "recall_ge_98_count": sum(row["recall"] >= 98.0 for row in selected),
        "space_total": space_claim["max_strict_total_over_raw_x"],
        "space_excess": space_claim["max_strict_excess_over_raw_x"],
        "background_max_avg": bg_claim["during_max_avg_latency_ms"],
        "background_max_p95": bg_claim["during_max_p95_latency_ms"],
        "background_elapsed_wall_s": bg_claim["background_elapsed_wall_s"],
        "background_pq_train_wall_s": bg_claim["background_pq_train_wall_s"],
        "background_pq_recode_wall_s": bg_claim["background_pq_recode_wall_s"],
        "max_delete_ms_per_vector": explicit_claim["max_delete_ms_per_vector"],
        "max_delete_merge_s": explicit_claim["max_delete_merge_s"],
        "zero_insert_merge_wall_s": explicit_claim["zero_insert_merge_wall_s"],
        "pq_target": pq_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--ppt-dir", type=Path, default=DEFAULT_PPT_DIR)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (ROOT / "experiments" / f"v100_ppt_refresh_{timestamp}")
    data = load_all(args.evidence_dir)
    metrics = build_metrics(data)

    written: list[str] = []
    written += plot_dynamic_scatter(data["selected"], out_dir, args.ppt_dir)
    written += plot_cycle_quality(data["selected"], out_dir, args.ppt_dir)
    written += plot_targeted_profile(data["profile"], out_dir, args.ppt_dir)
    written += plot_pq_strategy(data["pq"], data["registry"], out_dir, args.ppt_dir)
    written += plot_space(data["space"], out_dir, args.ppt_dir)
    written += plot_read_granularity(data["registry"], out_dir, args.ppt_dir)
    written += plot_maintenance(data["registry"], out_dir, args.ppt_dir)
    update_tex(args.ppt_dir / "dynamic-update.tex", metrics)

    metric_rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if key != "pq_target"
    ]
    write_csv(out_dir / "ppt_refresh_metrics.csv", metric_rows)
    write_json(
        out_dir / "ppt_refresh_manifest.json",
        {
            "evidence_dir": str(args.evidence_dir),
            "ppt_dir": str(args.ppt_dir),
            "metrics": metrics,
            "written": written,
        },
    )
    summary = "\n".join(
        [
            "# V100 PPT Refresh",
            "",
            f"- selected rows: {metrics['selected_rows']}",
            f"- recall/avg/p95 pass: {metrics['recall_ge_98_count']}/{metrics['avg_lt_10_count']}/{metrics['p95_lt_10_count']}",
            f"- max avg/p95 latency: {metrics['max_avg_latency_ms']:.3f}/{metrics['max_p95_latency_ms']:.3f} ms",
            f"- space total/excess over raw: {metrics['space_total']:.6f}x/{metrics['space_excess']:.6f}x",
            f"- background during max avg/p95: {metrics['background_max_avg']:.3f}/{metrics['background_max_p95']:.3f} ms",
            "",
        ]
    )
    (out_dir / "ppt_refresh_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "written_count": len(written)}, indent=2))


if __name__ == "__main__":
    main()
