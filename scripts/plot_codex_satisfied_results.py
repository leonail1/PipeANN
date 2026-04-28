#!/usr/bin/env python3
"""Plot the accepted Codex hybrid-search validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BUCKET_ORDER = [
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_equality_rows(root: Path) -> list[dict]:
    main = load_json(
        root
        / "experiments/codex_graph_build_r75_pq32_20260428/"
        / "auto_l115_equality_final_after_memopt/table.json"
    )
    missing = load_json(
        root
        / "experiments/codex_graph_build_r75_pq32_20260428/"
        / "auto_l115_equality_missing_buckets/table.json"
    )
    by_bucket = {row["bucket"]: row for row in main + missing}
    return [by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in by_bucket]


def write_summary_markdown(
    path: Path,
    equality: list[dict],
    range_row: dict,
    memory: list[dict],
    bloat: dict,
) -> None:
    lines = [
        "# Codex Satisfied Validation Results",
        "",
        "Thresholds used for this summary: recall@10 >= 98%, average latency < 10 ms, single-query RSS <= 30 MB, and extra index bloat <= 1x raw vectors.",
        "",
        "## Equality Queries",
        "",
        "| Bucket | Recall@10 (%) | Avg latency (ms) | P99 latency (ms) | Route |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in equality:
        route = "prefilter" if row.get("prefilter_count", 0) else "graph"
        lines.append(
            f"| {BUCKET_LABELS.get(row['bucket'], row['bucket'])} | "
            f"{row['recall']:.2f} | {row['avg_latency_us'] / 1000:.3f} | "
            f"{row['p99_latency_us'] / 1000:.3f} | {route} |"
        )
    lines.extend(
        [
            "",
            "## Range Query",
            "",
            f"- range_0_2: recall@10={range_row['recall']:.2f}%, avg={range_row['avg_latency_us'] / 1000:.3f} ms, p99={range_row['p99_latency_us'] / 1000:.3f} ms.",
            "",
            "## Resource Footprint",
            "",
        ]
    )
    for row in memory:
        lines.append(f"- {row['case']}: max RSS={row['max_rss_mib']:.2f} MiB.")
    lines.append(
        f"- Index extra bloat: {bloat['extra_over_raw_ratio']:.3f}x raw vectors; total/raw={bloat['total_to_raw_ratio']:.3f}x."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(root: Path, out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    equality = load_equality_rows(root)
    range_row = load_jsonl(
        root
        / "experiments/codex_graph_build_r75_pq32_20260428/"
        / "range_prefilter_identity_truth/range_rerank_132.jsonl"
    )[0]
    memory = load_json(
        root
        / "experiments/codex_graph_build_r75_pq32_20260428/"
        / "memory_l115_dropcache_batch64/table.json"
    )
    bloat = load_json(root / "experiments/codex_req_validation_20260427/bloat/summary.json")

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [BUCKET_LABELS[row["bucket"]] for row in equality]
    x = list(range(len(labels)))
    avg_ms = [row["avg_latency_us"] / 1000.0 for row in equality]
    p99_ms = [row["p99_latency_us"] / 1000.0 for row in equality]
    recall = [row["recall"] for row in equality]
    colors = ["#28a6a2" if row.get("prefilter_count", 0) else "#2f6fbb" for row in equality]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Satisfied Hybrid Search Validation Results", fontsize=20, fontweight="bold")

    ax = axes[0][0]
    ax.bar(x, avg_ms, color=colors, width=0.62, label="Avg latency")
    ax.plot(x, p99_ms, color="#4c4c4c", marker="o", linewidth=2, label="P99 latency")
    ax.axhline(10, color="#c43c35", linestyle="--", linewidth=1.8, label="10 ms avg target")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Equality Query Latency")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left")

    ax = axes[0][1]
    ax.plot(x, recall, marker="o", color="#166534", linewidth=2.5, label="Recall@10")
    ax.axhline(98, color="#c43c35", linestyle="--", linewidth=1.8, label="98% target")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(97.5, 100.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    ax.set_ylabel("Recall@10")
    ax.set_title("Equality Query Recall")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right")

    ax = axes[1][0]
    range_metrics = [range_row["recall"], range_row["avg_latency_us"] / 100.0, range_row["p99_latency_us"] / 100.0]
    range_labels = ["Recall@10 (%)", "Avg latency / 0.1ms", "P99 latency / 0.1ms"]
    ax.bar(range_labels, range_metrics, color=["#166534", "#28a6a2", "#4c4c4c"], width=0.58)
    ax.axhline(98, color="#c43c35", linestyle="--", linewidth=1.8, label="98% recall target")
    ax.set_title("Range Query: range_0_2")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right")

    ax = axes[1][1]
    mem_labels = [row["case"].replace("equality_", "eq_") for row in memory]
    mem_mib = [row["max_rss_mib"] for row in memory]
    ax.bar(mem_labels, mem_mib, color="#8b5cf6", width=0.58, label="Max RSS")
    ax.axhline(30, color="#c43c35", linestyle="--", linewidth=1.8, label="30 MB limit")
    ax.set_ylabel("MiB")
    ax.set_title("Single-Query Memory")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right")

    fig.savefig(out_dir / "codex_satisfied_metrics_overview.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(["Extra/raw", "Total/raw"], [bloat["extra_over_raw_ratio"], bloat["total_to_raw_ratio"]], color=["#2f6fbb", "#28a6a2"])
    ax.axhline(1.0, color="#c43c35", linestyle="--", linewidth=1.8, label="Extra bloat limit")
    ax.set_ylabel("Ratio")
    ax.set_title("Index Footprint on SIFT1M")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left")
    fig.savefig(out_dir / "codex_satisfied_index_footprint.png", dpi=dpi)
    plt.close(fig)

    write_summary_markdown(out_dir / "codex_satisfied_metrics_summary.md", equality, range_row, memory, bloat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/codex_req_validation_20260427/plots"),
    )
    parser.add_argument("--dpi", type=int, default=360)
    args = parser.parse_args()
    plot(args.root.resolve(), args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
