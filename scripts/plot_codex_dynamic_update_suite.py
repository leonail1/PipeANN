#!/usr/bin/env python3
"""Plot the Codex dynamic update suite outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


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
BUCKET_SELECTIVITY = {
    "u1e-03": 0.001,
    "u3e-03": 0.003,
    "u1e-02": 0.01,
    "u5e-02": 0.05,
    "u1e-01": 0.10,
    "u25": 0.25,
    "u30": 0.30,
    "u50": 0.50,
    "u75": 0.75,
    "u100": 1.00,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def load_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as reader:
        return list(csv.DictReader(reader))


def as_float(row: dict, key: str) -> float:
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def bucket_from_query_label(row: dict) -> str:
    value = row.get("query_label_file", "")
    match = re.search(r"query_\d+_(u[^/.]+)\.spmat", value)
    return match.group(1) if match else row.get("bucket", "")


def plot(root: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    exp1 = load_csv(root / "exp1_insert_vs_build_threads/table.csv")
    if exp1:
        exp1 = sorted(exp1, key=lambda r: as_int(r, "threads"))
        labels = [str(as_int(r, "threads")) for r in exp1]
        x = range(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.bar([i - width / 2 for i in x], [as_float(r, "insert_total_s") for r in exp1], width=width, label="seed build + insert")
        ax.bar([i + width / 2 for i in x], [as_float(r, "build_1m_s") for r in exp1], width=width, label="direct build")
        ax.set_xticks(list(x), labels)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Seconds")
        ax.set_title("Insert vs Direct Build Time")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(root / "exp1_insert_vs_build_threads/insert_vs_build_threads.png", dpi=dpi)
        plt.close(fig)

    exp2 = load_jsonl(root / "exp2_stage_recall_build_vs_insert/results.jsonl")
    exp2_dir = root / "exp2_stage_recall_build_vs_insert"
    exp2_seed_sweep = load_csv(exp2_dir / "seed_sweep_table.csv")

    if exp2_seed_sweep:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        if exp2:
            rows = sorted([r for r in exp2 if r.get("path") == "direct_build"], key=lambda r: r["points"])
            if rows:
                ax.plot([r["points"] for r in rows], [r.get("recall@10", 0) for r in rows],
                        marker="o", linestyle="-", label="Direct build")
            rows = sorted([r for r in exp2 if r.get("path") == "incremental_insert"], key=lambda r: r["points"])
            if rows:
                ax.plot([r["points"] for r in rows], [r.get("recall@10", 0) for r in rows],
                        marker="^", linestyle="--", label="10k seed insert")
        for start_n in sorted({as_int(r, "start_points") for r in exp2_seed_sweep}):
            rows = sorted([r for r in exp2_seed_sweep if as_int(r, "start_points") == start_n],
                          key=lambda r: as_int(r, "points"))
            ax.plot([as_int(r, "points") for r in rows], [as_float(r, "recall@10") for r in rows],
                    marker="s", linestyle="-.", label=f"{start_n // 1000}k seed insert")
        ax.set_xlabel("Index vectors")
        ax.set_ylabel("Recall@10 (%)")
        ax.set_title("Exp2 Seed Sweep: Graph Recall at Fixed L")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(exp2_dir / "seed_sweep_recall.png", dpi=dpi)
        plt.close(fig)

    exp3 = load_csv(root / "exp3_search_during_insert/table.csv")
    if exp3:
        exp3_dir = root / "exp3_search_during_insert"
        if any(row.get("bucket") for row in exp3):
            for stale in [
                exp3_dir / "search_during_insert_2x3.png",
                exp3_dir / "search_during_insert_ins1.png",
                exp3_dir / "search_during_insert_ins2.png",
                exp3_dir / "search_during_insert_ins4.png",
            ]:
                if stale.exists():
                    stale.unlink()
            colors = {0: "#1d4ed8", 1: "#16a34a", 2: "#d97706", 4: "#dc2626"}
            labels_by_threads = {0: "no insert", 1: "insert 1 thread", 2: "insert 2 threads", 4: "insert 4 threads"}
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
            x_all = list(range(len(BUCKET_ORDER)))
            x_labels = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
            for ins_t in sorted({as_int(r, "insert_threads") for r in exp3}):
                rows_by_bucket = {r["bucket"]: r for r in exp3 if as_int(r, "insert_threads") == ins_t and r.get("bucket")}
                rows = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not rows:
                    continue
                x = [BUCKET_ORDER.index(row["bucket"]) for row in rows]
                color = colors.get(ins_t)
                label = labels_by_threads.get(ins_t, f"insert {ins_t} threads")
                axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows],
                             marker="o", linewidth=2.0, color=color, label=label)
                axes[1].plot(x, [as_float(r, "qps") for r in rows],
                             marker="o", linewidth=2.0, color=color, label=label)
                for ax, values in [
                    (axes[0], [as_float(r, "avg_latency_us") / 1000.0 for r in rows]),
                    (axes[1], [as_float(r, "qps") for r in rows]),
                ]:
                    graph_x = [xi for xi, row in zip(x, rows) if as_int(row, "chosen_L") > 10]
                    graph_y = [yi for yi, row in zip(values, rows) if as_int(row, "chosen_L") > 10]
                    if graph_x:
                        ax.scatter(graph_x, graph_y, marker="^", s=52, color=color, edgecolor="black", linewidth=0.4, zorder=4)
            for ax in axes:
                ax.set_xticks(x_all, x_labels, rotation=25, ha="right")
                ax.grid(axis="y", alpha=0.3)
            axes[0].axhline(10.0, color="#991b1b", linestyle="--", linewidth=1.0, label="10 ms")
            axes[0].set_xlabel("Selectivity")
            axes[0].set_ylabel("Avg latency (ms)")
            axes[0].set_title("Foreground latency")
            axes[1].set_xlabel("Selectivity")
            axes[1].set_ylabel("QPS")
            axes[1].set_title("Foreground QPS")
            axes[1].legend(loc="best")
            axes[0].legend(loc="best")
            fig.suptitle("Exp3: foreground search during insertion")
            fig.savefig(exp3_dir / "search_during_insert_selectivity.png", dpi=dpi)
            plt.close(fig)

    exp4_dir = root / "exp4_intersect_range_selectivity"
    exp4 = load_csv(exp4_dir / "table.csv")
    if exp4:
        x_all = list(range(len(BUCKET_ORDER)))
        labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
        selector_styles = {
            "intersect": ("intersect", "o", "#2563eb"),
            "range": ("range", "s", "#dc2626"),
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        point_labels = [{xi: [] for xi in x_all} for _ in axes]
        for selector, (label, marker, color) in selector_styles.items():
            rows_by_bucket = {r["bucket"]: r for r in exp4 if r.get("selector_type") == selector}
            rows = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
            if not rows:
                continue
            x = [BUCKET_ORDER.index(r["bucket"]) for r in rows]
            axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows],
                         marker=marker, linewidth=2.0, color=color, label=label)
            axes[1].plot(x, [as_float(r, "qps") for r in rows],
                         marker=marker, linewidth=2.0, color=color, label=label)
            for ax, values in [
                (axes[0], [as_float(r, "avg_latency_us") / 1000.0 for r in rows]),
                (axes[1], [as_float(r, "qps") for r in rows]),
            ]:
                for xi, yi, row in zip(x, values, rows):
                    route = row.get("selected_route") or row.get("route", "")
                    route_short = "G" if route == "graph" else "P"
                    point_labels[0 if ax is axes[0] else 1][xi].append(
                        (label[0].upper(), f"{route_short}/L{as_int(row, 'chosen_L')}", yi)
                    )
        for ax_idx, ax in enumerate(axes):
            for xi, items in point_labels[ax_idx].items():
                if not items:
                    continue
                unique = []
                for selector_short, route_l, _ in items:
                    if route_l not in unique:
                        unique.append(route_l)
                if len(unique) == 1:
                    text = unique[0]
                else:
                    text = "\n".join(f"{selector_short}:{route_l}" for selector_short, route_l, _ in items)
                yi = max(item[2] for item in items)
                ax.annotate(text, (xi, yi), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color="#374151")
        axes[0].axhline(10.0, color="#991b1b", linestyle="--", linewidth=1.0, label="10 ms")
        for ax in axes:
            ax.set_xticks(x_all, labels_all, rotation=35, ha="right")
            ax.set_xlabel("Selectivity")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("Avg latency (ms)")
        axes[0].set_title("Latency")
        axes[1].set_ylabel("QPS")
        axes[1].set_title("QPS")
        fig.suptitle("Exp4: Direct 1M intersect/range search, selected route and minimum L for recall@10 >= 98%")
        fig.savefig(exp4_dir / "intersect_range_latency_qps.png", dpi=dpi)
        plt.close(fig)

        rss_rows = [r for r in exp4 if as_float(r, "rss_single_query_kb") > 0 or as_float(r, "max_rss_kb") > 0]
        if rss_rows:
            fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
            max_rss_mb = 0.0
            max_point: tuple[int, float, str] | None = None
            for selector, (label, marker, color) in selector_styles.items():
                rows_by_bucket = {r["bucket"]: r for r in rss_rows if r.get("selector_type") == selector}
                ordered = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not ordered:
                    continue
                x = [BUCKET_ORDER.index(row["bucket"]) for row in ordered]
                y = [
                    (as_float(row, "rss_single_query_kb") or as_float(row, "max_rss_kb")) / 1024.0
                    for row in ordered
                ]
                ax.plot(x, y, marker=marker, linewidth=2.0, markersize=6, color=color, label=label)
                for xi, yi, row in zip(x, y, ordered):
                    route = row.get("selected_route") or row.get("route", "")
                    ax.annotate("G" if route == "graph" else "P", (xi, yi), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8, color=color)
                    if yi > max_rss_mb:
                        max_rss_mb = yi
                        max_point = (xi, yi, label)
            ax.axhline(30.0, color="#dc2626", linestyle="--", linewidth=1.4, label="30 MB limit")
            if max_point is not None:
                ax.annotate(f"max {max_point[1]:.2f} MB", (max_point[0], max_point[1]),
                            textcoords="offset points", xytext=(12, 10), ha="left", fontsize=9,
                            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.9})
            ax.set_xticks(x_all, labels_all, rotation=35, ha="right")
            ax.set_ylabel("Single-query process RSS (MB)")
            ax.set_xlabel("Selectivity")
            ax.set_title("Exp4: single-query process RSS by filter type")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(ncols=2)
            ax.text(0.01, 0.02, "P=prefilter, G=graph; RSS process does not load query/GT/query-spmat files.",
                    transform=ax.transAxes, fontsize=8, color="#4b5563")
            fig.savefig(exp4_dir / "rss_by_selectivity.png", dpi=dpi)
            plt.close(fig)

    baseline = load_csv(root / "exp_baseline/table.csv")
    if baseline:
        baseline = [r for r in baseline if r.get("status") == "ok"]
        single_thread = [r for r in baseline if as_int(r, "threads") == 1]
        if single_thread:
            fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            markers = {"prefilter": "D", "graph": "o"}
            colors = {"prefilter": "#2ca02c", "graph": "#ff7f0e"}
            for route in ["prefilter", "graph"]:
                rows_by_bucket = {r["bucket"]: r for r in single_thread if r.get("route") == route}
                rows = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not rows:
                    continue
                x = [BUCKET_ORDER.index(r["bucket"]) for r in rows]
                ax.plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows],
                        marker=markers[route], color=colors[route], linewidth=2.0, label=route)
                for xi, row in zip(x, rows):
                    ax.annotate(f"L={as_int(row, 'chosen_L')}", (xi, as_float(row, "avg_latency_us") / 1000.0),
                                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
            ax.axhline(10.0, color="#991b1b", linestyle="--", linewidth=1.0, label="10 ms")
            ax.set_xticks(list(range(len(BUCKET_ORDER))), [BUCKET_LABELS[b] for b in BUCKET_ORDER],
                          rotation=35, ha="right")
            ax.set_xlabel("Selectivity")
            ax.set_ylabel("Avg latency (ms)")
            ax.set_title("SIFT-1M Direct 1M Baseline, 1 Thread")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.savefig(root / "exp_baseline/baseline_single_thread_latency.png", dpi=dpi)
            plt.close(fig)

        for route in ["prefilter", "graph"]:
            route_rows = [r for r in baseline if r.get("route") == route]
            if not route_rows:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            for threads in sorted({as_int(r, "threads") for r in route_rows}):
                rows_by_bucket = {r["bucket"]: r for r in route_rows if as_int(r, "threads") == threads}
                rows = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not rows:
                    continue
                x = [BUCKET_ORDER.index(r["bucket"]) for r in rows]
                label = f"{threads} thread" if threads == 1 else f"{threads} threads"
                axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows], marker="o", label=label)
                axes[1].plot(x, [as_float(r, "qps") for r in rows], marker="o", label=label)
            x_all = list(range(len(BUCKET_ORDER)))
            labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
            axes[0].set_xticks(x_all, labels_all, rotation=35, ha="right")
            axes[0].set_ylabel("Avg latency (ms)")
            axes[0].set_title(f"Direct 1M baseline: {route} latency")
            axes[0].grid(axis="y", alpha=0.3)
            axes[0].legend(ncols=2)
            axes[1].set_xticks(x_all, labels_all, rotation=35, ha="right")
            axes[1].set_ylabel("QPS")
            axes[1].set_title(f"Direct 1M baseline: {route} QPS")
            axes[1].grid(axis="y", alpha=0.3)
            axes[1].legend(ncols=2)
            fig.suptitle("Exp baseline: forced route, minimum L for recall@10 >= 98%; skipped points are omitted")
            fig.savefig(root / f"exp_baseline/baseline_{route}.png", dpi=dpi)
            plt.close(fig)

        if single_thread:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
            markers = {"prefilter": "D", "graph": "o"}
            colors = {"prefilter": "#2ca02c", "graph": "#ff7f0e"}
            for route in ["prefilter", "graph"]:
                rows_by_bucket = {r["bucket"]: r for r in single_thread if r.get("route") == route}
                rows = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not rows:
                    continue
                x = [BUCKET_ORDER.index(r["bucket"]) for r in rows]
                axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows],
                             marker=markers[route], color=colors[route], linewidth=2.0, label=route)
                axes[1].plot(x, [as_float(r, "qps") for r in rows],
                             marker=markers[route], color=colors[route], linewidth=2.0, label=route)
            labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
            for ax in axes:
                ax.set_xticks(list(range(len(BUCKET_ORDER))), labels_all, rotation=35, ha="right")
                ax.set_xlabel("Selectivity")
            axes[0].set_ylabel("Avg latency (ms)")
            axes[0].set_title("Latency")
            axes[0].grid(axis="y", alpha=0.3)
            axes[0].legend()
            axes[1].set_ylabel("QPS")
            axes[1].set_title("QPS")
            axes[1].grid(axis="y", alpha=0.3)
            axes[1].legend()
            fig.suptitle("Exp baseline: prefilter vs graph, 1 thread, minimum L for recall@10 >= 98%")
            fig.savefig(root / "exp_baseline/baseline_prefilter_vs_graph.png", dpi=dpi)
            plt.close(fig)

    calibration = load_jsonl(root / "exp_baseline/calibration.jsonl")
    if calibration:
        fixed_rows: dict[tuple[str, str], dict] = {}
        for row in calibration:
            if as_int(row, "threads") != 1:
                continue
            route = row.get("route")
            bucket = bucket_from_query_label(row)
            if route == "graph" and as_int(row, "chosen_L") == 100 and bucket:
                fixed_rows[(route, bucket)] = row
        if fixed_rows:
            ordered = [fixed_rows[("graph", bucket)] for bucket in BUCKET_ORDER if ("graph", bucket) in fixed_rows]
            labels = [BUCKET_LABELS[bucket_from_query_label(r)] for r in ordered]
            x = list(range(len(ordered)))
            lat_ms = [as_float(r, "avg_latency_us") / 1000.0 for r in ordered]
            qps = [as_float(r, "qps") for r in ordered]
            recall = [as_float(r, "recall@10") for r in ordered]
            fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
            ax.plot(x, lat_ms, marker="o", linewidth=2.0, color="#1d4ed8", label="Graph L=100 latency")
            ax.axhline(10.0, color="#991b1b", linestyle="--", linewidth=1.0, label="10 ms")
            ax.set_xticks(x, labels, rotation=35, ha="right")
            ax.set_xlabel("Selectivity")
            ax.set_ylabel("Avg latency (ms)")
            ax.set_title("SIFT-1M Direct 1M Graph Baseline, L=100")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper right")
            qps_ax = ax.twinx()
            qps_ax.plot(x, qps, marker="s", linewidth=1.7, color="#16a34a", label="QPS")
            qps_ax.set_ylabel("QPS")
            recall_ax = ax.twinx()
            recall_ax.spines.right.set_position(("axes", 1.12))
            recall_ax.plot(x, recall, marker="^", linestyle=":", linewidth=1.5, color="#7c3aed",
                           label="Recall@10")
            recall_ax.axhline(98.0, color="#dc2626", linestyle=":", linewidth=1.2)
            recall_ax.set_ylabel("Graph recall@10 (%)")
            recall_ax.set_ylim(0, 105)
            recall_ax.legend(loc="lower right")
            fig.savefig(root / "exp_baseline/baseline_fixed_l100_ref_like.png", dpi=dpi)
            plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    plot(args.out_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
