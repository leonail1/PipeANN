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
    if exp2:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        direct_rows = sorted([r for r in exp2 if r.get("path") == "direct_build"], key=lambda r: r["points"])
        if direct_rows:
            ax.plot([r["points"] for r in direct_rows], [r.get("recall@10", 0) for r in direct_rows],
                    marker="o", linewidth=2.0, label="Direct build")
        if exp2_seed_sweep:
            for start_n in [250_000, 500_000]:
                rows = sorted(
                    [r for r in exp2_seed_sweep if as_int(r, "start_points") == start_n],
                    key=lambda r: as_int(r, "points"),
                )
                if rows:
                    ax.plot([as_int(r, "points") for r in rows], [as_float(r, "recall@10") for r in rows],
                            marker="s", linestyle="-.", label=f"{start_n // 1000}k seed insert")
        else:
            rows = sorted([r for r in exp2 if r.get("path") == "incremental_insert"], key=lambda r: r["points"])
            if rows:
                ax.plot([r["points"] for r in rows], [r.get("recall@10", 0) for r in rows],
                        marker="o", label="10k seed insert")
        ax.set_xlabel("Index vectors")
        ax.set_ylabel("Recall@10 (%)")
        ax.set_title("Stage Recall: Direct Build vs Larger-Seed Insert")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(exp2_dir / "stage_recall_build_vs_insert.png", dpi=dpi)
        plt.close(fig)

    if exp2_seed_sweep:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        if exp2:
            rows = sorted([r for r in exp2 if r.get("path") == "direct_build"], key=lambda r: r["points"])
            if rows:
                ax.plot([r["points"] for r in rows], [r.get("recall@10", 0) for r in rows],
                        marker="o", linestyle="-", label="Direct build")
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
        for ins_t in sorted({as_int(r, "insert_threads") for r in exp3}):
            rows = sorted([r for r in exp3 if as_int(r, "insert_threads") == ins_t], key=lambda r: as_int(r, "query_threads"))
            x = [as_int(r, "query_threads") for r in rows]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
            axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in rows], marker="o", label="Avg")
            axes[0].plot(x, [as_float(r, "p95_latency_us") / 1000.0 for r in rows], marker="o", label="P95")
            axes[0].set_xlabel("Query threads")
            axes[0].set_ylabel("Latency (ms)")
            axes[0].set_title(f"Latency during insert, insert threads={ins_t}")
            axes[0].grid(axis="y", alpha=0.3)
            axes[0].legend()
            axes[1].plot(x, [as_float(r, "qps") for r in rows], marker="o", color="#166534")
            axes[1].set_xlabel("Query threads")
            axes[1].set_ylabel("QPS")
            axes[1].set_title("Foreground query throughput")
            axes[1].grid(axis="y", alpha=0.3)
            fig.savefig(root / f"exp3_search_during_insert/search_during_insert_ins{ins_t}.png", dpi=dpi)
            plt.close(fig)

    exp4 = load_csv(root / "exp4_delete_reinsert_selectivity/table.csv")
    if exp4:
        for state, filename in [
            ("1m_initial", "selectivity_1m_initial.png"),
            ("750k_after_delete", "selectivity_750k_after_delete.png"),
            ("1m_after_reinsert", "selectivity_1m_after_reinsert.png"),
        ]:
            rows_by_bucket = {r["bucket"]: r for r in exp4 if r.get("state") == state}
            rows = [rows_by_bucket[b] for b in BUCKET_ORDER if b in rows_by_bucket]
            labels = [BUCKET_LABELS.get(r["bucket"], r["bucket"]) for r in rows]
            x = list(range(len(rows)))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
            lat_ms = [as_float(r, "avg_latency_us") / 1000.0 for r in rows]
            qps = [as_float(r, "qps") for r in rows]
            recalls = [as_float(r, "recall@10") for r in rows]
            chosen_l = [as_int(r, "chosen_L") for r in rows]
            point_colors = ["#2563eb" if recall >= 98.0 else "#dc2626" for recall in recalls]
            routes = [r.get("selected_route") or r.get("route", "") for r in rows]
            route_labels = [
                {"prefilter": "prefilter", "graph": "graph", "auto": "auto"}.get(route, route or "unknown")
                for route in routes
            ]
            axes[0].plot(x, lat_ms, color="#2563eb", linewidth=1.5, alpha=0.75)
            axes[0].scatter(x, lat_ms, c=point_colors, zorder=3)
            for xi, yi, l_value, route_label in zip(x, lat_ms, chosen_l, route_labels):
                axes[0].annotate(f"{route_label}\nL={l_value}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                                 ha="center", fontsize=7)
            axes[0].set_xticks(x, labels, rotation=35, ha="right")
            axes[0].set_ylabel("Latency (ms)")
            axes[0].set_title(f"{state}: selected-route latency")
            axes[0].grid(axis="y", alpha=0.3)
            axes[1].plot(x, qps, color="#166534", linewidth=1.5, alpha=0.75)
            axes[1].scatter(x, qps, c=point_colors, zorder=3)
            for xi, yi, l_value, route_label in zip(x, qps, chosen_l, route_labels):
                axes[1].annotate(f"{route_label}\nL={l_value}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                                 ha="center", fontsize=7)
            axes[1].set_xticks(x, labels, rotation=35, ha="right")
            axes[1].set_ylabel("QPS")
            axes[1].set_title(f"{state}: selected-route QPS")
            axes[1].grid(axis="y", alpha=0.3)
            fig.suptitle("Exp4: latency/QPS after selected-route + L calibration for recall@10 >= 98%")
            fig.savefig(root / f"exp4_delete_reinsert_selectivity/{filename}", dpi=dpi)
            plt.close(fig)

    exp5 = load_csv(root / "exp5_index_bloat_by_size/table.csv")
    if exp5:
        rows = sorted(exp5, key=lambda r: as_int(r, "points"))
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.plot([as_int(r, "points") for r in rows], [as_float(r, "extra_over_raw_ratio") for r in rows], marker="o", label="Extra/raw")
        ax.plot([as_int(r, "points") for r in rows], [as_float(r, "total_to_raw_ratio") for r in rows], marker="o", label="Total/raw")
        ax.axhline(1.0, color="#c43c35", linestyle="--", linewidth=1.5, label="Extra <= 1.0")
        ax.set_xlabel("Index vectors")
        ax.set_ylabel("Ratio")
        ax.set_title("Index Bloat by Size")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(root / "exp5_index_bloat_by_size/index_bloat_by_size.png", dpi=dpi)
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
                rows = [r for r in single_thread if r.get("route") == route]
                rows_by_bucket = {r["bucket"]: r for r in rows}
                x = [i for i, bucket in enumerate(BUCKET_ORDER) if bucket in rows_by_bucket]
                ordered = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not ordered:
                    continue
                ax.plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in ordered],
                        marker=markers.get(route, "o"), color=colors.get(route), label=route)
            x_all = list(range(len(BUCKET_ORDER)))
            labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
            ax.set_xticks(x_all, labels_all, rotation=35, ha="right")
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
                rows = [r for r in route_rows if as_int(r, "threads") == threads]
                rows_by_bucket = {r["bucket"]: r for r in rows}
                x = [i for i, bucket in enumerate(BUCKET_ORDER) if bucket in rows_by_bucket]
                labels = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                ordered = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in ordered],
                             marker="o", label=f"{threads} threads")
                axes[1].plot(x, [as_float(r, "qps") for r in ordered],
                             marker="o", label=f"{threads} threads")
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
            x_all = list(range(len(BUCKET_ORDER)))
            labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]
            for route in ["prefilter", "graph"]:
                rows = [r for r in single_thread if r.get("route") == route]
                rows_by_bucket = {r["bucket"]: r for r in rows}
                x = [i for i, bucket in enumerate(BUCKET_ORDER) if bucket in rows_by_bucket]
                ordered = [rows_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in rows_by_bucket]
                if not ordered:
                    continue
                label = f"{route} (passing points)"
                axes[0].plot(x, [as_float(r, "avg_latency_us") / 1000.0 for r in ordered],
                             marker=markers.get(route, "o"), color=colors.get(route), label=label)
                axes[1].plot(x, [as_float(r, "qps") for r in ordered],
                             marker=markers.get(route, "o"), color=colors.get(route), label=label)
            axes[0].set_xticks(x_all, labels_all, rotation=35, ha="right")
            axes[0].set_ylabel("Avg latency (ms)")
            axes[0].set_title("Latency")
            axes[0].grid(axis="y", alpha=0.3)
            axes[0].legend()
            axes[1].set_xticks(x_all, labels_all, rotation=35, ha="right")
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
            bucket = bucket_from_query_label(row)
            route = row.get("route")
            l_value = as_int(row, "L")
            if route == "prefilter" and l_value == 10:
                fixed_rows[(bucket, route)] = row
            if route == "graph" and l_value == 100:
                fixed_rows[(bucket, route)] = row
        graph_rows = [fixed_rows[(bucket, "graph")] for bucket in BUCKET_ORDER if (bucket, "graph") in fixed_rows]
        prefilter_rows = [fixed_rows[(bucket, "prefilter")] for bucket in BUCKET_ORDER if (bucket, "prefilter") in fixed_rows]
        if graph_rows and prefilter_rows:
            fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            x_all = list(range(len(BUCKET_ORDER)))
            labels_all = [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER]

            def plot_fixed(route_rows: list[dict], color: str, marker: str, label: str) -> None:
                row_by_bucket = {bucket_from_query_label(row): row for row in route_rows}
                x = [i for i, bucket in enumerate(BUCKET_ORDER) if bucket in row_by_bucket]
                ordered = [row_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in row_by_bucket]
                ax.plot(x, [as_float(row, "avg_latency_us") / 1000.0 for row in ordered],
                        marker=marker, color=color, label=label)

            plot_fixed(prefilter_rows, "#2ca02c", "D", "prefilter, L=10")
            plot_fixed(graph_rows, "#ff7f0e", "o", "graph, L=100")
            ax.set_xticks(x_all, labels_all, rotation=35, ha="right")
            ax.set_ylabel("Avg latency (ms)")
            ax.set_title("SIFT-1M Direct 1M Baseline, Fixed L")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")

            row_by_bucket = {bucket_from_query_label(row): row for row in graph_rows}
            x = [i for i, bucket in enumerate(BUCKET_ORDER) if bucket in row_by_bucket]
            ordered = [row_by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in row_by_bucket]
            recall_ax = ax.twinx()
            recall_ax.plot(x, [as_float(row, "recall") for row in ordered],
                           color="#2563eb", linestyle="--", marker="x", label="graph recall@10, L=100")
            recall_ax.axhline(98.0, color="#dc2626", linestyle=":", linewidth=1.2)
            recall_ax.set_ylabel("Graph recall@10 (%)")
            recall_ax.set_ylim(0, 105)
            recall_ax.legend(loc="lower right")
            fig.savefig(root / "exp_baseline/baseline_fixed_l100_ref_like.png", dpi=dpi)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/codex_dynamic_update_suite_20260428"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    plot(args.out_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
