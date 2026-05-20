#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/r116_suite_pq16_aris_20260520_072453")
EXP6 = ROOT / "exp6_query_thread_budget"
EXP2_L200 = ROOT / "ppt_l200_exp2" / "exp2_stage_recall_build_vs_insert"
PPT_UPDATES = ROOT / "ppt_updates"


def load_exp6_rows() -> list[dict]:
    rows: list[dict] = []
    with (EXP6 / "table.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["threads_i"] = int(float(row["threads"]))
            row["avg_ms"] = float(row["avg_latency_us"]) / 1000.0
            row["p95_ms"] = float(row.get("p95_latency_us") or 0.0) / 1000.0
            row["p99_ms"] = float(row.get("p99_latency_us") or 0.0) / 1000.0
            rows.append(row)
    return rows


def write_exp6_plots() -> None:
    rows = load_exp6_rows()
    threads = sorted({row["threads_i"] for row in rows})

    summary_rows: list[dict] = []
    for threads_n in threads:
        thread_rows = [row for row in rows if row["threads_i"] == threads_n]
        vals = np.array([row["avg_ms"] for row in thread_rows], dtype=float)
        worst = max(thread_rows, key=lambda row: row["avg_ms"])
        summary_rows.append(
            {
                "threads": threads_n,
                "workload_count": len(vals),
                "p50_avg_latency_ms": np.percentile(vals, 50),
                "p90_avg_latency_ms": np.percentile(vals, 90),
                "p95_avg_latency_ms": np.percentile(vals, 95),
                "max_avg_latency_ms": vals.max(),
                "worst_selector": worst["selector_type"],
                "worst_bucket": worst["bucket"],
                "worst_route": worst.get("selected_route") or worst.get("route"),
                "worst_L": worst.get("chosen_L"),
                "worst_recall@10": worst.get("recall@10"),
            }
        )
    with (EXP6 / "thread_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    styles = {
        "p50": (50, "#2563eb"),
        "p90": (90, "#16a34a"),
        "p95": (95, "#f59e0b"),
        "max": (100, "#dc2626"),
    }
    for label, (percentile, color) in styles.items():
        y = []
        for threads_n in threads:
            vals = np.array([row["avg_ms"] for row in rows if row["threads_i"] == threads_n], dtype=float)
            y.append(vals.max() if percentile == 100 else np.percentile(vals, percentile))
        ax.plot(threads, y, marker="o", linewidth=2, label=label, color=color)
    ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms budget")
    ax.set_xticks(threads)
    ax.set_xlabel("Query threads")
    ax.set_ylabel("Avg latency across workloads (ms)")
    ax.set_title("Exp6 full 1-16 thread sweep: workload avg-latency percentiles")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=5, loc="upper left")
    worst = max(rows, key=lambda row: row["avg_ms"])
    ax.annotate(
        f"max {worst['avg_ms']:.2f} ms\n{worst['selector_type']}-{worst['bucket']}, L={worst.get('chosen_L')}",
        xy=(worst["threads_i"], worst["avg_ms"]),
        xytext=(-78, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=8,
    )
    fig.savefig(EXP6 / "latency_percentiles_equality_highres.png", dpi=240)
    fig.savefig(EXP6 / "latency_percentiles_equality_highres.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    selector_styles = {"intersect": ("#7c3aed", "o"), "range": ("#0891b2", "s")}
    for selector, (color, marker) in selector_styles.items():
        for stat, linestyle in [("p95", "-"), ("max", "--")]:
            y = []
            for threads_n in threads:
                vals = np.array(
                    [
                        row["avg_ms"]
                        for row in rows
                        if row["threads_i"] == threads_n and row["selector_type"] == selector
                    ],
                    dtype=float,
                )
                y.append(np.percentile(vals, 95) if stat == "p95" else vals.max())
            ax.plot(threads, y, marker=marker, linestyle=linestyle, linewidth=2, color=color, label=f"{selector} {stat}")
    ax.axhline(10.0, color="#111827", linestyle=":", linewidth=1.1, label="10 ms")
    ax.set_xticks(threads)
    ax.set_xlabel("Query threads")
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Exp6 full sweep by selector type")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(EXP6 / "latency_percentiles_by_selector_highres.png", dpi=240)
    fig.savefig(EXP6 / "latency_percentiles_by_selector_highres.pdf")
    plt.close(fig)

    keys: list[tuple[float, tuple[str, str]]] = []
    for key in {(row["selector_type"], row["bucket"]) for row in rows}:
        vals = [row["avg_ms"] for row in rows if (row["selector_type"], row["bucket"]) == key]
        keys.append((max(vals), key))
    worst_keys = [key for _, key in sorted(keys, reverse=True)[:6]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    palette = ["#dc2626", "#ea580c", "#7c3aed", "#2563eb", "#16a34a", "#0891b2"]
    for (selector, bucket), color in zip(worst_keys, palette):
        line_rows = [row for row in rows if row["selector_type"] == selector and row["bucket"] == bucket]
        by_thread = {row["threads_i"]: row for row in line_rows}
        y = [by_thread[threads_n]["avg_ms"] for threads_n in threads]
        route = by_thread[threads[0]].get("selected_route") or by_thread[threads[0]].get("route")
        chosen_l = by_thread[threads[0]].get("chosen_L")
        ax.plot(threads, y, marker="o", linewidth=2, color=color, label=f"{selector}-{bucket} {route}/L{chosen_l}")
    ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms")
    ax.set_xticks(threads)
    ax.set_xlabel("Query threads")
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Exp6 worst workloads, complete 1-16 sweep")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="upper left")
    fig.savefig(EXP6 / "latency_percentiles_worstcase_highres.png", dpi=240)
    fig.savefig(EXP6 / "latency_percentiles_worstcase_highres.pdf")
    plt.close(fig)


def write_yfcc_label_plot() -> None:
    table = Path("experiments/label_space_ratio/table.csv")
    rows = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
    yfcc = next(row for row in rows if row["dataset"] == "YFCC10M")
    values = [
        float(yfcc["original_mib"]),
        float(yfcc["processed_mib"]),
    ]
    labels = ["YFCC original spmat", "PipeANN hybrid labels"]
    colors = ["#64748b", "#0f766e"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f} MiB",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Disk space (MiB)")
    ax.set_title("YFCC10M label storage: original spmat vs hybrid sidecar")
    ax.grid(axis="y", alpha=0.25)
    ratio = float(yfcc["processed_over_original_percent"])
    ax.text(0.5, 0.88, f"Hybrid = {ratio:.2f}% of original", transform=ax.transAxes, ha="center", fontsize=11)
    fig.savefig("experiments/label_space_ratio/yfcc_label_space_comparison.png", dpi=240)
    fig.savefig("experiments/label_space_ratio/yfcc_label_space_comparison.pdf")
    plt.close(fig)


def write_demand1_plots() -> None:
    PPT_UPDATES.mkdir(parents=True, exist_ok=True)

    direct_rows = list(csv.DictReader((EXP2_L200 / "table.csv").open(newline="", encoding="utf-8")))
    seed_rows = list(csv.DictReader((EXP2_L200 / "seed_sweep_table.csv").open(newline="", encoding="utf-8")))
    all_l200_rows = direct_rows + seed_rows
    max_latency_ms = max(float(row["avg_latency_us"]) / 1000.0 for row in all_l200_rows)
    min_recall = min(float(row["recall@10"]) for row in all_l200_rows)
    with (PPT_UPDATES / "exp2_l200_latency_check.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "max_avg_latency_ms", "value": f"{max_latency_ms:.6f}"},
                {"metric": "min_recall@10", "value": f"{min_recall:.6f}"},
                {"metric": "fixed_L", "value": "200"},
            ]
        )

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    labels = {250000: "250k", 500000: "500k", 750000: "750k", 1000000: "1M"}
    x_order = [250000, 500000, 750000, 1000000]

    def plot_series(rows: list[dict], label: str, color: str, marker: str) -> None:
        by_points = {int(row["points"]): float(row["recall@10"]) for row in rows}
        xs = [points for points in x_order if points in by_points]
        ys = [by_points[points] for points in xs]
        ax.plot(xs, ys, marker=marker, linewidth=2.0, color=color, label=label)

    plot_series([row for row in direct_rows if row["path"] == "direct_build"], "Direct build", "#2563eb", "o")
    plot_series([row for row in direct_rows if row["path"] == "incremental_insert"], "10k seed insert", "#16a34a", "s")
    plot_series([row for row in seed_rows if row["path"] == "direct_seed_250k"], "250k seed insert", "#f59e0b", "^")
    plot_series([row for row in seed_rows if row["path"] == "direct_seed_500k"], "500k seed insert", "#7c3aed", "D")

    ax.axhline(98.0, color="#dc2626", linestyle="--", linewidth=1.2, label="98% target")
    ax.set_xticks(x_order, [labels[x] for x in x_order])
    ax.set_ylim(94.5, 100.2)
    ax.set_xlabel("Live points")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title(f"Demand 1 recall comparison, fixed graph L=200; max avg latency {max_latency_ms:.2f} ms")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="lower left")
    fig.savefig(PPT_UPDATES / "exp2_seed_sweep_recall_L200.png", dpi=240)
    fig.savefig(PPT_UPDATES / "exp2_seed_sweep_recall_L200.pdf")
    plt.close(fig)

    delete_rows = list(csv.DictReader((ROOT / "exp4_delete_reinsert_selectivity" / "table.csv").open(newline="", encoding="utf-8")))
    buckets = ["u25", "u30", "u50", "u75", "u100"]
    bucket_labels = {"u25": "25%", "u30": "30%", "u50": "50%", "u75": "75%", "u100": "100%"}
    states = [
        ("1m_initial", "Initial 1M", "#2563eb", "o"),
        ("750k_after_delete", "After delete", "#f59e0b", "s"),
        ("1m_after_reinsert", "After reinsert", "#16a34a", "^"),
    ]
    high_rows = [row for row in delete_rows if row.get("bucket") in buckets]
    max_delete_latency = max(float(row["avg_latency_us"]) / 1000.0 for row in high_rows)
    min_delete_recall = min(float(row.get("recall@10") or row["recall"]) for row in high_rows)

    with (PPT_UPDATES / "delete_reinsert_calibrated_recall.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["state", "bucket", "selected_route", "chosen_L", "recall@10", "avg_latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in high_rows:
            writer.writerow(
                {
                    "state": row["state"],
                    "bucket": row["bucket"],
                    "selected_route": row["selected_route"],
                    "chosen_L": row["chosen_L"],
                    "recall@10": row.get("recall@10") or row["recall"],
                    "avg_latency_ms": float(row["avg_latency_us"]) / 1000.0,
                }
            )

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    x = np.arange(len(buckets))
    for state, label, color, marker in states:
        state_rows = {row["bucket"]: row for row in high_rows if row["state"] == state}
        y = [float(state_rows[bucket].get("recall@10") or state_rows[bucket]["recall"]) for bucket in buckets]
        ax.plot(x, y, marker=marker, linewidth=2.0, color=color, label=label)
        for xi, bucket in zip(x, buckets):
            row = state_rows[bucket]
            route = "P" if row["selected_route"] == "prefilter" else "G"
            ax.annotate(f"{route}/L{row['chosen_L']}", (xi, float(row.get("recall@10") or row["recall"])),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=color)
    ax.axhline(98.0, color="#dc2626", linestyle="--", linewidth=1.2, label="98% target")
    ax.set_xticks(x, [bucket_labels[bucket] for bucket in buckets])
    ax.set_ylim(97.8, 100.1)
    ax.set_xlabel("Selectivity")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title(f"Delete/reinsert calibrated route/L; max avg latency {max_delete_latency:.2f} ms")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="lower left")
    ax.text(0.01, 0.02, f"min recall {min_delete_recall:.2f}%; P=prefilter, G=graph",
            transform=ax.transAxes, fontsize=8, color="#4b5563")
    fig.savefig(PPT_UPDATES / "delete_reinsert_calibrated_recall.png", dpi=240)
    fig.savefig(PPT_UPDATES / "delete_reinsert_calibrated_recall.pdf")
    plt.close(fig)


def write_exp6_l200_graph_plot() -> None:
    PPT_UPDATES.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in load_exp6_rows()
        if row["selector_type"] == "range"
        and row["bucket"] == "u75"
        and (row.get("selected_route") or row.get("route")) == "graph"
        and row.get("chosen_L") == "200"
    ]
    rows = sorted(rows, key=lambda row: row["threads_i"])
    with (PPT_UPDATES / "exp6_l200_graph_thread_latency.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "threads",
            "avg_latency_ms",
            "p95_query_latency_ms",
            "recall@10",
            "selector_type",
            "bucket",
            "selected_route",
            "chosen_L",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threads": row["threads_i"],
                    "avg_latency_ms": row["avg_ms"],
                    "p95_query_latency_ms": float(row["p95_latency_us"]) / 1000.0,
                    "recall@10": row.get("recall@10"),
                    "selector_type": row["selector_type"],
                    "bucket": row["bucket"],
                    "selected_route": row.get("selected_route") or row.get("route"),
                    "chosen_L": row.get("chosen_L"),
                }
            )

    threads = [row["threads_i"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    series = [
        ("avg", [row["avg_ms"] for row in rows], "#dc2626", "o"),
        ("query p95", [float(row["p95_latency_us"]) / 1000.0 for row in rows], "#f59e0b", "^"),
    ]
    for label, values, color, marker in series:
        ax.plot(threads, values, marker=marker, linewidth=2.0, color=color, label=label)
    ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms budget")
    ax.set_xticks(threads)
    ax.set_xlabel("Query threads")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Exp6 L=200 graph workload: range-u75")
    ax.set_ylim(3.5, 10.8)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=3, loc="upper left")
    ax.annotate(
        "max avg 7.50 ms",
        xy=(16, next(row["avg_ms"] for row in rows if row["threads_i"] == 16)),
        xytext=(-76, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=8,
    )
    fig.savefig(PPT_UPDATES / "exp6_l200_graph_thread_latency.png", dpi=240)
    fig.savefig(PPT_UPDATES / "exp6_l200_graph_thread_latency.pdf")
    plt.close(fig)


def main() -> None:
    write_exp6_plots()
    write_yfcc_label_plot()
    write_demand1_plots()
    write_exp6_l200_graph_plot()
    print("wrote updated exp6 and YFCC label-space plots")


if __name__ == "__main__":
    main()
