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
PPT_RC = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


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


def write_pq_residency_plots() -> None:
    rows = list(csv.DictReader((ROOT / "pq16_pq_residency_compare.csv").open(newline="", encoding="utf-8")))
    plt.rcParams.update(PPT_RC)
    mode_titles = {
        "pq_memory": "PQ code resident in memory",
        "pq_disk_no_cache": "PQ code on disk, no cache",
    }
    metrics = [
        ("latency_avg_ms", "Latency (ms)", "latency_vs_selectivity", "Avg latency"),
        ("adjusted_rss_mib", "Adjusted RSS (MiB)", "adjusted_rss_vs_selectivity", "Adjusted RSS"),
    ]
    selector_styles = {
        "intersect": ("intersect", "#2563eb", "o"),
        "range": ("range", "#dc2626", "s"),
    }
    for mode, mode_title in mode_titles.items():
        mode_rows = [row for row in rows if row["mode"] == mode]
        for metric, ylabel, suffix, title in metrics:
            fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
            for selector, (label, color, marker) in selector_styles.items():
                by_bucket = {row["bucket"]: row for row in mode_rows if row["selector_type"] == selector}
                ordered = [by_bucket[bucket] for bucket in BUCKET_ORDER if bucket in by_bucket]
                x = [BUCKET_ORDER.index(row["bucket"]) for row in ordered]
                y = [float(row[metric]) for row in ordered]
                ax.plot(x, y, marker=marker, linewidth=2.4, markersize=7, color=color, label=label)
            if metric == "adjusted_rss_mib":
                ax.axhline(30.0, color="#111827", linestyle="--", linewidth=1.1, label="30 MiB")
            else:
                ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms")
            ax.set_xticks(range(len(BUCKET_ORDER)), [BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER], rotation=30, ha="right")
            ax.set_xlabel("Selectivity")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: {mode_title}")
            ax.grid(axis="y", alpha=0.28)
            ax.legend(ncol=3, loc="best")
            fig.savefig(ROOT / f"pq16_{mode}_{suffix}.png", dpi=240)
            fig.savefig(ROOT / f"pq16_{mode}_{suffix}.pdf")
            plt.close(fig)


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

    plt.rcParams.update(PPT_RC)

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
        fontsize=11,
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


def write_sift1m_label_plot() -> None:
    table = Path("experiments/label_space_ratio/table.csv")
    rows = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
    sift = next(row for row in rows if row["dataset"] == "SIFT1M/r116")
    values = [
        float(sift["original_mib"]),
        float(sift["processed_mib"]),
    ]
    labels = ["SIFT1M/r116 original spmat", "PipeANN hybrid labels"]
    colors = ["#64748b", "#0f766e"]
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylim(0, max(values) * 1.24)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f} MiB",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set_ylabel("Disk space (MiB)")
    ax.set_title("SIFT1M/r116 label storage")
    ax.grid(axis="y", alpha=0.25)
    ratio = float(sift["processed_over_original_percent"])
    ax.text(0.5, 0.82, f"Hybrid = {ratio:.2f}% of original", transform=ax.transAxes, ha="center", fontsize=13)
    fig.savefig("experiments/label_space_ratio/sift1m_label_space_comparison.png", dpi=240)
    fig.savefig("experiments/label_space_ratio/sift1m_label_space_comparison.pdf")
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

    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
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
    ax.set_title(f"Recall comparison at L=200; max avg {max_latency_ms:.2f} ms")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="lower left")
    fig.savefig(PPT_UPDATES / "exp2_seed_sweep_recall_L200.png", dpi=240)
    fig.savefig(PPT_UPDATES / "exp2_seed_sweep_recall_L200.pdf")
    plt.close(fig)

    fixed_delete_path = ROOT / "exp4_delete_reinsert_fixed_params" / "fixed_params_fullgt_table.csv"
    delete_rows = list(csv.DictReader(fixed_delete_path.open(newline="", encoding="utf-8")))
    buckets = ["u25", "u30", "u50", "u75", "u100"]
    bucket_labels = {"u25": "25%", "u30": "30%", "u50": "50%", "u75": "75%", "u100": "100%"}
    states = [
        ("1m_initial", "Initial 1M", "#2563eb", "o"),
        ("750k_after_delete", "After delete", "#f59e0b", "s"),
        ("1m_after_reinsert", "After reinsert", "#16a34a", "^"),
    ]
    high_rows = [row for row in delete_rows if row.get("bucket") in buckets]
    max_delete_latency = max(float(row.get("avg_latency_ms") or float(row["avg_latency_us"]) / 1000.0) for row in high_rows)

    with (PPT_UPDATES / "delete_reinsert_fixed_fullgt_recall.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["state", "bucket", "selected_route", "chosen_L", "recall@10", "avg_latency_ms", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in high_rows:
            writer.writerow(
                {
                    "state": row["state"],
                    "bucket": row["bucket"],
                    "selected_route": row["selected_route"],
                    "chosen_L": row["chosen_L"],
                    "recall@10": row.get("recall@10") or row["recall"],
                    "avg_latency_ms": row.get("avg_latency_ms") or float(row["avg_latency_us"]) / 1000.0,
                    "status": row.get("status", ""),
                }
            )

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(buckets))
    params_by_bucket = {}
    for state, label, color, marker in states:
        state_rows = {row["bucket"]: row for row in high_rows if row["state"] == state}
        y = [float(state_rows[bucket].get("recall@10") or state_rows[bucket]["recall"]) for bucket in buckets]
        ax.plot(x, y, marker=marker, linewidth=2.0, color=color, label=label)
        for xi, bucket in zip(x, buckets):
            row = state_rows[bucket]
            route = "P" if row["selected_route"] == "prefilter" else "G"
            params_by_bucket.setdefault(bucket, f"{route}/L{row['chosen_L']}")
            if row.get("status") != "ok":
                ax.scatter([xi], [float(row.get("recall@10") or row["recall"])], marker="x",
                           s=90, linewidths=2.4, color="#dc2626", zorder=5)
                ax.annotate("1M GT includes deleted vectors",
                            (xi, float(row.get("recall@10") or row["recall"])),
                            textcoords="offset points", xytext=(-22, -22), ha="right",
                            fontsize=10, color="#dc2626")
    for xi, bucket in zip(x, buckets):
        ax.annotate(params_by_bucket[bucket], (xi, 100.05), textcoords="offset points",
                    xytext=(0, 0), ha="center", va="bottom", fontsize=10, color="#374151")
    ax.axhline(98.0, color="#dc2626", linestyle="--", linewidth=1.2, label="98% target")
    ax.set_xticks(x, [bucket_labels[bucket] for bucket in buckets])
    ax.set_ylim(74.0, 101.2)
    ax.set_xlabel("Selectivity")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title(f"Fixed route/L, no auto; max avg {max_delete_latency:.2f} ms")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="lower left")
    fig.savefig(PPT_UPDATES / "delete_reinsert_fixed_fullgt_recall.png", dpi=240)
    fig.savefig(PPT_UPDATES / "delete_reinsert_fixed_fullgt_recall.pdf")
    plt.close(fig)


def write_exp6_demand3_worst_plot() -> None:
    PPT_UPDATES.mkdir(parents=True, exist_ok=True)
    exp4_rows = list(csv.DictReader((ROOT / "exp4_intersect_range_selectivity" / "table.csv").open(newline="", encoding="utf-8")))
    worst = max(exp4_rows, key=lambda row: float(row["avg_latency_us"]))
    worst_route = worst.get("selected_route") or worst.get("route")
    worst_l = str(worst.get("chosen_L"))
    rows = [
        row
        for row in load_exp6_rows()
        if row["selector_type"] == worst["selector_type"]
        and row["bucket"] == worst["bucket"]
        and (row.get("selected_route") or row.get("route")) == worst_route
        and str(row.get("chosen_L")) == worst_l
    ]
    rows = sorted(rows, key=lambda row: row["threads_i"])
    if not rows:
        raise RuntimeError(f"no exp6 rows match demand3 worst row: {worst}")
    with (PPT_UPDATES / "exp6_demand3_worst_thread_latency.csv").open("w", newline="", encoding="utf-8") as f:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
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
    plt.rcParams.update(PPT_RC)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    avg_values = [row["avg_ms"] for row in rows]
    p95_values = [float(row["p95_latency_us"]) / 1000.0 for row in rows]
    series = [
        ("avg", avg_values, "#dc2626", "o"),
        ("query p95", p95_values, "#f59e0b", "^"),
    ]
    for label, values, color, marker in series:
        ax.plot(threads, values, marker=marker, linewidth=2.0, color=color, label=label)
    ax.axhline(10.0, color="#111827", linestyle="--", linewidth=1.1, label="10 ms budget")
    ax.set_xticks(threads)
    ax.set_xlabel("Query threads")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Demand 3 worst workload: {worst['selector_type']}-{worst['bucket']} {worst_route}/L{worst_l}")
    ax.set_ylim(max(0.0, min(avg_values + p95_values) - 0.8), max(10.0, max(avg_values + p95_values)) + 0.9)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=3, loc="upper left")
    max_avg_thread = max(rows, key=lambda row: row["avg_ms"])
    max_p95_thread = max(rows, key=lambda row: float(row["p95_latency_us"]))
    ax.annotate(
        f"max avg {max_avg_thread['avg_ms']:.2f} ms",
        xy=(max_avg_thread["threads_i"], max_avg_thread["avg_ms"]),
        xytext=(-84, -24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=11,
    )
    ax.annotate(
        f"max p95 {float(max_p95_thread['p95_latency_us']) / 1000.0:.2f} ms",
        xy=(max_p95_thread["threads_i"], float(max_p95_thread["p95_latency_us"]) / 1000.0),
        xytext=(-88, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=11,
    )
    fig.savefig(PPT_UPDATES / "exp6_demand3_worst_thread_latency.png", dpi=240)
    fig.savefig(PPT_UPDATES / "exp6_demand3_worst_thread_latency.pdf")
    plt.close(fig)


def main() -> None:
    write_pq_residency_plots()
    write_exp6_plots()
    write_sift1m_label_plot()
    write_demand1_plots()
    write_exp6_demand3_worst_plot()
    print("wrote updated exp6 and SIFT1M label-space plots")


if __name__ == "__main__":
    main()
