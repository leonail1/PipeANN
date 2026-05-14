#!/usr/bin/env python3
"""Plot qps_4sets formal reproduction results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "experiments" / "qps_4sets" / "summary" / "formal_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "qps_4sets" / "summary"
DEFAULT_LEGACY_QPS_OUTPUT = REPO_ROOT / "experiments" / "qps_4sets.png"

DATASET_LABELS = {
    "fashion_mnist784": "Fashion-MNIST",
    "glove100": "GloVe100",
    "gist960": "GIST960",
    "yfcc10m": "YFCC10M",
}

DATASET_ORDER = ["fashion_mnist784", "glove100", "gist960", "yfcc10m"]
THREAD_ORDER = [1, 2, 4, 8]
THREAD_MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}
ROUTE_COLORS = {"prefilter": "#26734d", "graph": "#b4552a", "fallback": "#6a6a6a"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def to_optional_float(value: str | None) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def route_for(row: dict[str, str]) -> str:
    if int(row["prefilter_count"]) > 0:
        return "prefilter"
    if int(row["graph_count"]) > 0:
        return "graph"
    return "fallback"


def grouped_by_dataset_bucket(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["bucket_name"])].append(row)
    return grouped


def sorted_dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (to_float(row["selectivity_midpoint"]), int(row["threads"])))


def percent_label(value: float) -> str:
    if value < 0.01:
        return f"{value * 100:.2f}%"
    if value < 0.1:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.0f}%"


def set_common_style(savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.dpi": 120,
            "savefig.dpi": savefig_dpi,
        }
    )


def plot_qps(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        dataset_rows = sorted_dataset_rows([row for row in rows if row["dataset"] == dataset])
        for thread in THREAD_ORDER:
            thread_rows = [row for row in dataset_rows if int(row["threads"]) == thread]
            x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in thread_rows]
            y_values = [to_float(row["qps"]) for row in thread_rows]
            axis.plot(x_values, y_values, marker=THREAD_MARKERS[thread], linewidth=1.8, label=f"t={thread}")

        for row in dataset_rows:
            if int(row["threads"]) != 1:
                continue
            color = ROUTE_COLORS[route_for(row)]
            axis.axvline(to_float(row["selectivity_midpoint"]) * 100, color=color, linewidth=0.9, alpha=0.18)

        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xscale("log")
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("QPS")
        axis.legend(ncol=2, frameon=False)

    fig.suptitle("qps_4sets reproduction: QPS by selectivity and threads", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    grouped = grouped_by_dataset_bucket(rows)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        dataset_points = []
        for (row_dataset, _bucket), bucket_rows in grouped.items():
            if row_dataset != dataset:
                continue
            by_thread = {int(row["threads"]): row for row in bucket_rows}
            if not all(thread in by_thread for thread in THREAD_ORDER):
                continue
            qps_1 = to_float(by_thread[1]["qps"])
            qps_8 = to_float(by_thread[8]["qps"])
            if qps_1 <= 0:
                continue
            representative = by_thread[1]
            dataset_points.append(
                {
                    "selectivity": to_float(representative["selectivity_midpoint"]) * 100,
                    "efficiency": qps_8 / qps_1 / 8.0,
                    "speedup": qps_8 / qps_1,
                    "route": route_for(representative),
                }
            )

        dataset_points.sort(key=lambda item: item["selectivity"])
        for route in ["prefilter", "graph", "fallback"]:
            points = [item for item in dataset_points if item["route"] == route]
            if not points:
                continue
            axis.plot(
                [item["selectivity"] for item in points],
                [item["efficiency"] for item in points],
                marker="o",
                linewidth=1.8,
                color=ROUTE_COLORS[route],
                label=route,
            )
            for item in points:
                axis.annotate(f"{item['speedup']:.1f}x", (item["selectivity"], item["efficiency"]),
                              textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.55, label="linear")
        axis.set_ylim(0, 1.05)
        axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("8-thread efficiency vs t=1")
        axis.legend(frameon=False)

    fig.suptitle("qps_4sets reproduction: scaling efficiency exposes non-linear bottlenecks", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_route_io(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        dataset_rows = sorted_dataset_rows([row for row in rows if row["dataset"] == dataset and int(row["threads"]) == 1])
        x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in dataset_rows]
        mean_ios = [to_float(row["mean_ios"]) for row in dataset_rows]
        candidates = [to_float(row["mean_candidate_count"]) for row in dataset_rows]

        axis.plot(x_values, mean_ios, color="#3b5f8a", marker="o", linewidth=1.8, label="Mean IOs/query")
        twin = axis.twinx()
        twin.plot(x_values, candidates, color="#8a3b52", marker="s", linewidth=1.5, label="Mean candidates")
        twin.set_yscale("log")
        twin.set_ylabel("Mean candidates/query")

        for row in dataset_rows:
            axis.scatter(
                [to_float(row["selectivity_midpoint"]) * 100],
                [to_float(row["mean_ios"])],
                color=ROUTE_COLORS[route_for(row)],
                s=48,
                zorder=4,
            )

        axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("Mean IOs/query")

        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, frameon=False, loc="upper left")

    fig.suptitle("qps_4sets reproduction: route behavior and per-query IO/candidate pressure", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_diagnostics(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        dataset_rows = sorted_dataset_rows([row for row in rows if row["dataset"] == dataset])
        for thread in THREAD_ORDER:
            thread_rows = [row for row in dataset_rows if int(row["threads"]) == thread]
            x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in thread_rows]
            y_values = [to_float(row["avg_cpu_pct"]) / max(thread, 1) for row in thread_rows]
            axis.plot(
                x_values,
                y_values,
                marker=THREAD_MARKERS[thread],
                linewidth=1.7,
                label=f"CPU/thread, t={thread}",
            )

        for row in dataset_rows:
            if int(row["threads"]) != 1:
                continue
            axis.scatter(
                [to_float(row["selectivity_midpoint"]) * 100],
                [to_float(row["avg_cpu_pct"])],
                color=ROUTE_COLORS[route_for(row)],
                s=42,
                zorder=4,
            )

        axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("Avg CPU pct per requested thread")
        axis.legend(frameon=False, ncol=2)

    fig.suptitle("qps_4sets reproduction: CPU activity", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_disk_diagnostics(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        thread1_rows = sorted_dataset_rows([row for row in rows if row["dataset"] == dataset and int(row["threads"]) == 1])
        ok_rows = [row for row in thread1_rows if row.get("disk_metrics_status", "ok") in ("", "ok")]
        invalid_rows = [row for row in thread1_rows if row.get("disk_metrics_status", "ok") not in ("", "ok")]

        x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in ok_rows]
        read_values = [to_optional_float(row.get("avg_read_mb_s")) for row in ok_rows]
        util_values = [to_optional_float(row.get("avg_disk_util_pct")) for row in ok_rows]
        qd_values = [to_optional_float(row.get("avg_qd")) for row in ok_rows]

        read_points = [(x, y) for x, y in zip(x_values, read_values) if y is not None]
        util_points = [(x, y) for x, y in zip(x_values, util_values) if y is not None]
        qd_points = [(x, y) for x, y in zip(x_values, qd_values) if y is not None]
        if read_points:
            axis.plot([x for x, _ in read_points], [y for _, y in read_points], marker="o", linewidth=1.7, label="Read MB/s")
        if util_points:
            axis.plot([x for x, _ in util_points], [y for _, y in util_points], marker="x", linewidth=1.4, linestyle="--", label="Disk util %")
        if qd_points:
            axis.plot([x for x, _ in qd_points], [y for _, y in qd_points], marker="+", linewidth=1.4, linestyle=":", label="Avg QD")

        for row in invalid_rows:
            x_value = to_float(row["selectivity_midpoint"]) * 100
            axis.axvline(x_value, color="#6a6a6a", linewidth=0.9, alpha=0.22)
            axis.annotate(
                "unavailable",
                (x_value, 0.02),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
                color="#6a6a6a",
            )

        axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("Disk metrics, t=1")
        axis.legend(frameon=False)

    fig.suptitle("qps_4sets reproduction: disk metrics with invalid counters marked", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_bottleneck_diagnostics(rows: list[dict[str, str]], output_path: Path, savefig_dpi: int = 400) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    set_common_style(savefig_dpi)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        dataset_rows = sorted_dataset_rows([row for row in rows if row["dataset"] == dataset])
        for thread in THREAD_ORDER:
            thread_rows = [row for row in dataset_rows if int(row["threads"]) == thread]
            x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in thread_rows]
            y_values = [to_float(row["avg_cpu_pct"]) / max(thread, 1) for row in thread_rows]
            axis.plot(
                x_values,
                y_values,
                marker=THREAD_MARKERS[thread],
                linewidth=1.7,
                color="#315f8d",
                alpha=0.35 + 0.08 * thread,
                label=f"CPU/thread, t={thread}",
            )

        thread1_rows = [row for row in dataset_rows if int(row["threads"]) == 1]
        x_values = [to_float(row["selectivity_midpoint"]) * 100 for row in thread1_rows]
        disk_values = [to_float(row["avg_disk_util_pct"]) for row in thread1_rows]
        read_values = [to_float(row["avg_read_mb_s"]) for row in thread1_rows]

        twin = axis.twinx()
        twin.plot(
            x_values,
            disk_values,
            marker="x",
            linestyle="--",
            linewidth=1.4,
            color="#8a3b52",
            label="Disk util %, t=1",
        )
        twin.plot(
            x_values,
            read_values,
            marker="+",
            linestyle=":",
            linewidth=1.4,
            color="#b4552a",
            label="Read MB/s, t=1",
        )
        twin.set_ylabel("Disk util (%) / read MB/s")

        for row in thread1_rows:
            axis.scatter(
                [to_float(row["selectivity_midpoint"]) * 100],
                [to_float(row["avg_cpu_pct"])],
                color=ROUTE_COLORS[route_for(row)],
                s=48,
                zorder=4,
            )

        axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Selectivity (%)")
        axis.set_ylabel("Avg CPU pct per requested thread")

        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        route_handles = [
            mlines.Line2D([], [], color=color, marker="o", linestyle="None", label=f"route: {route}")
            for route, color in ROUTE_COLORS.items()
        ]
        axis.legend(handles[:2] + twin_handles + route_handles[:2], labels[:2] + twin_labels + [h.get_label() for h in route_handles[:2]], frameon=False)

    fig.suptitle("qps_4sets reproduction: CPU activity vs observed SSD pressure", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_scaling_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped = grouped_by_dataset_bucket(rows)
    fields = [
        "dataset",
        "bucket_name",
        "selectivity_midpoint",
        "route_decision",
        "qps_t1",
        "qps_t2",
        "qps_t4",
        "qps_t8",
        "speedup_t8_vs_t1",
        "efficiency_t8_vs_t1",
        "mean_ios_t1",
        "mean_candidate_count_t1",
        "avg_read_mb_s_t1",
        "avg_disk_util_pct_t1",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_ORDER:
            bucket_items = [item for item in grouped.items() if item[0][0] == dataset]
            bucket_items.sort(key=lambda item: to_float(item[1][0]["selectivity_midpoint"]))
            for (_, bucket), bucket_rows in bucket_items:
                by_thread = {int(row["threads"]): row for row in bucket_rows}
                if not all(thread in by_thread for thread in THREAD_ORDER):
                    continue
                qps_1 = to_float(by_thread[1]["qps"])
                qps_8 = to_float(by_thread[8]["qps"])
                writer.writerow(
                    {
                        "dataset": dataset,
                        "bucket_name": bucket,
                        "selectivity_midpoint": by_thread[1]["selectivity_midpoint"],
                        "route_decision": route_for(by_thread[1]),
                        "qps_t1": f"{qps_1:.6f}",
                        "qps_t2": f"{to_float(by_thread[2]['qps']):.6f}",
                        "qps_t4": f"{to_float(by_thread[4]['qps']):.6f}",
                        "qps_t8": f"{qps_8:.6f}",
                        "speedup_t8_vs_t1": f"{qps_8 / qps_1:.6f}",
                        "efficiency_t8_vs_t1": f"{qps_8 / qps_1 / 8.0:.6f}",
                        "mean_ios_t1": by_thread[1]["mean_ios"],
                        "mean_candidate_count_t1": by_thread[1]["mean_candidate_count"],
                        "avg_read_mb_s_t1": by_thread[1]["avg_read_mb_s"],
                        "avg_disk_util_pct_t1": by_thread[1]["avg_disk_util_pct"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--legacy-qps-output",
        type=Path,
        default=DEFAULT_LEGACY_QPS_OUTPUT,
        help="Also write the main QPS figure to the historical qps_4sets.png path.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="DPI used for saved PNG figures.")
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_qps_path = args.out_dir / "qps_4sets_reproduction_qps.png"
    plot_qps(rows, main_qps_path, args.dpi)
    if args.legacy_qps_output:
        plot_qps(rows, args.legacy_qps_output, args.dpi)
    plot_scaling(rows, args.out_dir / "qps_4sets_scaling_efficiency.png", args.dpi)
    plot_route_io(rows, args.out_dir / "qps_4sets_route_io_pressure.png", args.dpi)
    plot_cpu_diagnostics(rows, args.out_dir / "qps_4sets_cpu_diagnostics.png", args.dpi)
    plot_disk_diagnostics(rows, args.out_dir / "qps_4sets_disk_diagnostics.png", args.dpi)
    plot_bottleneck_diagnostics(rows, args.out_dir / "qps_4sets_bottleneck_diagnostics.png", args.dpi)
    write_scaling_csv(rows, args.out_dir / "scaling_efficiency.csv")

    print(main_qps_path)
    if args.legacy_qps_output:
        print(args.legacy_qps_output)
    print(args.out_dir / "qps_4sets_scaling_efficiency.png")
    print(args.out_dir / "qps_4sets_route_io_pressure.png")
    print(args.out_dir / "qps_4sets_cpu_diagnostics.png")
    print(args.out_dir / "qps_4sets_disk_diagnostics.png")
    print(args.out_dir / "qps_4sets_bottleneck_diagnostics.png")
    print(args.out_dir / "scaling_efficiency.csv")


if __name__ == "__main__":
    main()
