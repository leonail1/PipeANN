#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a PQ8/PQ16/PQ32 prefilter comparison figure.")
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "RESULTS_JSONL"),
        required=True,
        help="Series label and results jsonl path. Repeat for each PQ variant.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-l", type=int, default=100)
    parser.add_argument("--title")
    parser.add_argument("--route", default="prefilter")
    return parser.parse_args()


def load_series(label: str, path: Path, route: str, plot_l: int) -> list[dict]:
    records = [record for record in read_jsonl(path) if record.get("format") == "pipeann.hybrid.search.v1"]
    records = [record for record in records if str(record.get("route")) == route and int(record.get("L", -1)) == plot_l]
    if not records:
        raise ValueError(f"no records for label={label}, route={route}, L={plot_l} in {path}")
    records.sort(key=lambda record: (float(record["selectivity_midpoint"]), str(record["bucket_name"])))
    return records


def main() -> int:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for label, raw_path in args.series:
        path = Path(raw_path)
        series.append((label, load_series(label, path, args.route, args.plot_l)))

    first_selectivities = [float(record["selectivity_midpoint"]) for record in series[0][1]]
    first_bucket_names = [str(record["bucket_name"]) for record in series[0][1]]
    for label, records in series[1:]:
        selectivities = [float(record["selectivity_midpoint"]) for record in records]
        bucket_names = [str(record["bucket_name"]) for record in records]
        if selectivities != first_selectivities or bucket_names != first_bucket_names:
            raise ValueError(f"series {label} does not align with the first series")

    x_positions = list(range(len(first_selectivities)))
    x_labels = [str(record["bucket_label"]) for record in series[0][1]]
    metric_specs = (
        ("avg_latency_us", "Latency (ms)", lambda value: value / 1000.0),
        ("qps", "QPS", lambda value: value),
        ("peak_memory_kb", "Peak RSS (MiB)", lambda value: value / 1024.0),
    )
    colors = {
        "PQ8": "#0b6e4f",
        "PQ16": "#c84c09",
        "PQ32": "#0b4f8c",
    }
    markers = {
        "PQ8": "o",
        "PQ16": "s",
        "PQ32": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), constrained_layout=True)
    for axis, (metric_key, axis_title, transform) in zip(axes, metric_specs):
        for label, records in series:
            y_values = [transform(float(record[metric_key])) for record in records]
            axis.plot(
                x_positions,
                y_values,
                label=label,
                color=colors.get(label, "#333333"),
                marker=markers.get(label, "o"),
                linewidth=2.0,
                markersize=6.0,
            )

        axis.set_title(axis_title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(x_labels, rotation=30, ha="right")
        axis.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    axes[0].set_ylabel("Value")
    axes[1].legend(loc="best")
    fig.suptitle(args.title or f"PipeANN prefilter comparison by selectivity (L={args.plot_l})", fontsize=14)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())