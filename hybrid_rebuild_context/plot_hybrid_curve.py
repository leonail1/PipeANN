#!/usr/bin/env python3
"""Canonical plotter for the latest hybrid combined curve."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from exact_hybrid_common import DEFAULT_SELECTIVITIES, artifacts_images_dir, artifacts_results_v2_dir, format_sel

CSV_PATH = artifacts_results_v2_dir() / "hybrid_results.csv"
RESULTS_DIR = artifacts_images_dir()
RECALL_TARGET = 98.0
ORDERED_SELS = [float(format_sel(sel)) for sel in DEFAULT_SELECTIVITIES]
PLOT_TITLE = (
    "Hybrid Filtered Search: Latency & Peak RSS vs Selectivity\\n"
    "(SIFT-1M, K=10, PQ16 prefilter + PQ16 graph-only)"
)


def load_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with CSV_PATH.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[format_sel(float(row["selectivity"]))] = row
    return rows


def maybe_float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def build_series() -> tuple[list[float], list[float | None], list[float | None]]:
    rows_by_sel = load_rows()
    sels_actual: list[float] = []
    hybrid_lat: list[float | None] = []
    hybrid_rss: list[float | None] = []

    for sel in ORDERED_SELS:
        row = rows_by_sel.get(format_sel(sel))
        if row is None:
            continue

        pf_us = maybe_float(row["pf_latency_us"])
        pf_recall = maybe_float(row["pf_recall"])
        gr_us = maybe_float(row["gr_latency_us"])
        gr_recall = maybe_float(row["gr_recall"])
        pf_rss = maybe_float(row["pf_rss_mb"])
        gr_rss = maybe_float(row["gr_rss_mb"])

        pf_ok = pf_recall is not None and pf_recall >= RECALL_TARGET
        gr_ok = gr_recall is not None and gr_recall >= RECALL_TARGET
        pf_lat_ms = (pf_us / 1000.0) if pf_ok and pf_us is not None else None
        gr_lat_ms = (gr_us / 1000.0) if gr_ok and gr_us is not None else None

        sels_actual.append(sel)
        if pf_lat_ms is not None and gr_lat_ms is not None:
            hybrid_lat.append(min(pf_lat_ms, gr_lat_ms))
        else:
            hybrid_lat.append(pf_lat_ms if pf_lat_ms is not None else gr_lat_ms)

        rss_candidates = [value for value in (pf_rss if pf_ok else None, gr_rss if gr_ok else None) if value is not None]
        hybrid_rss.append(min(rss_candidates) if rss_candidates else None)

    return sels_actual, hybrid_lat, hybrid_rss


def plot_with_matplotlib(sels_actual: list[float], hybrid_lat: list[float | None],
                         hybrid_rss: list[float | None], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sels_pct = np.array(sels_actual) * 100
    ticks = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100]

    fig, ax_lat = plt.subplots(figsize=(10, 5.5))
    ax_rss = ax_lat.twinx()

    hy_x = [sels_pct[i] for i, value in enumerate(hybrid_lat) if value is not None]
    hy_y = [value for value in hybrid_lat if value is not None]
    hy_rss_x = [sels_pct[i] for i, value in enumerate(hybrid_rss) if value is not None]
    hy_rss_y = [value for value in hybrid_rss if value is not None]

    ax_lat.plot(hy_x, hy_y, "D-", color="#2ca02c", linewidth=2.5, markersize=7, label="Hybrid Latency", zorder=5)
    ax_rss.plot(hy_rss_x, hy_rss_y, "s--", color="#9467bd", linewidth=2.5, markersize=7, label="Hybrid RSS",
                zorder=5)
    ax_rss.axhline(y=30, color="gray", linestyle=":", alpha=0.7, linewidth=1.5, label="30 MB")

    ax_lat.set_xlabel("Selectivity (%)", fontsize=13)
    ax_lat.set_ylabel("Latency (ms)", fontsize=13, color="#2ca02c")
    ax_rss.set_ylabel("Single-query Peak RSS (MB)", fontsize=13, color="#9467bd")
    ax_lat.set_xscale("log")
    ax_lat.set_ylim(bottom=0, top=max(hy_y) * 1.3 if hy_y else 1)
    ax_rss.set_ylim(bottom=0, top=max(35.0, max(hy_rss_y) * 1.25 if hy_rss_y else 35.0))
    ax_lat.set_xticks(ticks)
    ax_lat.set_xticklabels([f"{tick}%" for tick in ticks])
    ax_lat.grid(True, alpha=0.3)
    ax_lat.tick_params(axis="y", labelcolor="#2ca02c")
    ax_rss.tick_params(axis="y", labelcolor="#9467bd")

    lines1, labels1 = ax_lat.get_legend_handles_labels()
    lines2, labels2 = ax_rss.get_legend_handles_labels()
    ax_lat.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left", framealpha=0.9)
    ax_lat.set_title(PLOT_TITLE.replace("\\n", "\n"), fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_with_gnuplot(sels_actual: list[float], hybrid_lat: list[float | None],
                      hybrid_rss: list[float | None], output_path: Path) -> None:
    gnuplot = shutil.which("gnuplot")
    if gnuplot is None:
        raise RuntimeError("Neither matplotlib nor gnuplot is available for plotting")

    with tempfile.TemporaryDirectory(prefix="hybrid_plot_") as tmpdir:
        data_path = Path(tmpdir) / "hybrid_curve.tsv"
        script_path = Path(tmpdir) / "hybrid_curve.gp"

        with data_path.open("w", encoding="utf-8") as handle:
            handle.write("selectivity_pct\tlatency_ms\trss_mb\n")
            for sel, lat, rss in zip(sels_actual, hybrid_lat, hybrid_rss):
                lat_value = "NaN" if lat is None else f"{lat:.6f}"
                rss_value = "NaN" if rss is None else f"{rss:.6f}"
                handle.write(f"{sel * 100:.6f}\t{lat_value}\t{rss_value}\n")

        lat_max = max((value for value in hybrid_lat if value is not None), default=1.0)
        rss_max = max((value for value in hybrid_rss if value is not None), default=35.0)
        script = f"""
set terminal pngcairo size 1400,800 noenhanced font 'DejaVuSans,14'
set output '{output_path}'
set datafile separator '\\t'
set title '{PLOT_TITLE}'
set xlabel 'Selectivity (percent)'
set ylabel 'Latency (ms)'
set y2label 'Single-query Peak RSS (MB)'
set logscale x
set xrange [0.08:120]
set yrange [0:{max(1.0, lat_max * 1.3)}]
set y2range [0:{max(35.0, rss_max * 1.25)}]
set xtics ('0.1' 0.1, '0.5' 0.5, '1' 1, '2' 2, '5' 5, '10' 10, '25' 25, '50' 50, '100' 100)
set ytics nomirror
set y2tics
set grid xtics ytics
set key left top
plot '{data_path}' using 1:2 axes x1y1 with linespoints linewidth 2 pointtype 7 pointsize 1.2 linecolor rgb '#2ca02c' title 'Hybrid Latency', \\
     '{data_path}' using 1:3 axes x1y2 with linespoints linewidth 2 pointtype 5 pointsize 1.2 dashtype 2 linecolor rgb '#9467bd' title 'Hybrid RSS', \\
     30 axes x1y2 with lines linewidth 1 dashtype 3 linecolor rgb 'gray' title '30 MB'
"""
        script_path.write_text(script, encoding="utf-8")
        subprocess.run([gnuplot, str(script_path)], check=True)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"missing {CSV_PATH}")

    sels_actual, hybrid_lat, hybrid_rss = build_series()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "hybrid_combined.png"

    try:
        plot_with_matplotlib(sels_actual, hybrid_lat, hybrid_rss, output_path)
    except ModuleNotFoundError:
        plot_with_gnuplot(sels_actual, hybrid_lat, hybrid_rss, output_path)


if __name__ == "__main__":
    main()
