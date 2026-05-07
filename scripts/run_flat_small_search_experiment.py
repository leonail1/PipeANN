#!/usr/bin/env python3
"""Run the <=10k flat-index search RSS/latency/QPS experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


CSV_FIELDS = [
    "format",
    "dataset",
    "points",
    "dim",
    "threshold",
    "flat_mode",
    "live_point_count",
    "rss_query_count",
    "rss_before_single_query_kb",
    "rss_after_single_query_kb",
    "rss_single_query_peak_kb",
    "rss_single_query_delta_kb",
    "rss_after_insert_kb",
    "process_max_rss_kb",
    "qps_repeats",
    "elapsed_s",
    "qps",
    "avg_latency_us",
    "p50_latency_us",
    "p95_latency_us",
    "p99_latency_us",
    "insert_elapsed_s",
    "insert_qps",
]


def parse_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        size = int(token)
        if size < 0:
            raise argparse.ArgumentTypeError("sizes must be non-negative")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/flat_small_search"))
    parser.add_argument("--base-bin", type=Path, default=Path("data/sift1m/sift_base.bin"))
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("1,10,100,1000,5000,9999,10000"))
    parser.add_argument("--threshold", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--qps-repeats", type=int, default=1000)
    parser.add_argument("--rss-query-id", type=int, default=0)
    parser.add_argument("--query-buffer-count", type=int, default=16)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], cwd: Path) -> None:
    log("+ " + " ".join(command))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, cwd=cwd, check=True, env=env)


def build_benchmark(repo: Path) -> None:
    if not (repo / "build" / "CMakeCache.txt").exists():
        run(["cmake", "-S", ".", "-B", "build"], repo)
    run(["cmake", "--build", "build", "--target", "dynamic_flat_small_search_bench", "-j"], repo)


def cleanup_prefix(prefix: Path) -> None:
    parent = prefix.parent
    if not parent.exists():
        return
    for path in parent.glob(prefix.name + "*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def parse_json_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("benchmark did not emit a JSON object on stdout")


def run_one(repo: Path, args: argparse.Namespace, points: int, run_id: str) -> dict:
    bench = repo / "build" / "tests" / "dynamic_flat_small_search_bench"
    prefix = Path("/tmp") / f"pipeann_flat_small_search_{run_id}_{points}"
    cleanup_prefix(prefix)

    command = [
        str(bench),
        "--base-bin",
        str(args.base_bin),
        "--points",
        str(points),
        "--threshold",
        str(args.threshold),
        "--k",
        str(args.k),
        "--threads",
        str(args.threads),
        "--qps-repeats",
        str(args.qps_repeats),
        "--rss-query-id",
        str(args.rss_query_id),
        "--query-buffer-count",
        str(args.query_buffer_count),
        "--index-prefix",
        str(prefix),
    ]
    log("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        cleanup_prefix(prefix)
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)

    row = parse_json_stdout(completed.stdout)
    if row.get("flat_mode") is not True:
        raise RuntimeError(f"points={points} did not remain in flat mode")
    if int(row.get("rss_query_count", 0)) != 1:
        raise RuntimeError(f"points={points} did not use single-query RSS")
    for suffix in ("_disk.index", "_pq_pivots.bin", "_pq_compressed.bin"):
        artifact = Path(str(prefix) + suffix)
        if artifact.exists():
            cleanup_prefix(prefix)
            raise RuntimeError(f"unexpected disk artifact remains: {artifact}")
    cleanup_prefix(prefix)
    return row


def write_results(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "table.csv").open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=CSV_FIELDS, extrasaction="ignore")
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def gnuplot_quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_plot(out_dir: Path, rows: list[dict]) -> Path | None:
    png_path = out_dir / "rss_latency_qps.png"
    ordered = sorted(rows, key=lambda row: int(row["points"]))

    try:
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

        x_labels = [str(int(row["points"])) for row in ordered]
        x = list(range(len(ordered)))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)

        axes[0].plot(x, [float(row["rss_single_query_peak_kb"]) / 1024.0 for row in ordered],
                     marker="o", linewidth=2.0, color="#1d4ed8", label="single-query peak RSS")
        axes[0].set_ylabel("RSS (MiB)")
        axes[0].set_title("RSS")

        axes[1].plot(x, [float(row["avg_latency_us"]) / 1000.0 for row in ordered],
                     marker="o", linewidth=2.0, color="#1d4ed8", label="avg latency")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Latency")

        axes[2].plot(x, [float(row["qps"]) for row in ordered],
                     marker="o", linewidth=2.0, color="#7c3aed", label="QPS")
        axes[2].set_ylabel("QPS")
        axes[2].set_title("QPS")
        axes[2].set_yscale("log")

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_xlabel("Indexed points")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

        fig.suptitle("Flat small search: RSS, latency and QPS")
        fig.savefig(png_path, dpi=240)
        plt.close(fig)
        log(f"wrote {png_path}")
        return png_path
    except ImportError:
        pass

    gnuplot = shutil.which("gnuplot")
    if gnuplot is None:
        log("gnuplot not found; skipped plot generation")
        return None

    data_lines = [
        "# idx points rss_after_insert_mib rss_single_peak_mib process_max_mib avg_ms p95_ms p99_ms qps",
    ]
    xtics = []
    for idx, row in enumerate(ordered):
        xtics.append(f'"{int(row["points"])}" {idx}')
        data_lines.append(
            " ".join(
                [
                    str(idx),
                    str(int(row["points"])),
                    f"{float(row['rss_after_insert_kb']) / 1024.0:.6f}",
                    f"{float(row['rss_single_query_peak_kb']) / 1024.0:.6f}",
                    f"{float(row['process_max_rss_kb']) / 1024.0:.6f}",
                    f"{float(row['avg_latency_us']) / 1000.0:.6f}",
                    f"{float(row['p95_latency_us']) / 1000.0:.6f}",
                    f"{float(row['p99_latency_us']) / 1000.0:.6f}",
                    f"{float(row['qps']):.6f}",
                ]
            )
        )
    data_block = "\n".join(data_lines)

    script = f"""
set terminal pngcairo size 3120,1152 enhanced font 'Arial,10'
set output {gnuplot_quote(png_path)}
set datafile separator whitespace
set style line 1 lc rgb '#1d4ed8' lt 1 lw 2 pt 7 ps 1.1
set style line 2 lc rgb '#16a34a' lt 1 lw 2 pt 5 ps 1.1
set style line 3 lc rgb '#7c3aed' lt 1 lw 2 pt 9 ps 1.1
set style line 4 lc rgb '#dc2626' lt 1 lw 2 pt 9 ps 1.1
set border lw 1
set key boxed opaque
set tics out
set xtics ({", ".join(xtics)}) rotate by -35
set grid ytics lc rgb '#e5e7eb' lw 1
unset grid xtics
set xlabel 'Indexed points'
set multiplot layout 1,3 title 'Flat small search: RSS, latency and QPS'
$DATA << EOD
{data_block}
EOD
unset logscale y
set ylabel 'RSS (MiB)'
set title 'RSS'
plot $DATA using 1:4 with linespoints linestyle 1 title 'single-query peak RSS'
set ylabel 'Latency (ms)'
set title 'Latency'
plot $DATA using 1:6 with linespoints linestyle 1 title 'avg latency'
set ylabel 'QPS'
set logscale y 10
set title 'QPS'
plot $DATA using 1:9 with linespoints linestyle 3 title 'QPS'
unset multiplot
"""
    subprocess.run([gnuplot], input=script, text=True, check=True)
    log(f"wrote {png_path}")
    return png_path


def main() -> int:
    args = parse_args()
    repo = repo_root()
    args.base_bin = (repo / args.base_bin).resolve() if not args.base_bin.is_absolute() else args.base_bin
    args.out_dir = (repo / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir

    if args.threshold < 0 or args.k <= 0 or args.threads <= 0 or args.qps_repeats <= 0 or args.query_buffer_count <= 0:
        raise ValueError("threshold must be non-negative; k, threads, qps-repeats and query-buffer-count must be positive")
    if any(size > args.threshold for size in args.sizes):
        raise ValueError("all sizes must be <= threshold for this flat-only experiment")
    if args.plot_only:
        write_plot(args.out_dir, load_results(args.out_dir / "results.jsonl"))
        return 0
    if not args.base_bin.exists():
        raise FileNotFoundError(args.base_bin)

    if not args.skip_build:
        build_benchmark(repo)

    run_id = str(time.time_ns())
    rows = [run_one(repo, args, points, run_id) for points in args.sizes]
    write_results(args.out_dir, rows)
    log(f"wrote {args.out_dir / 'results.jsonl'}")
    log(f"wrote {args.out_dir / 'table.csv'}")
    if not args.skip_plot:
        write_plot(args.out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
