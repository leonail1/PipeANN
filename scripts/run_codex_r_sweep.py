#!/usr/bin/env python3
"""Sweep graph build R/L for direct SIFT1M graph-search performance."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import run_codex_dynamic_update_suite as suite


HIGH_BUCKETS = [
    ("u25", 0.25),
    ("u30", 0.30),
    ("u50", 0.50),
    ("u75", 0.75),
    ("u100", 1.00),
]

SUFFIXES = [
    "_disk.index",
    "_disk.index.tags",
    "_pq_compressed.bin",
    "_pq_pivots.bin",
    "_labels.densebit",
    "_hybrid.meta",
]


def parse_config(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) == 2:
        r_value, l_value = int(parts[0]), int(parts[1])
        return f"R{r_value}_L{l_value}", r_value, l_value
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    raise argparse.ArgumentTypeError("config must be R:L or name:R:L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/exp_r_sweep"))
    parser.add_argument("--base-bin", type=Path, default=Path("data/sift1m/sift_base.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("data/sift1m/sift_query.bin"))
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--total-n", type=int, default=1_000_000)
    parser.add_argument("--pq-bytes", type=int, default=32)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--latency-cutoff-ms", type=float, default=100.0)
    parser.add_argument("--skip-build-tools", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        default=[],
        help="Sweep config as R:L or name:R:L. Defaults: 75:150, 96:150, 96:180.",
    )
    return parser.parse_args()


def plot(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = suite.load_jsonl(out_dir / "results.jsonl")
    bloat_rows = suite.load_jsonl(out_dir / "bloat.jsonl")
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        buckets = [bucket for bucket, _ in HIGH_BUCKETS]
        labels = {"u25": "25%", "u30": "30%", "u50": "50%", "u75": "75%", "u100": "100%"}
        configs = []
        for row in ok_rows:
            if row["config"] not in configs:
                configs.append(row["config"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for config in configs:
            by_bucket = {row["bucket"]: row for row in ok_rows if row["config"] == config}
            ordered = [by_bucket[bucket] for bucket in buckets if bucket in by_bucket]
            x = [buckets.index(row["bucket"]) for row in ordered]
            axes[0].plot(x, [float(row["avg_latency_us"]) / 1000.0 for row in ordered],
                         marker="o", linewidth=2.0, label=config)
            axes[1].plot(x, [float(row["qps"]) for row in ordered],
                         marker="o", linewidth=2.0, label=config)
            for xi, row in zip(x, ordered):
                axes[0].annotate(f"L={int(float(row['chosen_L']))}", (xi, float(row["avg_latency_us"]) / 1000.0),
                                 textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        for ax in axes:
            ax.set_xticks(range(len(buckets)), [labels[bucket] for bucket in buckets])
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
        axes[0].axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.0, label="10 ms")
        axes[0].set_xlabel("Selectivity")
        axes[0].set_ylabel("Avg latency (ms)")
        axes[0].set_title("Forced graph latency, recall@10 >= 98")
        axes[1].set_xlabel("Selectivity")
        axes[1].set_ylabel("QPS")
        axes[1].set_title("Forced graph QPS")
        fig.savefig(out_dir / "r_sweep_graph_perf.png", dpi=240)
        plt.close(fig)

    if bloat_rows:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        labels = [row["config"] for row in bloat_rows]
        x = range(len(labels))
        ax.bar(x, [float(row["total_to_raw_ratio"]) for row in bloat_rows], label="total/raw")
        ax.plot(x, [float(row["extra_over_raw_ratio"]) for row in bloat_rows],
                marker="o", color="#dc2626", linewidth=2.0, label="extra/raw")
        ax.axhline(2.0, color="#111827", linestyle="--", linewidth=1.0, label="2x total/raw")
        ax.set_xticks(list(x), labels)
        ax.set_ylabel("Ratio")
        ax.set_title("Index bloat by R/L config")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / "r_sweep_bloat.png", dpi=240)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out_dir = args.out_dir.resolve()
    args.base_bin = args.base_bin.resolve()
    args.query_bin = args.query_bin.resolve()
    configs = args.config or [
        ("R75_L150", 75, 150),
        ("R96_L150", 96, 150),
        ("R96_L180", 96, 180),
    ]

    if args.rerun:
        suite.clear_experiment_dir(args.out_dir, keep_names={"start.sh", "README.md"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build_tools:
        suite.build_tools(repo)

    total_points, dim = suite.read_bin_header(args.base_bin)
    query_total, query_dim = suite.read_bin_header(args.query_bin)
    if total_points < args.total_n:
        raise ValueError(f"base dataset has {total_points}, need {args.total_n}")
    if query_total < args.query_count:
        raise ValueError(f"query dataset has {query_total}, need {args.query_count}")
    if dim != query_dim:
        raise ValueError("base/query dimensionality mismatch")

    data_dir = args.out_dir / "data"
    label_dir = args.out_dir / "labels"
    query_bin = data_dir / f"sift_query_{args.query_count}.bin"
    suite.copy_prefix_bin(args.query_bin, query_bin, args.query_count, dim)
    base_labels = suite.write_selectivity_label_file(label_dir / f"base_{suite.stage_name(args.total_n)}.spmat",
                                                     args.total_n, args.total_n)
    query_labels = suite.ensure_query_label_files(label_dir, args.query_count)

    run_args = SimpleNamespace(
        build_r=64,
        build_l=96,
        pq_bytes=args.pq_bytes,
        memory_gb=args.memory_gb,
        beamwidth=args.beamwidth,
        k=args.k,
        metric=args.metric,
        nbr_type=args.nbr_type,
        gt_numa_node=1,
        gt_threads=0,
    )

    results_path = args.out_dir / "results.jsonl"
    bloat_path = args.out_dir / "bloat.jsonl"
    calibration_path = args.out_dir / "calibration.jsonl"
    if args.rerun:
        for path in [results_path, bloat_path, calibration_path, args.out_dir / "table.csv",
                     args.out_dir / "bloat.csv"]:
            if path.exists():
                path.unlink()

    existing = suite.load_jsonl(results_path)
    completed = {(row.get("config"), row.get("bucket")) for row in existing}
    bloat_existing = suite.load_jsonl(bloat_path)
    bloat_completed = {row.get("config") for row in bloat_existing}

    rows = existing
    bloat_rows = bloat_existing
    for name, r_value, l_value in configs:
        run_args.build_r = r_value
        run_args.build_l = l_value
        prefix = args.out_dir / "tmp" / name / "direct_1m"
        if not suite.prefix_exists(prefix):
            suite.build_index_with_pq_bytes(repo, run_args, args.base_bin, prefix, base_labels,
                                            args.threads, args.pq_bytes)

        if name not in bloat_completed:
            sizes = {suffix: Path(str(prefix) + suffix).stat().st_size
                     if Path(str(prefix) + suffix).exists() else 0 for suffix in SUFFIXES}
            total = sum(sizes.values())
            raw = args.total_n * dim * 4
            bloat = {
                "status": "ok",
                "config": name,
                "R": r_value,
                "build_L": l_value,
                "points": args.total_n,
                "raw_vector_bytes": raw,
                "total_index_bytes": total,
                "extra_over_raw_ratio": (total - raw) / raw,
                "total_to_raw_ratio": total / raw,
                **sizes,
            }
            bloat_rows.append(bloat)
            bloat_completed.add(name)
            suite.append_jsonl(bloat_path, bloat)

        for bucket, _selectivity in HIGH_BUCKETS:
            if (name, bucket) in completed:
                continue
            truth = args.out_dir / "truth" / f"gt_{suite.stage_name(args.total_n)}_{bucket}.bin"
            suite.compute_truth(repo, run_args, args.base_bin, query_bin, truth, base_labels, query_labels[bucket])
            chosen = None
            last_row = {}
            for search_l in suite.EXP4_L_CANDIDATES:
                row = suite.static_hybrid_search(repo, run_args, prefix=prefix, jsonl=calibration_path,
                                                 query_bin=query_bin, truthset=truth,
                                                 query_label_file=query_labels[bucket], route="graph",
                                                 threads=1, search_l=search_l)
                last_row = row
                if float(row.get("recall@10", 0.0)) >= 98.0:
                    chosen = row
                    break
                if float(row.get("avg_latency_us", 0.0)) > args.latency_cutoff_ms * 1000.0:
                    break
            measured = chosen or last_row
            measured.update({
                "status": "ok" if chosen is not None else "skipped_recall_or_latency",
                "config": name,
                "R": r_value,
                "build_L": l_value,
                "bucket": bucket,
                "route": "graph",
                "target_recall@10": 98.0,
                "latency_cutoff_ms": args.latency_cutoff_ms,
                "baseline_pq_bytes": args.pq_bytes,
            })
            rows.append(measured)
            completed.add((name, bucket))
            suite.append_jsonl(results_path, measured)

        suite.clean_prefix(prefix)

    suite.write_csv(args.out_dir / "table.csv", rows)
    suite.write_csv(args.out_dir / "bloat.csv", bloat_rows)
    plot(args.out_dir)
    suite.clean_large_files(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
