#!/usr/bin/env python3
"""Run representative PipeANN QPS/SSD bottleneck experiments and fio baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
QPS_ROOT = REPO_ROOT / "experiments" / "qps_4sets"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "io_representative_20260514"
RUN_QPS_SCRIPT = REPO_ROOT / "scripts" / "run_qps_4sets_reproduction.py"

DATASET_ORDER = ["fashion_mnist784", "gist960", "glove100", "yfcc10m"]
DATASET_LABELS = {
    "fashion_mnist784": "Fashion-MNIST",
    "gist960": "GIST960",
    "glove100": "GloVe100",
    "yfcc10m": "YFCC10M",
}
REPRESENTATIVE_BUCKETS = {
    "fashion_mnist784": ["u0p1", "u0p5", "u1", "u5", "u10", "u25", "u50", "u100"],
    "gist960": ["u0p1", "u0p5", "u1", "u5", "u10", "u25", "u50", "u100"],
    "glove100": ["u0p1", "u0p5", "u1", "u5", "u10", "u25", "u50", "u100"],
    "yfcc10m": [
        "real_t4e-03_l600",
        "real_t5e-03_l902",
        "real_t1e-02_l0",
        "real_t5e-02_l24",
        "real_t1e-01_l17",
        "real_t2e-01_l29",
        "real_extra_l23",
    ],
}
FIO_SPECS = {
    "fashion_mnist784": {
        "filename": REPO_ROOT / "data" / "fashion_mnist784" / "fashion_mnist784_qps4_pipeann_disk.index",
        "bs": "4k",
    },
    "gist960": {
        "filename": REPO_ROOT / "data" / "gist960" / "gist960_qps4_pipeann_disk.index",
        "bs": "8k",
    },
    "glove100": {
        "filename": REPO_ROOT / "data" / "glove100" / "glove100_qps4_pipeann_disk.index",
        "bs": "4k",
    },
    "yfcc10m": {
        "filename": REPO_ROOT / "data" / "yfcc100M" / "yfcc10m_pipeann_disk.index",
        "bs": "8k",
    },
}
DEFAULT_IODEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_NUMJOBS = [1, 4, 8]
COMPARISON_FIELDS = [
    "dataset",
    "bucket_name",
    "selectivity",
    "thread",
    "route_decision",
    "qps",
    "query_p99_latency_us",
    "query_p999_latency_us",
    "cpu_pct",
    "mean_ios",
    "real_read_iops",
    "real_read_mb_s",
    "real_qd",
    "real_read_await_ms",
    "real_ssd_lat_status",
    "real_ssd_lat_p50_ms",
    "real_ssd_lat_p95_ms",
    "real_ssd_lat_p99_ms",
    "fio_matched_iodepth",
    "fio_iops",
    "fio_bw_mb_s",
    "fio_lat_mean_ms",
    "fio_lat_p99_ms",
    "fio_capacity_ratio",
    "disk_metrics_status",
    "bottleneck_conclusion",
]


def log(message: str) -> None:
    print(message, flush=True)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True)
        writer.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_parent(path)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fields, extrasaction="ignore")
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_command(command: list[str], cwd: Path = REPO_ROOT, timeout: int | None = None) -> None:
    log("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)


def build_yfcc_manifest(out_dir: Path) -> Path:
    base_manifest = read_json(QPS_ROOT / "yfcc10m" / "manifests" / "yfcc10m_real_target_manifest.json")
    workload_summary = read_json(QPS_ROOT / "yfcc10m" / "workloads" / "random_single_label" / "random_single_label_workloads_summary.json")
    by_name = {bucket["name"]: bucket for bucket in base_manifest.get("buckets", [])}
    for workload in workload_summary.get("real_workloads", []):
        if workload.get("bucket_name") != "real_extra_l23":
            continue
        by_name["real_extra_l23"] = {
            "name": workload["bucket_name"],
            "label": workload["bucket_label"],
            "lower": workload["selectivity"],
            "midpoint": workload["selectivity"],
            "upper": workload["selectivity"],
            "query_bin": workload["query_bin"],
            "query_labels": workload["query_labels"],
            "probe_query_bin": workload["probe_query_bin"],
            "probe_query_labels": workload["probe_query_labels"],
            "query_count": workload["query_count"],
        }
        break
    missing = [bucket for bucket in REPRESENTATIVE_BUCKETS["yfcc10m"] if bucket not in by_name]
    if missing:
        raise RuntimeError(f"missing YFCC representative buckets: {missing}")
    manifest = {
        **base_manifest,
        "buckets": [by_name[name] for name in REPRESENTATIVE_BUCKETS["yfcc10m"]],
    }
    path = out_dir / "manifests" / "yfcc10m_representative_manifest.json"
    write_json(path, manifest)
    return path


def run_representative(args: argparse.Namespace) -> None:
    out_dir = resolve_path(args.out_dir)
    threads = ",".join(str(thread) for thread in args.threads)
    yfcc_manifest = build_yfcc_manifest(out_dir)
    datasets = args.dataset or DATASET_ORDER
    for dataset in datasets:
        buckets = REPRESENTATIVE_BUCKETS[dataset]
        if args.smoke:
            buckets = ["u25"] if dataset == "gist960" else []
        if not buckets:
            continue
        command = [
            sys.executable,
            str(RUN_QPS_SCRIPT),
            "run-thread-sweep",
            "--dataset",
            dataset,
            "--experiment-root",
            str(QPS_ROOT),
            "--out-dir",
            str(out_dir / dataset / "thread_sweep"),
            "--threads",
            "1" if args.smoke else threads,
            "--sample-interval-s",
            str(args.sample_interval_s),
            "--capture-iostat",
            "--capture-block-latency",
            "--timeout",
            str(args.timeout),
        ]
        if dataset == "yfcc10m":
            command.extend(["--manifest", str(yfcc_manifest)])
        for bucket in buckets:
            command.extend(["--bucket", bucket])
        run_command(command, timeout=args.timeout * max(1, len(buckets)) * max(1, len(args.threads)))


def fio_json_path(out_dir: Path, dataset: str, numjobs: int, iodepth: int) -> Path:
    return out_dir / "fio" / dataset / f"fio_{dataset}_jobs{numjobs}_qd{iodepth}.json"


def percentile_ms(payload: dict[str, Any], key: str) -> float | None:
    percentiles = payload.get("percentile") or {}
    value = percentiles.get(key) or percentiles.get(f"{float(key):.6f}")
    if value is None:
        return None
    return float(value) / 1_000_000.0


def achieved_qd(job: dict[str, Any], requested_iodepth: int) -> float:
    levels = job.get("iodepth_level") or {}
    total = 0.0
    weight = 0.0
    for key, pct in levels.items():
        try:
            depth = float(key.replace(">=", ""))
        except ValueError:
            depth = float(requested_iodepth)
        total += depth * float(pct)
        weight += float(pct)
    return requested_iodepth if weight <= 0 else total / weight


def parse_fio_result(path: Path, dataset: str, bs: str, numjobs: int, iodepth: int) -> dict[str, Any]:
    payload = read_json(path)
    job = payload["jobs"][0]
    read = job.get("read", {})
    clat = read.get("clat_ns", {})
    bw_bytes = float(read.get("bw_bytes", 0.0))
    read_iops = float(read.get("iops", 0.0))
    read_mb_s = bw_bytes / (1024.0 * 1024.0)
    read_lat_mean_ms = None if clat.get("mean") is None else float(clat["mean"]) / 1_000_000.0
    return {
        "dataset": dataset,
        "filename": str(FIO_SPECS[dataset]["filename"]),
        "bs": bs,
        "numjobs": numjobs,
        "iodepth": iodepth,
        "fio_json": str(path),
        "error": job.get("error"),
        "total_ios": read.get("total_ios"),
        "read_iops": read_iops,
        "read_mb_s": read_mb_s,
        "read_lat_mean_ms": read_lat_mean_ms,
        "lat_p50_ms": percentile_ms(clat, "50.000000"),
        "lat_p95_ms": percentile_ms(clat, "95.000000"),
        "lat_p99_ms": percentile_ms(clat, "99.000000"),
        "achieved_qd": achieved_qd(job, iodepth),
        # Backward-compatible aliases for older analysis notebooks.
        "iops": read_iops,
        "bw_mb_s": read_mb_s,
        "lat_mean_ms": read_lat_mean_ms,
    }


def run_fio(args: argparse.Namespace) -> None:
    if shutil.which("fio") is None:
        raise RuntimeError("fio is required; install fio before running baseline")
    out_dir = resolve_path(args.out_dir)
    datasets = args.dataset or DATASET_ORDER
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = FIO_SPECS[dataset]
        filename = Path(spec["filename"])
        if not filename.exists():
            raise RuntimeError(f"missing fio target file: {filename}")
        iodepths = [1] if args.smoke else args.iodepths
        numjobs_values = [1] if args.smoke else args.numjobs
        for numjobs in numjobs_values:
            for iodepth in iodepths:
                output = fio_json_path(out_dir, dataset, numjobs, iodepth)
                ensure_parent(output)
                command = [
                    "fio",
                    f"--name={dataset}_randread",
                    f"--filename={filename}",
                    "--readonly",
                    "--allow_file_create=0",
                    "--rw=randread",
                    f"--bs={spec['bs']}",
                    "--direct=1",
                    "--ioengine=io_uring",
                    f"--iodepth={iodepth}",
                    f"--numjobs={numjobs}",
                    "--time_based",
                    f"--runtime={args.runtime_s}",
                    f"--ramp_time={args.ramp_time_s}",
                    "--group_reporting=1",
                    "--output-format=json",
                    f"--output={output}",
                ]
                run_command(command, timeout=args.runtime_s + args.ramp_time_s + 120)
                rows.append(parse_fio_result(output, dataset, str(spec["bs"]), numjobs, iodepth))
    write_jsonl(out_dir / "fio_baseline.jsonl", rows)
    write_csv(out_dir / "fio_baseline.csv", rows)


def collect_representative_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        rows.extend(load_jsonl(out_dir / dataset / "thread_sweep" / "aggregated_results.jsonl"))
    return rows


def route_decision(row: dict[str, Any]) -> str:
    if int(row.get("prefilter_count") or 0) > 0:
        return "prefilter"
    if int(row.get("graph_count") or 0) > 0:
        return "graph"
    return "fallback"


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def scaling_efficiencies(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float | None]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset")), str(row.get("bucket_name"))), {})[int(row.get("threads"))] = row
    out: dict[tuple[str, str], float | None] = {}
    for key, by_thread in grouped.items():
        if 1 not in by_thread or 8 not in by_thread:
            out[key] = None
            continue
        qps1 = safe_float(by_thread[1].get("qps"))
        qps8 = safe_float(by_thread[8].get("qps"))
        out[key] = None if not qps1 or qps1 <= 0 or qps8 is None else qps8 / qps1 / 8.0
    return out


def match_fio(row: dict[str, Any], fio_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    dataset = str(row.get("dataset"))
    threads = int(row.get("threads"))
    real_qd = safe_float(row.get("avg_qd"))
    candidates = [item for item in fio_rows if item["dataset"] == dataset and int(item["numjobs"]) == threads]
    if not candidates:
        return None
    target = 1.0 if real_qd is None else real_qd
    return sorted(candidates, key=lambda item: (abs(float(item["iodepth"]) - target), float(item["iodepth"])))[0]


def conclude(row: dict[str, Any], fio: dict[str, Any] | None, scaling_efficiency: float | None) -> tuple[str, float | None]:
    if row.get("disk_metrics_status") not in (None, "", "ok"):
        return "inconclusive_monitor_invalid", None
    if fio is None:
        return "inconclusive_missing_fio", None
    real_iops = safe_float(row.get("avg_read_iops")) or 0.0
    real_bw = safe_float(row.get("avg_read_mb_s")) or 0.0
    fio_iops = safe_float(fio.get("read_iops")) or 0.0
    fio_bw = safe_float(fio.get("read_mb_s")) or 0.0
    iops_ratio = 0.0 if fio_iops <= 0 else real_iops / fio_iops
    bw_ratio = 0.0 if fio_bw <= 0 else real_bw / fio_bw
    capacity_ratio = max(iops_ratio, bw_ratio)
    real_latency = safe_float(row.get("block_latency_p99_ms"))
    fio_latency = safe_float(fio.get("lat_p99_ms"))
    if real_latency is None or fio_latency is None:
        real_latency = safe_float(row.get("avg_read_await_ms"))
        fio_latency = safe_float(fio.get("read_lat_mean_ms"))
    cpu = safe_float(row.get("avg_cpu_pct")) or 0.0
    real_qd = safe_float(row.get("avg_qd"))
    latency_lifted = real_latency is not None and fio_latency is not None and real_latency >= fio_latency * 1.2
    if capacity_ratio >= 0.8 and latency_lifted:
        return "ssd_capacity_bound", capacity_ratio
    if real_qd is not None and real_qd < 2.0 and capacity_ratio < 0.5 and real_latency is not None:
        return "io_latency_or_queue_underfilled", capacity_ratio
    if capacity_ratio < 0.5 and (cpu >= 80.0 or (scaling_efficiency is not None and scaling_efficiency < 0.8)):
        return "cpu_or_algorithm_bound", capacity_ratio
    return "inconclusive_mixed", capacity_ratio


def summarize(args: argparse.Namespace) -> None:
    out_dir = resolve_path(args.out_dir)
    representative_rows = collect_representative_rows(out_dir)
    fio_rows = load_jsonl(out_dir / "fio_baseline.jsonl")
    write_jsonl(out_dir / "representative_results.jsonl", representative_rows)
    write_csv(out_dir / "representative_results.csv", representative_rows)

    efficiencies = scaling_efficiencies(representative_rows)
    comparison_rows: list[dict[str, Any]] = []
    for row in representative_rows:
        fio = match_fio(row, fio_rows)
        efficiency = efficiencies.get((str(row.get("dataset")), str(row.get("bucket_name"))))
        conclusion, ratio = conclude(row, fio, efficiency)
        comparison_rows.append(
            {
                "dataset": row.get("dataset"),
                "bucket_name": row.get("bucket_name"),
                "selectivity": row.get("selectivity_midpoint"),
                "thread": row.get("threads"),
                "route_decision": route_decision(row),
                "qps": row.get("qps"),
                "query_p99_latency_us": row.get("p99_latency_us"),
                "query_p999_latency_us": row.get("p999_latency_us"),
                "cpu_pct": row.get("avg_cpu_pct"),
                "mean_ios": row.get("mean_ios"),
                "real_read_iops": row.get("avg_read_iops"),
                "real_read_mb_s": row.get("avg_read_mb_s"),
                "real_qd": row.get("avg_qd"),
                "real_read_await_ms": row.get("avg_read_await_ms"),
                "real_ssd_lat_status": row.get("block_latency_status"),
                "real_ssd_lat_p50_ms": row.get("block_latency_p50_ms"),
                "real_ssd_lat_p95_ms": row.get("block_latency_p95_ms"),
                "real_ssd_lat_p99_ms": row.get("block_latency_p99_ms"),
                "fio_matched_iodepth": None if fio is None else fio.get("iodepth"),
                "fio_iops": None if fio is None else fio.get("read_iops"),
                "fio_bw_mb_s": None if fio is None else fio.get("read_mb_s"),
                "fio_lat_mean_ms": None if fio is None else fio.get("read_lat_mean_ms"),
                "fio_lat_p99_ms": None if fio is None else fio.get("lat_p99_ms"),
                "fio_capacity_ratio": ratio,
                "disk_metrics_status": row.get("disk_metrics_status"),
                "bottleneck_conclusion": conclusion,
            }
        )
    write_csv(out_dir / "comparison_table.csv", comparison_rows, COMPARISON_FIELDS)
    write_jsonl(out_dir / "comparison_table.jsonl", comparison_rows)
    plot_all(out_dir, comparison_rows)


def plot_all(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("[warn] matplotlib unavailable; skip plots")
        return

    thread_order = [1, 4, 8]
    thread_colors = {1: "#315f8d", 4: "#26734d", 8: "#b4552a"}
    thread_markers = {1: "o", 4: "^", 8: "D"}

    def as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    def dataset_rows(dataset: str, thread: int | None = None) -> list[dict[str, Any]]:
        selected = [row for row in rows if row["dataset"] == dataset]
        if thread is not None:
            selected = [row for row in selected if int(row["thread"]) == thread]
        return sorted(selected, key=lambda row: (float(row["selectivity"]), int(row["thread"])))

    def xy_for(dataset: str, thread: int, field: str, scale: float = 1.0, skip_missing: bool = True) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        for row in dataset_rows(dataset, thread):
            value = as_float(row.get(field))
            if value is None:
                if skip_missing:
                    continue
                value = 0.0
            x_values.append(float(row["selectivity"]) * 100.0)
            y_values.append(value * scale)
        return x_values, y_values

    def common_axes(title: str, ylabel: str):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
        fig.suptitle(title, fontsize=14)
        for axis, dataset in zip(axes.flat, DATASET_ORDER):
            axis.set_title(DATASET_LABELS[dataset])
            axis.set_xscale("log")
            axis.set_xlabel("Selectivity (%)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.22)
        return fig, axes

    fig, axes = common_axes("Representative search: QPS and query p99 by thread", "QPS")
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        twin = axis.twinx()
        for thread in thread_order:
            x, qps = xy_for(dataset, thread, "qps")
            if not x:
                continue
            color = thread_colors[thread]
            axis.plot(x, qps, marker=thread_markers[thread], color=color, linewidth=1.7, label=f"QPS t={thread}")
            x_lat, p99_ms = xy_for(dataset, thread, "query_p99_latency_us", scale=0.001)
            twin.plot(x_lat, p99_ms, marker=thread_markers[thread], color=color, linestyle="--", linewidth=1.1, alpha=0.72, label=f"p99 t={thread}")
        twin.set_ylabel("Query p99 (ms)")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, frameon=False, ncol=2)
    fig.savefig(out_dir / "representative_qps_latency.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, axes = common_axes("Representative search: CPU by thread", "CPU pct")
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        for thread in thread_order:
            x, cpu = xy_for(dataset, thread, "cpu_pct")
            if x:
                axis.plot(x, cpu, marker=thread_markers[thread], color=thread_colors[thread], linewidth=1.7, label=f"t={thread}")
        axis.legend(frameon=False, ncol=3)
    fig.savefig(out_dir / "representative_cpu.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    disk_metrics = [
        ("real_read_iops", "Read IOPS", "symlog"),
        ("real_read_mb_s", "Read MB/s", "symlog"),
        ("real_qd", "Avg queue depth", "symlog"),
        ("real_read_await_ms", "Read await (ms)", "linear"),
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), len(disk_metrics), figsize=(17.0, 11.5), constrained_layout=True)
    fig.suptitle("Representative search: real disk metrics by thread", fontsize=14)
    for row_index, dataset in enumerate(DATASET_ORDER):
        for col_index, (field, label, scale_name) in enumerate(disk_metrics):
            axis = axes[row_index][col_index]
            for thread in thread_order:
                x, y = xy_for(dataset, thread, field)
                if y:
                    axis.plot(x, y, marker=thread_markers[thread], color=thread_colors[thread], linewidth=1.35, label=f"t={thread}")
            if scale_name == "symlog":
                axis.set_yscale("symlog", linthresh=0.01)
            axis.set_xscale("log")
            axis.set_xlabel("Selectivity (%)")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.22)
            if col_index == 0:
                axis.set_title(DATASET_LABELS[dataset], loc="left")
            else:
                axis.set_title(label)
            if row_index == 0 and col_index == len(disk_metrics) - 1:
                axis.legend(frameon=False, ncol=1)
    fig.savefig(out_dir / "representative_disk.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(len(DATASET_ORDER), 2, figsize=(14.5, 11.5), constrained_layout=True)
    fig.suptitle("Representative search vs fio same-QD capacity: solid=search, dashed=fio, zero-read search points omitted", fontsize=13)
    for row_index, dataset in enumerate(DATASET_ORDER):
        for col_index, (real_field, fio_field, label) in enumerate(
            [("real_read_iops", "fio_iops", "Read IOPS"), ("real_read_mb_s", "fio_bw_mb_s", "Read MB/s")]
        ):
            axis = axes[row_index][col_index]
            for thread in thread_order:
                x_real, y_real = xy_for(dataset, thread, real_field)
                x_fio, y_fio = xy_for(dataset, thread, fio_field)
                real_points = [(x, y) for x, y in zip(x_real, y_real) if y > 0.0]
                fio_points = [(x, y) for x, y in zip(x_fio, y_fio) if y > 0.0]
                if y_real:
                    if real_points:
                        axis.plot([x for x, _ in real_points], [y for _, y in real_points], marker=thread_markers[thread], color=thread_colors[thread], linewidth=1.45, label=f"search t={thread}")
                    elif col_index == 0:
                        axis.text(0.03, 0.1 + 0.06 * thread_order.index(thread), f"search t={thread}: 0 reads", transform=axis.transAxes, fontsize=8, color=thread_colors[thread])
                if fio_points:
                    axis.plot([x for x, _ in fio_points], [y for _, y in fio_points], marker=thread_markers[thread], color=thread_colors[thread], linestyle="--", linewidth=1.2, alpha=0.72, label=f"fio t={thread}")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Selectivity (%)")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.22)
            title = DATASET_LABELS[dataset] if col_index == 0 else label
            axis.set_title(title, loc="left" if col_index == 0 else "center")
            if row_index == 0 and col_index == 1:
                axis.legend(frameon=False, ncol=2)
    fig.savefig(out_dir / "representative_vs_fio.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-representative")
    run_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    run_parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)
    run_parser.add_argument("--threads", type=parse_csv_ints, default=[1, 4, 8])
    run_parser.add_argument("--sample-interval-s", type=float, default=0.5)
    run_parser.add_argument("--timeout", type=int, default=7200)
    run_parser.add_argument("--smoke", action="store_true")

    fio_parser = subparsers.add_parser("run-fio")
    fio_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    fio_parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)
    fio_parser.add_argument("--iodepths", type=parse_csv_ints, default=DEFAULT_IODEPTHS)
    fio_parser.add_argument("--numjobs", type=parse_csv_ints, default=DEFAULT_NUMJOBS)
    fio_parser.add_argument("--runtime-s", type=int, default=30)
    fio_parser.add_argument("--ramp-time-s", type=int, default=3)
    fio_parser.add_argument("--smoke", action="store_true")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    all_parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)
    all_parser.add_argument("--threads", type=parse_csv_ints, default=[1, 4, 8])
    all_parser.add_argument("--iodepths", type=parse_csv_ints, default=DEFAULT_IODEPTHS)
    all_parser.add_argument("--numjobs", type=parse_csv_ints, default=DEFAULT_NUMJOBS)
    all_parser.add_argument("--runtime-s", type=int, default=30)
    all_parser.add_argument("--ramp-time-s", type=int, default=3)
    all_parser.add_argument("--sample-interval-s", type=float, default=0.5)
    all_parser.add_argument("--timeout", type=int, default=7200)
    all_parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run-representative":
        run_representative(args)
        return 0
    if args.command == "run-fio":
        run_fio(args)
        return 0
    if args.command == "summarize":
        summarize(args)
        return 0
    if args.command == "all":
        run_representative(args)
        run_fio(args)
        summarize(args)
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
