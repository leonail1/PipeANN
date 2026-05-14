#!/usr/bin/env python3
"""Orchestrate qps_4sets reproduction on the current mainline tree."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "qps_4sets"
HYBRID_SCRIPT = REPO_ROOT / "scripts" / "pipeann_hybrid_experiment.py"
BUILD_DISK_INDEX = REPO_ROOT / "build" / "tests" / "build_disk_index"
BIN_HEADER = struct.Struct("<ii")
GT_HEADER = struct.Struct("<ii")
DEFAULT_THREAD_SWEEP = (1, 2, 4, 8)
DEFAULT_SMALL_SELECTIVITIES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
DEFAULT_YFCC_SELECTIVITIES = (0.004, 0.005, 0.01, 0.02, 0.05, 0.099, 0.192)
DEFAULT_SAMPLE_INTERVAL_S = 0.5
SECTOR_SIZE_BYTES = 512


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    similarity: str
    normalize_for_l2: bool
    index_type: str
    source_url: str | None = None


SMALL_DATASETS: dict[str, DatasetSpec] = {
    "fashion_mnist784": DatasetSpec(
        name="fashion_mnist784",
        kind="ann_benchmarks_hdf5",
        similarity="l2",
        normalize_for_l2=False,
        index_type="float",
        source_url="https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    ),
    "gist960": DatasetSpec(
        name="gist960",
        kind="ann_benchmarks_hdf5",
        similarity="l2",
        normalize_for_l2=False,
        index_type="float",
        source_url="https://ann-benchmarks.com/gist-960-euclidean.hdf5",
    ),
    "glove100": DatasetSpec(
        name="glove100",
        kind="ann_benchmarks_hdf5",
        similarity="l2",
        normalize_for_l2=True,
        index_type="float",
        source_url="https://ann-benchmarks.com/glove-100-angular.hdf5",
    ),
}

YFCC_DATASET = DatasetSpec(
    name="yfcc10m",
    kind="existing",
    similarity="l2",
    normalize_for_l2=False,
    index_type="uint8",
)

ALL_DATASETS: dict[str, DatasetSpec] = {**SMALL_DATASETS, "yfcc10m": YFCC_DATASET}


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
    rows: list[dict[str, Any]] = []
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


def parse_csv_ints(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("at least one integer is required")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_csv_floats(value: str) -> list[float]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("at least one float is required")
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def import_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for ann-benchmarks HDF5 conversion; install it into the workspace venv first"
        ) from exc
    return h5py


def dataset_data_dir(data_root: Path, dataset: str) -> Path:
    return data_root / dataset


def dataset_experiment_dir(experiment_root: Path, dataset: str) -> Path:
    return experiment_root / dataset


def default_small_paths(data_root: Path, dataset: str) -> dict[str, Path]:
    root = dataset_data_dir(data_root, dataset)
    return {
        "root": root,
        "raw_hdf5": root / f"{dataset}.hdf5",
        "base_bin": root / "base.bin",
        "query_bin": root / "query.bin",
        "groundtruth_ibin": root / "groundtruth.ibin",
        "meta_json": root / "conversion_meta.json",
        "index_prefix": root / f"{dataset}_qps4_pipeann",
    }


def default_yfcc_paths(data_root: Path, experiment_root: Path) -> dict[str, Path]:
    data_dir = data_root / "yfcc100M"
    exp_dir = dataset_experiment_dir(experiment_root, "yfcc10m")
    return {
        "root": data_dir,
        "experiment_root": exp_dir,
        "base_bin": data_dir / "base.10M.u8bin",
        "query_bin": data_dir / "query.public.100K.u8bin",
        "base_labels": data_dir / "base.metadata.10M.spmat",
        "query_labels": data_dir / "query.metadata.public.100K.spmat",
        "index_prefix": data_dir / "yfcc10m_pipeann",
        "scan_dir": exp_dir / "workloads" / "scan",
        "workload_dir": exp_dir / "workloads" / "random_single_label",
        "manifest_dir": exp_dir / "manifests",
        "filtered_manifest": exp_dir / "manifests" / "yfcc10m_real_target_manifest.json",
        "target_summary": exp_dir / "workloads" / "target_recommendations.json",
    }


def workload_dir(experiment_root: Path, dataset: str) -> Path:
    return dataset_experiment_dir(experiment_root, dataset) / "workloads" / "uniform_exact_selectivity"


def manifest_path(experiment_root: Path, dataset: str) -> Path:
    return dataset_experiment_dir(experiment_root, dataset) / "manifests" / "uniform_exact_selectivity_manifest.json"


def sweep_root(experiment_root: Path, dataset: str) -> Path:
    return dataset_experiment_dir(experiment_root, dataset) / "thread_sweep"


def bucket_name_for_selectivity(selectivity: float) -> str:
    percent = selectivity * 100.0
    if abs(percent - round(percent)) < 1e-12:
        return f"u{int(round(percent))}"
    text = f"{percent:.1f}".replace(".", "p")
    return f"u{text}"


def selectivity_specs(selectivities: list[float]) -> list[str]:
    return [f"{bucket_name_for_selectivity(value)}:{value}" for value in selectivities]


def percent_label(selectivity: float) -> str:
    percent = selectivity * 100.0
    if abs(percent - round(percent)) < 1e-12:
        return f"{int(round(percent))}%"
    return f"{percent:.1f}%"


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        log(f"reuse {destination}")
        return destination

    ensure_parent(destination)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "PipeANN-qps4sets/1.0"})
    log(f"download {url} -> {destination}")
    with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as writer:
        copied = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            copied += len(chunk)
            if copied and copied % (128 * 1024 * 1024) < len(chunk):
                log(f"  downloaded {copied / (1024 * 1024):.1f} MiB")
    os.replace(tmp_path, destination)
    return destination


def write_matrix_bin(path: Path, dataset: Any, normalize: bool, chunk_rows: int) -> tuple[int, int]:
    nrows, ndim = int(dataset.shape[0]), int(dataset.shape[1])
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as writer:
        writer.write(BIN_HEADER.pack(nrows, ndim))
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            block = np.asarray(dataset[start:stop], dtype=np.float32)
            if normalize:
                block = normalize_rows(block)
            block.astype("<f4", copy=False).tofile(writer)
    os.replace(tmp_path, path)
    return nrows, ndim


def write_groundtruth_ibin(path: Path, neighbors: np.ndarray, distances: np.ndarray) -> tuple[int, int]:
    if neighbors.shape != distances.shape:
        raise ValueError("groundtruth neighbors/distances shape mismatch")
    nrows, topk = int(neighbors.shape[0]), int(neighbors.shape[1])
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    neighbors_i32 = np.asarray(neighbors, dtype=np.int32)
    distances_f32 = np.asarray(distances, dtype=np.float32)
    with tmp_path.open("wb") as writer:
        writer.write(GT_HEADER.pack(nrows, topk))
        neighbors_i32.tofile(writer)
        distances_f32.tofile(writer)
        neighbors_i32.tofile(writer)
    os.replace(tmp_path, path)
    return nrows, topk


def fetch_small_dataset(spec: DatasetSpec, data_root: Path, force: bool, chunk_rows: int) -> dict[str, Any]:
    if spec.source_url is None:
        raise ValueError(f"dataset {spec.name} does not have a downloadable source")
    paths = default_small_paths(data_root, spec.name)
    raw_hdf5 = download_file(spec.source_url, paths["raw_hdf5"], force=force)
    h5py = import_h5py()
    with h5py.File(raw_hdf5, "r") as handle:
        train = handle["train"]
        test = handle["test"]
        neighbors = np.asarray(handle["neighbors"], dtype=np.int32)
        distances = np.asarray(handle["distances"], dtype=np.float32)
        base_count, dim = write_matrix_bin(paths["base_bin"], train, spec.normalize_for_l2, chunk_rows)
        query_count, query_dim = write_matrix_bin(paths["query_bin"], test, spec.normalize_for_l2, chunk_rows)
        if dim != query_dim:
            raise ValueError(f"dimension mismatch for {spec.name}: base={dim} query={query_dim}")
        gt_count, gt_topk = write_groundtruth_ibin(paths["groundtruth_ibin"], neighbors, distances)

    summary = {
        "dataset": spec.name,
        "source_url": spec.source_url,
        "raw_hdf5": str(raw_hdf5),
        "base_bin": str(paths["base_bin"]),
        "query_bin": str(paths["query_bin"]),
        "groundtruth_ibin": str(paths["groundtruth_ibin"]),
        "index_prefix": str(paths["index_prefix"]),
        "similarity": spec.similarity,
        "normalize_for_l2": spec.normalize_for_l2,
        "index_type": spec.index_type,
        "base_count": base_count,
        "query_count": query_count,
        "dim": dim,
        "groundtruth_rows": gt_count,
        "groundtruth_topk": gt_topk,
    }
    write_json(paths["meta_json"], summary)
    log(f"[ok] prepared {spec.name} under {paths['root']}")
    return summary


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_logged(command: list[str], cwd: Path, stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    log("+ " + shell_join(command))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if stdout_path is not None:
        ensure_parent(stdout_path)
    if stderr_path is not None:
        ensure_parent(stderr_path)
    stdout_file = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr_file = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env, stdout=stdout_file, stderr=stderr_file)
    finally:
        if stdout_file is not None:
            stdout_file.close()
        if stderr_file is not None:
            stderr_file.close()


def build_small_index(
    spec: DatasetSpec,
    data_root: Path,
    build_r: int,
    build_l: int,
    pq_bytes: int,
    memory_gb: int,
    threads: int,
    force: bool,
) -> dict[str, Any]:
    paths = default_small_paths(data_root, spec.name)
    marker = Path(str(paths["index_prefix"]) + "_disk.index")
    if marker.exists() and not force:
        log(f"reuse {marker}")
    else:
        if not BUILD_DISK_INDEX.exists():
            raise RuntimeError(f"missing build target: {BUILD_DISK_INDEX}")
        build_command = [
            str(BUILD_DISK_INDEX),
            spec.index_type,
            str(paths["base_bin"]),
            str(paths["index_prefix"]),
            str(build_r),
            str(build_l),
            str(pq_bytes),
            str(memory_gb),
            str(threads),
            spec.similarity,
            "pq",
        ]
        build_log = paths["root"] / "build_disk_index.log"
        run_logged(build_command, REPO_ROOT, stdout_path=build_log, stderr_path=build_log.with_suffix(".err.log"))
    summary = {
        "dataset": spec.name,
        "index_prefix": str(paths["index_prefix"]),
        "index_type": spec.index_type,
        "similarity": spec.similarity,
        "build_R": build_r,
        "build_L": build_l,
        "pq_bytes": pq_bytes,
        "memory_gb": memory_gb,
        "threads": threads,
    }
    write_json(paths["root"] / "index_build_meta.json", summary)
    return summary


def run_hybrid(command: list[str], stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    run_logged([sys.executable, str(HYBRID_SCRIPT), *command], REPO_ROOT, stdout_path=stdout_path, stderr_path=stderr_path)


def prepare_small_workload(
    spec: DatasetSpec,
    data_root: Path,
    experiment_root: Path,
    selectivities: list[float],
    queries_per_bucket: int,
    probe_queries_per_bucket: int,
    seed: int,
) -> dict[str, Any]:
    data_paths = default_small_paths(data_root, spec.name)
    work_dir = workload_dir(experiment_root, spec.name)
    manifest = manifest_path(experiment_root, spec.name)
    ensure_parent(manifest)

    generate_log = dataset_experiment_dir(experiment_root, spec.name) / "logs" / "generate_uniform_exact.log"
    build_manifest_log = dataset_experiment_dir(experiment_root, spec.name) / "logs" / "build_manifest.log"
    generate_args = [
        "generate-uniform-exact-selectivity-workloads",
        "--base-bin",
        str(data_paths["base_bin"]),
        "--query-bin",
        str(data_paths["query_bin"]),
        "--index-type",
        spec.index_type,
        "--selector-type",
        "intersect",
        "--out-dir",
        str(work_dir),
        "--seed",
        str(seed),
        "--queries-per-bucket",
        str(queries_per_bucket),
        "--probe-queries-per-bucket",
        str(probe_queries_per_bucket),
    ]
    for item in selectivity_specs(selectivities):
        generate_args.extend(["--selectivity-spec", item])
    run_hybrid(generate_args, stdout_path=generate_log, stderr_path=generate_log.with_suffix(".err.log"))

    summary_json = work_dir / "uniform_exact_selectivity_summary.json"
    manifest_args = [
        "build-manifest-from-summary",
        "--summary-json",
        str(summary_json),
        "--index-prefix",
        str(data_paths["index_prefix"]),
        "--index-type",
        spec.index_type,
        "--selector-type",
        "intersect",
        "--manifest",
        str(manifest),
    ]
    run_hybrid(manifest_args, stdout_path=build_manifest_log, stderr_path=build_manifest_log.with_suffix(".err.log"))

    payload = {
        "dataset": spec.name,
        "workload_dir": str(work_dir),
        "summary_json": str(summary_json),
        "manifest": str(manifest),
        "selectivities": selectivities,
    }
    write_json(dataset_experiment_dir(experiment_root, spec.name) / "workload_meta.json", payload)
    return payload


def filter_manifest(manifest: dict[str, Any], allowed_labels: set[str]) -> dict[str, Any]:
    filtered = {**manifest}
    filtered["buckets"] = [bucket for bucket in manifest.get("buckets", []) if bucket.get("label") in allowed_labels]
    return filtered


def prepare_yfcc_workloads(
    data_root: Path,
    experiment_root: Path,
    targets: list[float],
    min_query_count: int,
    max_scanned_queries: int,
) -> dict[str, Any]:
    paths = default_yfcc_paths(data_root, experiment_root)
    scan_log = paths["experiment_root"] / "logs" / "scan_single_label.log"
    generate_log = paths["experiment_root"] / "logs" / "generate_random_single_label.log"
    manifest_log = paths["experiment_root"] / "logs" / "build_manifest.log"
    paths["scan_dir"].mkdir(parents=True, exist_ok=True)
    paths["workload_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifest_dir"].mkdir(parents=True, exist_ok=True)

    scan_args = [
        "scan-single-label",
        "--base-labels",
        str(paths["base_labels"]),
        "--query-labels",
        str(paths["query_labels"]),
        "--out-dir",
        str(paths["scan_dir"]),
        "--min-query-count",
        str(min_query_count),
    ]
    if max_scanned_queries > 0:
        scan_args.extend(["--max-scanned-queries", str(max_scanned_queries)])
    for target in targets:
        scan_args.extend(["--target", str(target)])
    run_hybrid(scan_args, stdout_path=scan_log, stderr_path=scan_log.with_suffix(".err.log"))

    summary_json = paths["scan_dir"] / "single_label_scan_summary.json"
    generate_args = [
        "generate-random-single-label-workloads",
        "--base-bin",
        str(paths["base_bin"]),
        "--base-labels",
        str(paths["base_labels"]),
        "--query-bin",
        str(paths["query_bin"]),
        "--scan-summary",
        str(summary_json),
        "--out-dir",
        str(paths["workload_dir"]),
        "--index-type",
        YFCC_DATASET.index_type,
        "--selector-type",
        "intersect",
        "--skip-synthetic-high",
    ]
    run_hybrid(generate_args, stdout_path=generate_log, stderr_path=generate_log.with_suffix(".err.log"))

    workload_summary = paths["workload_dir"] / "random_single_label_workloads_summary.json"
    full_manifest = paths["manifest_dir"] / "yfcc10m_real_full_manifest.json"
    run_hybrid(
        [
            "build-manifest-from-summary",
            "--summary-json",
            str(workload_summary),
            "--index-prefix",
            str(paths["index_prefix"]),
            "--index-type",
            YFCC_DATASET.index_type,
            "--selector-type",
            "intersect",
            "--manifest",
            str(full_manifest),
        ],
        stdout_path=manifest_log,
        stderr_path=manifest_log.with_suffix(".err.log"),
    )

    scan_payload = read_json(summary_json)
    allowed_labels = {item["target_name"] for item in scan_payload.get("recommendations", []) if item.get("target_name")}
    filtered = filter_manifest(read_json(full_manifest), allowed_labels)
    write_json(paths["filtered_manifest"], filtered)
    write_json(paths["target_summary"], scan_payload.get("recommendations", []))

    payload = {
        "dataset": "yfcc10m",
        "scan_summary": str(summary_json),
        "workload_summary": str(workload_summary),
        "manifest": str(paths["filtered_manifest"]),
        "targets": targets,
        "allowed_labels": sorted(allowed_labels),
    }
    write_json(paths["experiment_root"] / "workload_meta.json", payload)
    return payload


def load_bucket_plan(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = read_json(path)
    defaults = payload.get("defaults", {}) if isinstance(payload, dict) else {}
    entries_raw = payload.get("entries", []) if isinstance(payload, dict) else []
    entries: dict[str, dict[str, Any]] = {}
    for item in entries_raw:
        key = str(item["bucket"])
        entries[key] = {
            "route": item.get("route"),
            "L": item.get("L"),
        }
    return entries, defaults


def resolve_bucket_settings(
    bucket: dict[str, Any],
    plan_entries: dict[str, dict[str, Any]],
    plan_defaults: dict[str, Any],
    default_route: str,
    default_l: int,
) -> tuple[str, int]:
    for key in (str(bucket.get("name")), str(bucket.get("label"))):
        if key in plan_entries:
            entry = plan_entries[key]
            route = str(entry.get("route") or plan_defaults.get("route") or default_route)
            l_value = int(entry.get("L") or plan_defaults.get("L") or default_l)
            return route, l_value
    route = str(plan_defaults.get("route") or default_route)
    l_value = int(plan_defaults.get("L") or default_l)
    return route, l_value


def find_descendants(root_pid: int) -> set[int]:
    result: set[int] = set()
    pending = [root_pid]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        children_path = Path("/proc") / str(current) / "task" / str(current) / "children"
        try:
            raw = children_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if not raw:
            continue
        for token in raw.split():
            try:
                pending.append(int(token))
            except ValueError:
                continue
    return result


def read_proc_cpu_ticks(pid: int) -> int:
    stat_path = Path("/proc") / str(pid) / "stat"
    content = stat_path.read_text(encoding="utf-8")
    close_paren = content.rfind(")")
    if close_paren < 0:
        return 0
    fields = content[close_paren + 2 :].split()
    if len(fields) < 15:
        return 0
    return int(fields[11]) + int(fields[12])


def sample_process_tree_cpu(root_pid: int) -> int:
    total = 0
    for pid in find_descendants(root_pid):
        try:
            total += read_proc_cpu_ticks(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def infer_block_device(path: Path) -> str | None:
    target = path.resolve()
    best_mount: tuple[int, str] | None = None
    with Path("/proc/self/mountinfo").open("r", encoding="utf-8") as reader:
        for line in reader:
            left, separator, right = line.strip().partition(" - ")
            if not separator:
                continue
            left_fields = left.split()
            right_fields = right.split()
            if len(left_fields) < 5 or len(right_fields) < 2:
                continue
            mount_point = Path(left_fields[4])
            source = right_fields[1]
            try:
                target.relative_to(mount_point)
            except ValueError:
                continue
            if not source.startswith("/dev/"):
                continue
            match_len = len(str(mount_point))
            if best_mount is None or match_len > best_mount[0]:
                best_mount = (match_len, source)
    if best_mount is None:
        return None
    return Path(best_mount[1]).name


def read_block_stats(device: str) -> dict[str, int]:
    stat_path = Path("/sys/class/block") / device / "stat"
    fields = stat_path.read_text(encoding="utf-8").split()
    if len(fields) < 11:
        raise RuntimeError(f"unexpected stat payload for device {device}")
    return {
        "reads_completed": int(fields[0]),
        "reads_merged": int(fields[1]),
        "sectors_read": int(fields[2]),
        "read_ms": int(fields[3]),
        "writes_completed": int(fields[4]),
        "writes_merged": int(fields[5]),
        "sectors_written": int(fields[6]),
        "write_ms": int(fields[7]),
        "io_in_progress": int(fields[8]),
        "io_ms": int(fields[9]),
        "weighted_io_ms": int(fields[10]),
    }


def summarize_monitor_samples(
    samples: list[dict[str, Any]],
    clock_ticks_per_second: int,
) -> dict[str, Any]:
    if len(samples) < 2:
        first = samples[0] if samples else {}
        return {
            "sample_count": len(samples),
            "device": first.get("device"),
            "elapsed_s": 0.0,
            "avg_cpu_pct": None,
            "max_cpu_pct": None,
            "avg_read_mb_s": None,
            "max_read_mb_s": None,
            "avg_write_mb_s": None,
            "max_write_mb_s": None,
            "avg_read_iops": None,
            "max_read_iops": None,
            "avg_write_iops": None,
            "max_write_iops": None,
            "avg_disk_util_pct": None,
            "max_disk_util_pct": None,
            "avg_await_ms": None,
            "max_await_ms": None,
        }

    total_wall = 0.0
    total_cpu_s = 0.0
    total_read_mb = 0.0
    total_write_mb = 0.0
    total_read_ios = 0.0
    total_write_ios = 0.0
    total_io_util_ms = 0.0
    total_io_wait_weight = 0.0
    total_io_ops = 0.0
    max_cpu = 0.0
    max_read_mb_s = 0.0
    max_write_mb_s = 0.0
    max_read_iops = 0.0
    max_write_iops = 0.0
    max_util = 0.0
    max_await = 0.0

    for earlier, later in zip(samples, samples[1:]):
        dt = float(later["timestamp"] - earlier["timestamp"])
        if dt <= 0.0:
            continue
        total_wall += dt
        delta_ticks = max(int(later["cpu_ticks"]) - int(earlier["cpu_ticks"]), 0)
        delta_cpu_s = delta_ticks / float(clock_ticks_per_second)
        total_cpu_s += delta_cpu_s
        cpu_pct = 100.0 * delta_cpu_s / dt
        max_cpu = max(max_cpu, cpu_pct)

        if earlier.get("disk") is None or later.get("disk") is None:
            continue
        disk_a = earlier["disk"]
        disk_b = later["disk"]
        delta_read_sectors = max(int(disk_b["sectors_read"]) - int(disk_a["sectors_read"]), 0)
        delta_write_sectors = max(int(disk_b["sectors_written"]) - int(disk_a["sectors_written"]), 0)
        delta_reads = max(int(disk_b["reads_completed"]) - int(disk_a["reads_completed"]), 0)
        delta_writes = max(int(disk_b["writes_completed"]) - int(disk_a["writes_completed"]), 0)
        delta_io_ms = max(int(disk_b["io_ms"]) - int(disk_a["io_ms"]), 0)
        delta_read_ms = max(int(disk_b["read_ms"]) - int(disk_a["read_ms"]), 0)
        delta_write_ms = max(int(disk_b["write_ms"]) - int(disk_a["write_ms"]), 0)

        read_mb = delta_read_sectors * SECTOR_SIZE_BYTES / (1024.0 * 1024.0)
        write_mb = delta_write_sectors * SECTOR_SIZE_BYTES / (1024.0 * 1024.0)
        total_read_mb += read_mb
        total_write_mb += write_mb
        total_read_ios += delta_reads
        total_write_ios += delta_writes
        total_io_util_ms += delta_io_ms
        total_io_wait_weight += delta_read_ms + delta_write_ms
        total_io_ops += delta_reads + delta_writes

        read_mb_s = read_mb / dt
        write_mb_s = write_mb / dt
        read_iops = delta_reads / dt
        write_iops = delta_writes / dt
        util_pct = delta_io_ms / (dt * 10.0)
        await_ms = 0.0 if delta_reads + delta_writes == 0 else (delta_read_ms + delta_write_ms) / float(delta_reads + delta_writes)
        max_read_mb_s = max(max_read_mb_s, read_mb_s)
        max_write_mb_s = max(max_write_mb_s, write_mb_s)
        max_read_iops = max(max_read_iops, read_iops)
        max_write_iops = max(max_write_iops, write_iops)
        max_util = max(max_util, util_pct)
        max_await = max(max_await, await_ms)

    avg_cpu = None if total_wall == 0.0 else 100.0 * total_cpu_s / total_wall
    avg_read_mb_s = None if total_wall == 0.0 else total_read_mb / total_wall
    avg_write_mb_s = None if total_wall == 0.0 else total_write_mb / total_wall
    avg_read_iops = None if total_wall == 0.0 else total_read_ios / total_wall
    avg_write_iops = None if total_wall == 0.0 else total_write_ios / total_wall
    avg_util = None if total_wall == 0.0 else total_io_util_ms / (total_wall * 10.0)
    avg_await = None if total_io_ops == 0.0 else total_io_wait_weight / total_io_ops

    return {
        "sample_count": len(samples),
        "device": samples[0].get("device"),
        "elapsed_s": round(total_wall, 6),
        "avg_cpu_pct": None if avg_cpu is None else round(avg_cpu, 6),
        "max_cpu_pct": None if total_wall == 0.0 else round(max_cpu, 6),
        "avg_read_mb_s": None if avg_read_mb_s is None else round(avg_read_mb_s, 6),
        "max_read_mb_s": None if total_wall == 0.0 else round(max_read_mb_s, 6),
        "avg_write_mb_s": None if avg_write_mb_s is None else round(avg_write_mb_s, 6),
        "max_write_mb_s": None if total_wall == 0.0 else round(max_write_mb_s, 6),
        "avg_read_iops": None if avg_read_iops is None else round(avg_read_iops, 6),
        "max_read_iops": None if total_wall == 0.0 else round(max_read_iops, 6),
        "avg_write_iops": None if avg_write_iops is None else round(avg_write_iops, 6),
        "max_write_iops": None if total_wall == 0.0 else round(max_write_iops, 6),
        "avg_disk_util_pct": None if avg_util is None else round(avg_util, 6),
        "max_disk_util_pct": None if total_wall == 0.0 else round(max_util, 6),
        "avg_await_ms": None if avg_await is None else round(avg_await, 6),
        "max_await_ms": None if total_wall == 0.0 else round(max_await, 6),
    }


def run_with_monitor(
    command: list[str],
    cwd: Path,
    device: str | None,
    sample_interval_s: float,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    log("+ " + shell_join(command))
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    samples: list[dict[str, Any]] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while True:
            disk_sample = read_block_stats(device) if device is not None else None
            samples.append(
                {
                    "timestamp": time.monotonic(),
                    "cpu_ticks": sample_process_tree_cpu(process.pid),
                    "disk": disk_sample,
                    "device": device,
                }
            )
            if process.poll() is not None:
                break
            time.sleep(sample_interval_s)
        return_code = process.wait()
    finally:
        stdout_file.close()
        stderr_file.close()

    summary = summarize_monitor_samples(samples, clock_ticks)
    summary["returncode"] = return_code
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames, extrasaction="ignore")
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def collect_sweep_rows(experiment_root: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        path = sweep_root(experiment_root, dataset) / "aggregated_results.jsonl"
        if not path.exists():
            log(f"[warn] missing sweep results for {dataset}: {path}")
            continue
        rows.extend(load_jsonl(path))
    return rows


def classify_selectivity(selectivity: float | None, low_threshold: float, high_threshold: float) -> str:
    if selectivity is None:
        return "unknown"
    if selectivity <= low_threshold:
        return "low"
    if selectivity >= high_threshold:
        return "high"
    return "middle"


def infer_bottleneck(row: dict[str, Any], cpu_saturation_pct: float, disk_util_saturation_pct: float) -> str:
    cpu = row.get("avg_cpu_pct")
    disk_util = row.get("avg_disk_util_pct")
    read_mb_s = row.get("avg_read_mb_s")
    if cpu is None or disk_util is None:
        return "unknown"
    cpu_high = float(cpu) >= cpu_saturation_pct
    disk_high = float(disk_util) >= disk_util_saturation_pct or (read_mb_s is not None and float(read_mb_s) > 0.0 and disk_high_by_iops(row))
    if cpu_high and not disk_high:
        return "cpu-bound-evidence"
    if disk_high and not cpu_high:
        return "ssd-bound-evidence"
    if cpu_high and disk_high:
        return "mixed-bound-evidence"
    return "not-saturated"


def disk_high_by_iops(row: dict[str, Any]) -> bool:
    mean_ios = row.get("mean_ios")
    read_mb_s = row.get("avg_read_mb_s")
    if mean_ios is None or read_mb_s is None:
        return False
    return float(mean_ios) >= 128.0 and float(read_mb_s) >= 50.0


def summarize_formal_results(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = resolve_path(args.experiment_root)
    datasets = args.dataset or ["fashion_mnist784", "gist960", "glove100", "yfcc10m"]
    rows = collect_sweep_rows(experiment_root, datasets)
    out_dir = resolve_path(args.out_dir) if args.out_dir else experiment_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        selectivity = row.get("selectivity_midpoint", row.get("bucket_midpoint"))
        selectivity_value = None if selectivity is None else float(selectivity)
        group = classify_selectivity(selectivity_value, args.low_threshold, args.high_threshold)
        enriched_row = {
            **row,
            "selectivity_group": group,
            "bottleneck_evidence": infer_bottleneck(row, args.cpu_saturation_pct, args.disk_util_saturation_pct),
        }
        enriched.append(enriched_row)

    write_jsonl(out_dir / "formal_results.jsonl", enriched)
    write_csv(out_dir / "formal_results.csv", enriched)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        key = (str(row.get("dataset")), str(row.get("selectivity_group")))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, group), group_rows in sorted(grouped.items()):
        if group == "middle":
            continue
        cpu_values = [float(row["avg_cpu_pct"]) for row in group_rows if row.get("avg_cpu_pct") is not None]
        disk_values = [float(row["avg_disk_util_pct"]) for row in group_rows if row.get("avg_disk_util_pct") is not None]
        read_values = [float(row["avg_read_mb_s"]) for row in group_rows if row.get("avg_read_mb_s") is not None]
        qps_values = [float(row["qps"]) for row in group_rows if row.get("qps") is not None]
        evidence_counts: dict[str, int] = {}
        for row in group_rows:
            evidence = str(row.get("bottleneck_evidence"))
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
        summary_rows.append(
            {
                "dataset": dataset,
                "selectivity_group": group,
                "rows": len(group_rows),
                "avg_cpu_pct_mean": None if not cpu_values else sum(cpu_values) / len(cpu_values),
                "avg_disk_util_pct_mean": None if not disk_values else sum(disk_values) / len(disk_values),
                "avg_read_mb_s_mean": None if not read_values else sum(read_values) / len(read_values),
                "qps_min": None if not qps_values else min(qps_values),
                "qps_max": None if not qps_values else max(qps_values),
                "evidence_counts": json.dumps(evidence_counts, sort_keys=True),
            }
        )

    write_csv(out_dir / "bottleneck_summary.csv", summary_rows)
    write_json(out_dir / "bottleneck_summary.json", summary_rows)
    payload = {
        "datasets": datasets,
        "rows": len(enriched),
        "formal_results_jsonl": str(out_dir / "formal_results.jsonl"),
        "formal_results_csv": str(out_dir / "formal_results.csv"),
        "bottleneck_summary_csv": str(out_dir / "bottleneck_summary.csv"),
        "bottleneck_summary_json": str(out_dir / "bottleneck_summary.json"),
    }
    write_json(out_dir / "summary_meta.json", payload)
    return payload


def write_single_bucket_manifest(manifest: dict[str, Any], bucket: dict[str, Any], destination: Path) -> None:
    payload = {**manifest, "buckets": [bucket]}
    write_json(destination, payload)


def default_manifest_for_dataset(data_root: Path, experiment_root: Path, dataset: str) -> Path:
    if dataset == "yfcc10m":
        return default_yfcc_paths(data_root, experiment_root)["filtered_manifest"]
    return manifest_path(experiment_root, dataset)


def run_thread_sweep(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    experiment_root = resolve_path(args.experiment_root)
    manifest_file = resolve_path(args.manifest) if args.manifest else default_manifest_for_dataset(data_root, experiment_root, args.dataset)
    manifest = read_json(manifest_file)
    run_root = resolve_path(args.out_dir) if args.out_dir else sweep_root(experiment_root, args.dataset)
    run_root.mkdir(parents=True, exist_ok=True)

    bucket_filter = set(args.bucket or [])
    plan_entries, plan_defaults = load_bucket_plan(resolve_path(args.bucket_plan_json) if args.bucket_plan_json else None)
    device = args.ssd_device or infer_block_device(Path(manifest["index_prefix"]))
    if device is None:
        log("[warn] could not infer block device; CPU monitoring will still run")
    else:
        log(f"[info] monitor device={device}")

    rows: list[dict[str, Any]] = []
    for bucket in manifest.get("buckets", []):
        bucket_name = str(bucket.get("name"))
        bucket_label = str(bucket.get("label"))
        if bucket_filter and bucket_name not in bucket_filter and bucket_label not in bucket_filter:
            continue

        route, l_value = resolve_bucket_settings(bucket, plan_entries, plan_defaults, args.default_route, args.default_l)
        bucket_root = run_root / bucket_name
        bucket_root.mkdir(parents=True, exist_ok=True)
        single_manifest = bucket_root / "manifest.json"
        write_single_bucket_manifest(manifest, bucket, single_manifest)

        for thread in args.threads:
            run_dir = bucket_root / f"t{thread}"
            result_path = run_dir / "results.jsonl"
            monitor_path = run_dir / "monitor.json"
            stdout_path = run_dir / "run.stdout.log"
            stderr_path = run_dir / "run.stderr.log"

            if args.reuse_existing and result_path.exists() and monitor_path.exists():
                log(f"reuse {run_dir}")
                result_rows = load_jsonl(result_path)
                if not result_rows:
                    raise RuntimeError(f"no rows found in {result_path}")
                merged = {**result_rows[0], **read_json(monitor_path)}
                merged.update(
                    {
                        "dataset": args.dataset,
                        "bucket_name": bucket_name,
                        "bucket_label": bucket_label,
                        "bucket_midpoint": bucket.get("midpoint"),
                        "configured_route": route,
                        "configured_L": l_value,
                        "target_threads": thread,
                    }
                )
                rows.append(merged)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(HYBRID_SCRIPT),
                "run",
                "--manifest",
                str(single_manifest),
                "--build-dir",
                str(resolve_path(args.build_dir)),
                "--out-dir",
                str(run_dir),
                "--dataset-name",
                args.dataset_name or args.dataset,
                "--threads",
                str(thread),
                "--beamwidth",
                str(args.beamwidth),
                "--k",
                str(args.k),
                "--similarity",
                args.similarity,
                "--nbr-type",
                args.nbr_type,
                "--mem-l",
                str(args.mem_l),
                "--routes",
                route,
                "--l-values",
                str(l_value),
                "--timeout",
                str(args.timeout),
            ]
            if args.prefilter_rerank_json:
                command.extend(["--prefilter-rerank-json", str(resolve_path(args.prefilter_rerank_json))])

            if args.dry_run:
                log(f"[dry-run] {shell_join(command)}")
                continue

            monitor_summary = run_with_monitor(
                command,
                REPO_ROOT,
                device,
                args.sample_interval_s,
                stdout_path,
                stderr_path,
            )
            write_json(monitor_path, monitor_summary)

            result_rows = load_jsonl(result_path)
            if len(result_rows) != 1:
                raise RuntimeError(f"expected exactly one row in {result_path}, found {len(result_rows)}")
            merged = {
                **result_rows[0],
                **monitor_summary,
                "dataset": args.dataset,
                "bucket_name": bucket_name,
                "bucket_label": bucket_label,
                "bucket_midpoint": bucket.get("midpoint"),
                "bucket_lower": bucket.get("lower"),
                "bucket_upper": bucket.get("upper"),
                "configured_route": route,
                "configured_L": l_value,
                "target_threads": thread,
            }
            rows.append(merged)

    aggregate_jsonl = run_root / "aggregated_results.jsonl"
    aggregate_csv = run_root / "aggregated_results.csv"
    if not args.dry_run:
        write_jsonl(aggregate_jsonl, rows)
        write_csv(aggregate_csv, rows)
    summary = {
        "dataset": args.dataset,
        "manifest": str(manifest_file),
        "run_root": str(run_root),
        "device": device,
        "rows": len(rows),
        "aggregated_jsonl": str(aggregate_jsonl),
        "aggregated_csv": str(aggregate_csv),
    }
    write_json(run_root / "sweep_meta.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    defaults_parser = subparsers.add_parser("show-defaults", help="Print built-in dataset and selectivity defaults.")
    defaults_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    defaults_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))

    fetch_parser = subparsers.add_parser(
        "fetch-small-datasets",
        help="Download ann-benchmarks HDF5 datasets and convert them to PipeANN .bin/.ibin.",
    )
    fetch_parser.add_argument("--dataset", action="append", choices=sorted(SMALL_DATASETS), help="Dataset to fetch. May be repeated. Defaults to all 3 small datasets.")
    fetch_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.add_argument("--chunk-rows", type=int, default=100_000)

    build_parser_small = subparsers.add_parser(
        "build-small-indexes",
        help="Build canonical PipeANN disk indexes for the 3 downloadable small datasets.",
    )
    build_parser_small.add_argument("--dataset", action="append", choices=sorted(SMALL_DATASETS), help="Dataset to build. May be repeated. Defaults to all 3 small datasets.")
    build_parser_small.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    build_parser_small.add_argument("--R", type=int, default=64)
    build_parser_small.add_argument("--build-L", type=int, default=96)
    build_parser_small.add_argument("--pq-bytes", type=int, default=16)
    build_parser_small.add_argument("--memory-gb", type=int, default=16)
    build_parser_small.add_argument("--threads", type=int, default=16)
    build_parser_small.add_argument("--force", action="store_true")

    prepare_small_parser = subparsers.add_parser(
        "prepare-small-workloads",
        help="Generate exact-selectivity workloads and manifests for the 3 downloadable small datasets.",
    )
    prepare_small_parser.add_argument("--dataset", action="append", choices=sorted(SMALL_DATASETS), help="Dataset to prepare. May be repeated. Defaults to all 3 small datasets.")
    prepare_small_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    prepare_small_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    prepare_small_parser.add_argument("--selectivities", type=parse_csv_floats, default=list(DEFAULT_SMALL_SELECTIVITIES))
    prepare_small_parser.add_argument("--queries-per-bucket", type=int, default=1000)
    prepare_small_parser.add_argument("--probe-queries-per-bucket", type=int, default=5000)
    prepare_small_parser.add_argument("--seed", type=int, default=20260513)

    prepare_yfcc_parser = subparsers.add_parser(
        "prepare-yfcc-workloads",
        help="Scan YFCC real labels for the target selectivities and build a filtered manifest.",
    )
    prepare_yfcc_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    prepare_yfcc_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    prepare_yfcc_parser.add_argument("--targets", type=parse_csv_floats, default=list(DEFAULT_YFCC_SELECTIVITIES))
    prepare_yfcc_parser.add_argument("--min-query-count", type=int, default=20)
    prepare_yfcc_parser.add_argument("--max-scanned-queries", type=int, default=0)

    sweep_parser = subparsers.add_parser(
        "run-thread-sweep",
        help="Run single-bucket single-thread sweeps with external CPU/SSD monitoring.",
    )
    sweep_parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    sweep_parser.add_argument("--manifest")
    sweep_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    sweep_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    sweep_parser.add_argument("--out-dir")
    sweep_parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    sweep_parser.add_argument("--threads", type=parse_csv_ints, default=list(DEFAULT_THREAD_SWEEP))
    sweep_parser.add_argument("--bucket", action="append", help="Specific bucket name or label to run. May be repeated.")
    sweep_parser.add_argument("--bucket-plan-json")
    sweep_parser.add_argument("--dataset-name")
    sweep_parser.add_argument("--default-route", default="auto")
    sweep_parser.add_argument("--default-l", type=int, default=100)
    sweep_parser.add_argument("--beamwidth", type=int, default=4)
    sweep_parser.add_argument("--k", type=int, default=10)
    sweep_parser.add_argument("--similarity", default="l2")
    sweep_parser.add_argument("--nbr-type", default="pq")
    sweep_parser.add_argument("--mem-l", type=int, default=0)
    sweep_parser.add_argument("--prefilter-rerank-json")
    sweep_parser.add_argument("--timeout", type=int, default=7200)
    sweep_parser.add_argument("--ssd-device")
    sweep_parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    sweep_parser.add_argument("--reuse-existing", action="store_true")
    sweep_parser.add_argument("--dry-run", action="store_true")

    summarize_parser = subparsers.add_parser(
        "summarize-results",
        help="Merge formal thread-sweep results and emit CPU/SSD bottleneck evidence tables.",
    )
    summarize_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    summarize_parser.add_argument("--out-dir")
    summarize_parser.add_argument("--dataset", action="append", choices=sorted(ALL_DATASETS))
    summarize_parser.add_argument("--low-threshold", type=float, default=0.01)
    summarize_parser.add_argument("--high-threshold", type=float, default=0.10)
    summarize_parser.add_argument("--cpu-saturation-pct", type=float, default=80.0)
    summarize_parser.add_argument("--disk-util-saturation-pct", type=float, default=80.0)

    return parser


def handle_show_defaults(args: argparse.Namespace) -> int:
    data_root = resolve_path(args.data_root)
    experiment_root = resolve_path(args.experiment_root)
    payload = {
        "python": sys.executable,
        "data_root": str(data_root),
        "experiment_root": str(experiment_root),
        "threads": list(DEFAULT_THREAD_SWEEP),
        "small_datasets": {
            name: {
                "url": spec.source_url,
                "similarity": spec.similarity,
                "normalize_for_l2": spec.normalize_for_l2,
                "index_type": spec.index_type,
                "selectivities": list(DEFAULT_SMALL_SELECTIVITIES),
                "default_paths": {key: str(value) for key, value in default_small_paths(data_root, name).items()},
            }
            for name, spec in SMALL_DATASETS.items()
        },
        "yfcc10m": {
            "similarity": YFCC_DATASET.similarity,
            "index_type": YFCC_DATASET.index_type,
            "targets": list(DEFAULT_YFCC_SELECTIVITIES),
            "default_paths": {key: str(value) for key, value in default_yfcc_paths(data_root, experiment_root).items()},
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def datasets_or_all(values: list[str] | None) -> list[str]:
    return values or sorted(SMALL_DATASETS)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "show-defaults":
        return handle_show_defaults(args)
    if args.command == "fetch-small-datasets":
        data_root = resolve_path(args.data_root)
        summaries = [fetch_small_dataset(SMALL_DATASETS[name], data_root, args.force, args.chunk_rows) for name in datasets_or_all(args.dataset)]
        print(json.dumps({"datasets": summaries}, indent=2, sort_keys=True))
        return 0
    if args.command == "build-small-indexes":
        data_root = resolve_path(args.data_root)
        summaries = [
            build_small_index(
                SMALL_DATASETS[name],
                data_root,
                args.R,
                args.build_L,
                args.pq_bytes,
                args.memory_gb,
                args.threads,
                args.force,
            )
            for name in datasets_or_all(args.dataset)
        ]
        print(json.dumps({"indexes": summaries}, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-small-workloads":
        data_root = resolve_path(args.data_root)
        experiment_root = resolve_path(args.experiment_root)
        summaries = [
            prepare_small_workload(
                SMALL_DATASETS[name],
                data_root,
                experiment_root,
                args.selectivities,
                args.queries_per_bucket,
                args.probe_queries_per_bucket,
                args.seed,
            )
            for name in datasets_or_all(args.dataset)
        ]
        print(json.dumps({"workloads": summaries}, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-yfcc-workloads":
        payload = prepare_yfcc_workloads(
            resolve_path(args.data_root),
            resolve_path(args.experiment_root),
            args.targets,
            args.min_query_count,
            args.max_scanned_queries,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "run-thread-sweep":
        payload = run_thread_sweep(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize-results":
        payload = summarize_formal_results(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise