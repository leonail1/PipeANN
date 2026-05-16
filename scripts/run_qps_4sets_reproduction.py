#!/usr/bin/env python3
"""Orchestrate qps_4sets reproduction on the current mainline tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import shutil
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

from pipeann_hybrid_experiment import (
    DenseBitsetSidecar,
    PREFILTER_RERANK_ENV,
    SpmatMatrix,
    compute_exact_topk_ids,
    default_prefilter_rerank_l,
    load_bin_matrix,
    load_tags_by_id,
    write_bin_subset,
    write_spmat_subset,
    write_truthset_ids,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional runtime dependency.
    threadpool_limits = None


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
DEFAULT_RECALL_QUERY_COUNT = 1000
DEFAULT_RECALL_TARGET = 98.0
DEFAULT_RECALL_CALIBRATION_THREADS = 32
DEFAULT_RECALL_MAX_L = 65536
SECTOR_SIZE_BYTES = 512
DISK_METRIC_FIELDS = (
    "avg_read_mb_s",
    "max_read_mb_s",
    "avg_write_mb_s",
    "max_write_mb_s",
    "avg_read_iops",
    "max_read_iops",
    "avg_write_iops",
    "max_write_iops",
    "avg_disk_util_pct",
    "max_disk_util_pct",
    "avg_await_ms",
    "max_await_ms",
    "avg_read_await_ms",
    "max_read_await_ms",
    "avg_qd",
    "max_qd",
    "read_iops",
    "read_mb_s",
    "read_await_ms",
    "await_ms",
    "qdepth",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    similarity: str
    normalize_for_l2: bool
    index_type: str
    source_url: str | None = None


@dataclass(frozen=True)
class MonitorDeviceResolution:
    requested_device: str | None
    effective_device: str | None
    disk_metrics_status: str
    reason: str
    sanity_read_bytes: int = 0


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


_FILE_HASH_CACHE: dict[str, str] = {}


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def normalize_gt_threads(value: int) -> int:
    return available_cpu_count() if value == 0 else max(1, int(value))


def sha256_file(path: Path) -> str:
    resolved = str(path.resolve())
    cached = _FILE_HASH_CACHE.get(resolved)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[resolved] = value
    return value


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & 0x7FFFFFFF


def sample_row_ids(total_rows: int, requested_rows: int, seed: int, dataset: str, bucket: str) -> list[int]:
    sample_count = min(max(1, int(requested_rows)), int(total_rows))
    if sample_count >= total_rows:
        return list(range(total_rows))
    rng = np.random.default_rng(stable_seed(seed, dataset, bucket))
    return sorted(int(value) for value in rng.choice(total_rows, size=sample_count, replace=False))


def first_search_record(path: Path) -> dict[str, Any]:
    rows = [row for row in load_jsonl(path) if row.get("format") == "pipeann.hybrid.search.v1"]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one search record in {path}, found {len(rows)}")
    return rows[0]


def thread_limited_exact_topk(*args: Any, gt_threads: int, **kwargs: Any) -> np.ndarray:
    if threadpool_limits is None:
        old_env = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        try:
            for name in old_env:
                os.environ[name] = str(gt_threads)
            return compute_exact_topk_ids(*args, **kwargs)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    with threadpool_limits(limits=gt_threads):
        return compute_exact_topk_ids(*args, **kwargs)


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
    workload_summary = read_json(summary_json)
    runtime_log = dataset_experiment_dir(experiment_root, spec.name) / "logs" / "prepare_label_runtime.log"
    run_hybrid(
        [
            "prepare-index-prefix-for-labels",
            "--source-prefix",
            str(data_paths["index_prefix"]),
            "--dest-prefix",
            str(data_paths["index_prefix"]),
            "--label-file",
            str(workload_summary["base_labels"]),
            "--base-bin",
            str(data_paths["base_bin"]),
            "--index-type",
            spec.index_type,
            "--selector-type",
            "intersect",
            "--similarity",
            spec.similarity,
            "--nbr-type",
            "pq",
            "--sidecar-mode",
            "mixed",
        ],
        stdout_path=runtime_log,
        stderr_path=runtime_log.with_suffix(".err.log"),
    )
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
    runtime_log = paths["experiment_root"] / "logs" / "prepare_label_runtime.log"
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
    run_hybrid(
        [
            "prepare-index-prefix-for-labels",
            "--source-prefix",
            str(paths["index_prefix"]),
            "--dest-prefix",
            str(paths["index_prefix"]),
            "--label-file",
            str(paths["base_labels"]),
            "--base-bin",
            str(paths["base_bin"]),
            "--index-type",
            YFCC_DATASET.index_type,
            "--selector-type",
            "intersect",
            "--similarity",
            YFCC_DATASET.similarity,
            "--nbr-type",
            "pq",
            "--sidecar-mode",
            "mixed",
        ],
        stdout_path=runtime_log,
        stderr_path=runtime_log.with_suffix(".err.log"),
    )
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


def load_bucket_plan(path: Path | None, dataset: str | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = read_json(path)
    defaults = payload.get("defaults", {}) if isinstance(payload, dict) else {}
    entries_raw = payload.get("entries", []) if isinstance(payload, dict) else []
    entries: dict[str, dict[str, Any]] = {}
    for item in entries_raw:
        if dataset is not None and item.get("dataset") not in (None, dataset):
            continue
        key = str(item["bucket"])
        entries[key] = dict(item)
        entries[key]["route"] = item.get("route")
        entries[key]["L"] = item.get("L") or item.get("chosen_L")
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


def disk_index_file_for_prefix(index_prefix: str | Path) -> Path:
    return Path(str(index_prefix) + "_disk.index")


def block_stat_path(device: str) -> Path:
    return Path("/sys/class/block") / device / "stat"


def block_size_path(device: str) -> Path:
    return Path("/sys/class/block") / device / "size"


def block_device_size(device: str) -> int | None:
    try:
        return int(block_size_path(device).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def is_whole_nvme_name(device: str) -> bool:
    return re.fullmatch(r"nvme\d+(?:c\d+)?n\d+", device) is not None


def candidate_counter_devices(requested_device: str) -> list[str]:
    candidates: list[str] = []
    requested_size = block_device_size(requested_device)
    for path in sorted(Path("/sys/class/block").glob("nvme*")):
        name = path.name
        if not is_whole_nvme_name(name) or not block_stat_path(name).exists():
            continue
        if name == requested_device or requested_size is None or block_device_size(name) == requested_size:
            candidates.append(name)
    if requested_device not in candidates and block_stat_path(requested_device).exists():
        candidates.insert(0, requested_device)
    return candidates


def read_block_stats_or_none(device: str) -> dict[str, int] | None:
    try:
        return read_block_stats(device)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None


def read_delta_score(before: dict[str, int] | None, after: dict[str, int] | None) -> tuple[int, int]:
    if before is None or after is None:
        return (0, 0)
    read_ios = max(int(after["reads_completed"]) - int(before["reads_completed"]), 0)
    read_sectors = max(int(after["sectors_read"]) - int(before["sectors_read"]), 0)
    return (read_sectors, read_ios)


def resolve_monitor_device(requested_device: str | None, index_prefix: str | Path) -> MonitorDeviceResolution:
    if requested_device is None:
        return MonitorDeviceResolution(None, None, "no_device", "could not infer a block device for the index prefix")
    if not block_stat_path(requested_device).exists():
        return MonitorDeviceResolution(requested_device, None, "invalid_device", f"missing {block_stat_path(requested_device)}")

    disk_index_path = disk_index_file_for_prefix(index_prefix)
    if not disk_index_path.exists():
        return MonitorDeviceResolution(
            requested_device,
            None,
            "unavailable_counter_static",
            f"cannot run direct-read sanity check because {disk_index_path} is missing",
        )

    candidates = candidate_counter_devices(requested_device)
    if not candidates:
        return MonitorDeviceResolution(requested_device, None, "invalid_device", "no candidate block stat counters found")

    before = {device: read_block_stats_or_none(device) for device in candidates}
    count_4k = min(16_384, max(1, disk_index_path.stat().st_size // 4096))
    command = [
        "dd",
        f"if={disk_index_path}",
        "of=/dev/null",
        "bs=4096",
        f"count={count_4k}",
        "iflag=direct",
        "status=none",
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return MonitorDeviceResolution(
            requested_device,
            None,
            "unavailable_counter_static",
            f"direct-read sanity check failed: {exc}",
        )

    after = {device: read_block_stats_or_none(device) for device in candidates}
    scored = [(read_delta_score(before.get(device), after.get(device)), device) for device in candidates]
    scored.sort(reverse=True)
    (read_sectors, read_ios), effective_device = scored[0]
    if read_sectors <= 0 and read_ios <= 0:
        return MonitorDeviceResolution(
            requested_device,
            None,
            "unavailable_counter_static",
            "no candidate block counter changed during direct-read sanity check",
            sanity_read_bytes=count_4k * 4096,
        )
    reason = "requested counter passed sanity check" if effective_device == requested_device else f"remapped static counter {requested_device} to active counter {effective_device}"
    return MonitorDeviceResolution(
        requested_device,
        effective_device,
        "ok",
        reason,
        sanity_read_bytes=count_4k * 4096,
    )


def read_block_stats(device: str) -> dict[str, int]:
    stat_path = block_stat_path(device)
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
            "avg_read_await_ms": None,
            "max_read_await_ms": None,
            "avg_qd": None,
            "max_qd": None,
            "read_iops": None,
            "read_mb_s": None,
            "read_await_ms": None,
            "await_ms": None,
            "qdepth": None,
        }

    total_wall = 0.0
    total_cpu_s = 0.0
    total_read_mb = 0.0
    total_write_mb = 0.0
    total_read_ios = 0.0
    total_write_ios = 0.0
    total_io_util_ms = 0.0
    total_weighted_io_ms = 0.0
    total_io_wait_weight = 0.0
    total_read_wait_ms = 0.0
    total_io_ops = 0.0
    max_cpu = 0.0
    max_read_mb_s = 0.0
    max_write_mb_s = 0.0
    max_read_iops = 0.0
    max_write_iops = 0.0
    max_util = 0.0
    max_await = 0.0
    max_read_await = 0.0
    max_qd = 0.0

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
        delta_weighted_io_ms = max(int(disk_b["weighted_io_ms"]) - int(disk_a["weighted_io_ms"]), 0)
        delta_read_ms = max(int(disk_b["read_ms"]) - int(disk_a["read_ms"]), 0)
        delta_write_ms = max(int(disk_b["write_ms"]) - int(disk_a["write_ms"]), 0)

        read_mb = delta_read_sectors * SECTOR_SIZE_BYTES / (1024.0 * 1024.0)
        write_mb = delta_write_sectors * SECTOR_SIZE_BYTES / (1024.0 * 1024.0)
        total_read_mb += read_mb
        total_write_mb += write_mb
        total_read_ios += delta_reads
        total_write_ios += delta_writes
        total_io_util_ms += delta_io_ms
        total_weighted_io_ms += delta_weighted_io_ms
        total_io_wait_weight += delta_read_ms + delta_write_ms
        total_read_wait_ms += delta_read_ms
        total_io_ops += delta_reads + delta_writes

        read_mb_s = read_mb / dt
        write_mb_s = write_mb / dt
        read_iops = delta_reads / dt
        write_iops = delta_writes / dt
        util_pct = delta_io_ms / (dt * 10.0)
        qd = delta_weighted_io_ms / (dt * 1000.0)
        await_ms = 0.0 if delta_reads + delta_writes == 0 else (delta_read_ms + delta_write_ms) / float(delta_reads + delta_writes)
        read_await_ms = 0.0 if delta_reads == 0 else delta_read_ms / float(delta_reads)
        max_read_mb_s = max(max_read_mb_s, read_mb_s)
        max_write_mb_s = max(max_write_mb_s, write_mb_s)
        max_read_iops = max(max_read_iops, read_iops)
        max_write_iops = max(max_write_iops, write_iops)
        max_util = max(max_util, util_pct)
        max_await = max(max_await, await_ms)
        max_read_await = max(max_read_await, read_await_ms)
        max_qd = max(max_qd, qd)

    avg_cpu = None if total_wall == 0.0 else 100.0 * total_cpu_s / total_wall
    avg_read_mb_s = None if total_wall == 0.0 else total_read_mb / total_wall
    avg_write_mb_s = None if total_wall == 0.0 else total_write_mb / total_wall
    avg_read_iops = None if total_wall == 0.0 else total_read_ios / total_wall
    avg_write_iops = None if total_wall == 0.0 else total_write_ios / total_wall
    avg_util = None if total_wall == 0.0 else total_io_util_ms / (total_wall * 10.0)
    avg_await = None if total_io_ops == 0.0 else total_io_wait_weight / total_io_ops
    avg_read_await = None if total_read_ios == 0.0 else total_read_wait_ms / total_read_ios
    avg_qd = None if total_wall == 0.0 else total_weighted_io_ms / (total_wall * 1000.0)

    avg_read_iops_value = None if avg_read_iops is None else round(avg_read_iops, 6)
    avg_read_mb_s_value = None if avg_read_mb_s is None else round(avg_read_mb_s, 6)
    avg_await_value = None if avg_await is None else round(avg_await, 6)
    avg_read_await_value = None if avg_read_await is None else round(avg_read_await, 6)
    avg_qd_value = None if avg_qd is None else round(avg_qd, 6)
    return {
        "sample_count": len(samples),
        "device": samples[0].get("device"),
        "elapsed_s": round(total_wall, 6),
        "avg_cpu_pct": None if avg_cpu is None else round(avg_cpu, 6),
        "max_cpu_pct": None if total_wall == 0.0 else round(max_cpu, 6),
        "avg_read_mb_s": avg_read_mb_s_value,
        "max_read_mb_s": None if total_wall == 0.0 else round(max_read_mb_s, 6),
        "avg_write_mb_s": None if avg_write_mb_s is None else round(avg_write_mb_s, 6),
        "max_write_mb_s": None if total_wall == 0.0 else round(max_write_mb_s, 6),
        "avg_read_iops": avg_read_iops_value,
        "max_read_iops": None if total_wall == 0.0 else round(max_read_iops, 6),
        "avg_write_iops": None if avg_write_iops is None else round(avg_write_iops, 6),
        "max_write_iops": None if total_wall == 0.0 else round(max_write_iops, 6),
        "avg_disk_util_pct": None if avg_util is None else round(avg_util, 6),
        "max_disk_util_pct": None if total_wall == 0.0 else round(max_util, 6),
        "avg_await_ms": avg_await_value,
        "max_await_ms": None if total_wall == 0.0 else round(max_await, 6),
        "avg_read_await_ms": avg_read_await_value,
        "max_read_await_ms": None if total_wall == 0.0 else round(max_read_await, 6),
        "avg_qd": avg_qd_value,
        "max_qd": None if total_wall == 0.0 else round(max_qd, 6),
        "read_iops": avg_read_iops_value,
        "read_mb_s": avg_read_mb_s_value,
        "read_await_ms": avg_read_await_value,
        "await_ms": avg_await_value,
        "qdepth": avg_qd_value,
    }


def invalidate_disk_metrics(summary: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    updated = {**summary}
    for field in DISK_METRIC_FIELDS:
        updated[field] = None
    updated["disk_metrics_status"] = status
    updated["disk_metrics_reason"] = reason
    return updated


def add_disk_status(
    summary: dict[str, Any],
    resolution: MonitorDeviceResolution,
    iostat_path: Path | None,
    block_latency_path: Path | None,
) -> dict[str, Any]:
    if resolution.disk_metrics_status != "ok":
        summary = invalidate_disk_metrics(summary, resolution.disk_metrics_status, resolution.reason)
    else:
        summary = {
            **summary,
            "disk_metrics_status": "ok",
            "disk_metrics_reason": resolution.reason,
        }
    summary.update(
        {
            "requested_device": resolution.requested_device,
            "effective_device": resolution.effective_device,
            "disk_sanity_read_bytes": resolution.sanity_read_bytes,
            "iostat_log": None if iostat_path is None else str(iostat_path),
            "block_latency_log": None if block_latency_path is None else str(block_latency_path),
            "block_latency_status": "unavailable_not_captured" if block_latency_path is not None else None,
            "block_latency_p50_ms": None,
            "block_latency_p95_ms": None,
            "block_latency_p99_ms": None,
            "block_latency_mean_ms": None,
            "block_latency_count": None,
            "block_latency_trace_dev": None,
        }
    )
    return summary


def start_iostat(device: str | None, sample_interval_s: float, iostat_path: Path | None) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    if device is None or iostat_path is None or shutil.which("iostat") is None:
        return None, None
    ensure_parent(iostat_path)
    interval = max(1, int(round(sample_interval_s)))
    handle = iostat_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["iostat", "-x", "-y", str(interval), device],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle


def stop_iostat(process: subprocess.Popen[Any] | None, handle: Any | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if handle is not None:
        handle.close()


def sudo_noninteractive_available() -> bool:
    if shutil.which("sudo") is None:
        return False
    return subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def bpftrace_block_latency_script() -> str:
    return "\n".join(
        [
            'tracepoint:block:block_rq_issue /strncmp(args->rwbs, "R", 1) == 0/ {',
            "  @start[args->dev, args->sector] = nsecs;",
            "}",
            "tracepoint:block:block_rq_complete /@start[args->dev, args->sector]/ {",
            "  $lat_us = (nsecs - @start[args->dev, args->sector]) / 1000;",
            "  @lat[args->dev] = lhist($lat_us, 0, 5000, 10);",
            "  @count[args->dev] = count();",
            "  @sum[args->dev] = sum($lat_us);",
            "  delete(@start[args->dev, args->sector]);",
            "}",
            "",
        ]
    )


def start_block_latency_trace(
    resolution: MonitorDeviceResolution,
    block_latency_path: Path | None,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    if block_latency_path is None:
        return None, None
    ensure_parent(block_latency_path)
    if resolution.disk_metrics_status != "ok":
        block_latency_path.write_text(
            f"BLOCK_LATENCY_STATUS unavailable_monitor_{resolution.disk_metrics_status}\n",
            encoding="utf-8",
        )
        return None, None
    if shutil.which("bpftrace") is None:
        block_latency_path.write_text("BLOCK_LATENCY_STATUS unavailable_no_bpftrace\n", encoding="utf-8")
        return None, None
    if not sudo_noninteractive_available():
        block_latency_path.write_text("BLOCK_LATENCY_STATUS unavailable_no_sudo\n", encoding="utf-8")
        return None, None
    script_path = block_latency_path.with_suffix(block_latency_path.suffix + ".bt")
    script_path.write_text(bpftrace_block_latency_script(), encoding="utf-8")
    handle = block_latency_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["sudo", "-n", "bpftrace", str(script_path)],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    time.sleep(0.8)
    if process.poll() is not None:
        handle.close()
        return None, None
    return process, handle


def stop_block_latency_trace(process: subprocess.Popen[Any] | None, handle: Any | None) -> None:
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    if handle is not None:
        handle.close()


def parse_block_latency_log(block_latency_path: Path | None) -> dict[str, Any]:
    empty = {
        "block_latency_status": None,
        "block_latency_p50_ms": None,
        "block_latency_p95_ms": None,
        "block_latency_p99_ms": None,
        "block_latency_mean_ms": None,
        "block_latency_count": None,
        "block_latency_trace_dev": None,
    }
    if block_latency_path is None:
        return empty
    if not block_latency_path.exists():
        return {**empty, "block_latency_status": "unavailable_missing_log"}
    text = block_latency_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("BLOCK_LATENCY_STATUS "):
            return {**empty, "block_latency_status": line.split(None, 1)[1].strip()}

    count_re = re.compile(r"@count\[(\d+)\]:\s+(\d+)")
    sum_re = re.compile(r"@sum\[(\d+)\]:\s+(\d+)")
    header_re = re.compile(r"@lat\[(\d+)\]:")
    bucket_re = re.compile(r"\[\s*(\d+),\s*(\d+)\)\s+(\d+)\s+\|")
    overflow_re = re.compile(r"\[\s*(\d+),\s*\.\.\.\)\s+(\d+)\s+\|")
    counts: dict[str, int] = {}
    sums: dict[str, int] = {}
    buckets: dict[str, list[tuple[int, int, int]]] = {}
    current_dev: str | None = None
    for line in text.splitlines():
        count_match = count_re.search(line)
        if count_match:
            counts[count_match.group(1)] = int(count_match.group(2))
            current_dev = None
            continue
        sum_match = sum_re.search(line)
        if sum_match:
            sums[sum_match.group(1)] = int(sum_match.group(2))
            current_dev = None
            continue
        header_match = header_re.search(line)
        if header_match:
            current_dev = header_match.group(1)
            buckets.setdefault(current_dev, [])
            continue
        if current_dev is None:
            continue
        bucket_match = bucket_re.search(line)
        if bucket_match:
            lo = int(bucket_match.group(1))
            hi = int(bucket_match.group(2))
            count = int(bucket_match.group(3))
            buckets[current_dev].append((lo, hi, count))
            continue
        overflow_match = overflow_re.search(line)
        if overflow_match:
            lo = int(overflow_match.group(1))
            count = int(overflow_match.group(2))
            buckets[current_dev].append((lo, lo, count))

    if not counts and not buckets:
        status = "unavailable_bpftrace_error" if "ERROR:" in text else "unavailable_no_events"
        return {**empty, "block_latency_status": status}

    def bucket_total(dev: str) -> int:
        return counts.get(dev) or sum(item[2] for item in buckets.get(dev, []))

    trace_dev = max(set(counts) | set(buckets), key=bucket_total)
    total = bucket_total(trace_dev)
    if total <= 0:
        return {**empty, "block_latency_status": "unavailable_no_events", "block_latency_trace_dev": trace_dev}

    def percentile_ms(pct: float) -> float | None:
        seen = 0
        target = max(1, int(np.ceil(total * pct)))
        for lo, hi, count in sorted(buckets.get(trace_dev, [])):
            seen += count
            if seen >= target:
                return round((hi if hi > lo else lo) / 1000.0, 6)
        return None

    mean_ms = None
    if trace_dev in sums and counts.get(trace_dev):
        mean_ms = round((sums[trace_dev] / counts[trace_dev]) / 1000.0, 6)
    return {
        "block_latency_status": "ok_bpftrace",
        "block_latency_p50_ms": percentile_ms(0.50),
        "block_latency_p95_ms": percentile_ms(0.95),
        "block_latency_p99_ms": percentile_ms(0.99),
        "block_latency_mean_ms": mean_ms,
        "block_latency_count": total,
        "block_latency_trace_dev": trace_dev,
    }


def run_with_monitor(
    command: list[str],
    cwd: Path,
    resolution: MonitorDeviceResolution,
    sample_interval_s: float,
    stdout_path: Path,
    stderr_path: Path,
    iostat_path: Path | None = None,
    block_latency_path: Path | None = None,
) -> dict[str, Any]:
    log("+ " + shell_join(command))
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    samples: list[dict[str, Any]] = []
    iostat_process, iostat_handle = start_iostat(resolution.effective_device, sample_interval_s, iostat_path)
    block_latency_process, block_latency_handle = start_block_latency_trace(resolution, block_latency_path)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while True:
            disk_sample = read_block_stats(resolution.effective_device) if resolution.effective_device is not None else None
            samples.append(
                {
                    "timestamp": time.monotonic(),
                    "cpu_ticks": sample_process_tree_cpu(process.pid),
                    "disk": disk_sample,
                    "device": resolution.effective_device,
                }
            )
            if process.poll() is not None:
                break
            time.sleep(sample_interval_s)
        return_code = process.wait()
    finally:
        stop_iostat(iostat_process, iostat_handle)
        stop_block_latency_trace(block_latency_process, block_latency_handle)
        stdout_file.close()
        stderr_file.close()

    summary = summarize_monitor_samples(samples, clock_ticks)
    summary = add_disk_status(summary, resolution, iostat_path, block_latency_path)
    summary.update(parse_block_latency_log(block_latency_path))
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
    if row.get("disk_metrics_status") not in (None, "", "ok"):
        return "inconclusive_monitor_invalid"
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

    if args.require_recall_at_least is not None:
        bad_rows = []
        for row in rows:
            recall = row.get("recall")
            if recall is None or float(recall) < float(args.require_recall_at_least):
                bad_rows.append(
                    {
                        "dataset": row.get("dataset"),
                        "bucket_name": row.get("bucket_name"),
                        "threads": row.get("threads"),
                        "recall": recall,
                    }
                )
        if bad_rows:
            raise RuntimeError(
                f"{len(bad_rows)} formal rows do not satisfy recall >= {args.require_recall_at_least}: "
                f"{bad_rows[:5]}"
            )

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


def base_bin_for_dataset(data_root: Path, experiment_root: Path, dataset: str) -> Path:
    if dataset == "yfcc10m":
        return default_yfcc_paths(data_root, experiment_root)["base_bin"]
    return default_small_paths(data_root, dataset)["base_bin"]


def run_hybrid_search_record(
    *,
    search_binary: Path,
    index_type: str,
    index_prefix: str,
    threads: int,
    beamwidth: int,
    query_bin: Path,
    truthset_bin: Path | str,
    k: int,
    similarity: str,
    nbr_type: str,
    selector_type: str,
    query_labels: Path,
    route: str,
    mem_l: int,
    l_value: int,
    output_jsonl: Path,
    log_path: Path,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    ensure_parent(output_jsonl)
    ensure_parent(log_path)
    if output_jsonl.exists():
        try:
            return first_search_record(output_jsonl)
        except Exception:
            output_jsonl.unlink()
    command = [
        str(search_binary),
        index_type,
        index_prefix,
        str(threads),
        str(beamwidth),
        str(query_bin),
        str(truthset_bin),
        str(k),
        similarity,
        nbr_type,
        selector_type,
        str(query_labels),
        route,
        "0",
        str(mem_l),
        str(l_value),
        "--jsonl-output",
        str(output_jsonl),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env_overrides:
        env.update(env_overrides)
    log("+ " + shell_join(command))
    attempts: list[subprocess.CompletedProcess[str]] = []
    max_attempts = 3
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, max_attempts + 1):
        if output_jsonl.exists():
            output_jsonl.unlink()
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout, env=env)
        attempts.append(result)
        if result.returncode == 0:
            break
        if attempt < max_attempts:
            log(f"[warn] search failed with code {result.returncode}; retrying attempt {attempt + 1}/{max_attempts}")
            time.sleep(2)
    assert result is not None
    with log_path.open("w", encoding="utf-8") as writer:
        writer.write("$ ")
        if env_overrides:
            writer.write(" ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())))
            writer.write(" ")
        writer.write(shell_join(command))
        writer.write("\n\n")
        for attempt, attempt_result in enumerate(attempts, start=1):
            if len(attempts) > 1:
                writer.write(f"\n[attempt {attempt}/{len(attempts)} exit_code={attempt_result.returncode}]\n")
            writer.write(attempt_result.stdout)
            writer.write(attempt_result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return first_search_record(output_jsonl)


def route_from_record(record: dict[str, Any]) -> str:
    query_count = int(record.get("query_count") or 0)
    if query_count > 0 and int(record.get("prefilter_count") or 0) == query_count:
        return "prefilter"
    if query_count > 0 and int(record.get("graph_count") or 0) == query_count:
        return "graph"
    raise RuntimeError(
        "route probe did not resolve to a single route: "
        f"prefilter={record.get('prefilter_count')} graph={record.get('graph_count')} query_count={query_count}"
    )


def ensure_recall_workload_and_truthset(
    *,
    dataset: str,
    manifest: dict[str, Any],
    bucket: dict[str, Any],
    data_root: Path,
    experiment_root: Path,
    query_count: int,
    seed: int,
    gt_threads: int,
    k: int,
    similarity: str,
    block_candidates: int,
    force_gt: bool,
    base_rows: np.memmap,
    sidecar: DenseBitsetSidecar,
    tags_by_id: np.ndarray,
) -> dict[str, Any]:
    bucket_name = str(bucket["name"])
    workload_dir_path = dataset_experiment_dir(experiment_root, dataset) / "recall_workloads" / bucket_name
    workload_dir_path.mkdir(parents=True, exist_ok=True)
    query_bin = workload_dir_path / "queries.bin"
    query_labels_path = workload_dir_path / "query_labels.spmat"
    truthset_path = workload_dir_path / "truthset.bin"
    metadata_path = workload_dir_path / "truthset_meta.json"

    source_query_bin = resolve_path(bucket["query_bin"])
    source_query_labels = resolve_path(bucket["query_labels"])
    query_total, _, query_rows = load_bin_matrix(source_query_bin, manifest["index_type"])
    query_labels = SpmatMatrix.load(source_query_labels)
    if query_total != query_labels.nrow:
        raise ValueError(f"query/query-label row mismatch for {dataset}/{bucket_name}")
    row_ids = sample_row_ids(query_total, query_count, seed, dataset, bucket_name)
    write_bin_subset(query_bin, query_rows, row_ids)
    write_spmat_subset(query_labels_path, query_labels, row_ids)

    sampled_labels = [query_labels.row_labels(row_id) for row_id in row_ids]
    filter_labels = sampled_labels[0]
    if any(labels != filter_labels for labels in sampled_labels[1:]):
        raise ValueError(f"recall workload for {dataset}/{bucket_name} must have one filter label set")

    candidate_index_ids = sidecar.materialize_candidates(manifest["selector_type"], filter_labels)
    candidate_ids = np.asarray(tags_by_id[candidate_index_ids], dtype=np.uint32)
    if candidate_ids.size < k:
        raise ValueError(f"{dataset}/{bucket_name} has only {candidate_ids.size} candidates, smaller than k={k}")

    expected_meta = {
        "format": "pipeann.qps4.recall_truthset.v1",
        "dataset": dataset,
        "bucket": bucket_name,
        "k": int(k),
        "similarity": similarity,
        "index_type": manifest["index_type"],
        "selector_type": manifest["selector_type"],
        "sample_seed": int(seed),
        "source_query_bin": str(source_query_bin),
        "source_query_labels": str(source_query_labels),
        "sampled_row_ids_hash": hashlib.sha256(json.dumps(row_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "base_hash": sha256_file(base_bin_for_dataset(data_root, experiment_root, dataset)),
        "query_hash": sha256_file(query_bin),
        "query_label_hash": sha256_file(query_labels_path),
        "candidate_count": int(candidate_ids.size),
        "query_count": int(len(row_ids)),
        "gt_threads": int(gt_threads),
    }
    reuse_gt = False
    if truthset_path.exists() and metadata_path.exists() and not force_gt:
        existing_meta = read_json(metadata_path)
        reuse_gt = all(existing_meta.get(key) == value for key, value in expected_meta.items())
    if not reuse_gt:
        _, _, sampled_rows = load_bin_matrix(query_bin, manifest["index_type"])
        log(f"[gt] compute {dataset}/{bucket_name}: queries={len(row_ids)} candidates={candidate_ids.size} threads={gt_threads}")
        exact_topk_ids = thread_limited_exact_topk(
            np.asarray(sampled_rows, dtype=np.float32),
            base_rows,
            candidate_ids,
            k=k,
            similarity=similarity,
            block_candidates=block_candidates,
            gt_threads=gt_threads,
        )
        write_truthset_ids(truthset_path, exact_topk_ids)
        write_json(metadata_path, expected_meta)
    else:
        log(f"[gt] reuse {dataset}/{bucket_name}: {truthset_path}")

    recall_bucket = {
        **bucket,
        "query_bin": str(query_bin),
        "query_labels": str(query_labels_path),
        "probe_query_bin": str(query_bin),
        "probe_query_labels": str(query_labels_path),
        "truthset_bin": str(truthset_path),
        "query_count": int(len(row_ids)),
        "recall_workload_source_query_bin": str(source_query_bin),
        "recall_workload_source_query_labels": str(source_query_labels),
        "recall_candidate_count": int(candidate_ids.size),
    }
    return {
        "bucket": recall_bucket,
        "filter_labels": filter_labels,
        "candidate_count": int(candidate_ids.size),
        "query_count": int(len(row_ids)),
        "truthset_bin": str(truthset_path),
        "query_bin": str(query_bin),
        "query_labels": str(query_labels_path),
    }


def calibrate_graph_l(
    *,
    search_binary: Path,
    manifest: dict[str, Any],
    bucket_info: dict[str, Any],
    threads: int,
    beamwidth: int,
    k: int,
    similarity: str,
    nbr_type: str,
    mem_l: int,
    target_recall: float,
    max_l: int,
    timeout: int,
    calib_dir: Path,
) -> dict[str, Any]:
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate(l_value: int) -> dict[str, Any]:
        l_value = max(int(k), int(l_value))
        if l_value not in evaluations:
            evaluations[l_value] = run_hybrid_search_record(
                search_binary=search_binary,
                index_type=manifest["index_type"],
                index_prefix=manifest["index_prefix"],
                threads=threads,
                beamwidth=beamwidth,
                query_bin=Path(bucket_info["query_bin"]),
                truthset_bin=bucket_info["truthset_bin"],
                k=k,
                similarity=similarity,
                nbr_type=nbr_type,
                selector_type=manifest["selector_type"],
                query_labels=Path(bucket_info["query_labels"]),
                route="graph",
                mem_l=mem_l,
                l_value=l_value,
                output_jsonl=calib_dir / f"graph_L{l_value}.jsonl",
                log_path=calib_dir / f"graph_L{l_value}.log",
                timeout=timeout,
            )
        return evaluations[l_value]

    low = k - 1
    high = None
    current = k
    while current <= max_l:
        record = evaluate(current)
        if float(record.get("recall") or 0.0) >= target_recall:
            high = current
            break
        low = current
        next_l = max(current + 1, int(math.ceil(current * 1.7)))
        current = max_l if next_l > max_l and current < max_l else next_l
    if high is None:
        return {
            "status": "unreachable_recall",
            "route": "graph",
            "chosen_L": None,
            "achieved_recall": max(float(row.get("recall") or 0.0) for row in evaluations.values()),
            "evaluations": sorted(evaluations.values(), key=lambda row: int(row["L"])),
        }
    while high > low + 1:
        mid = (low + high) // 2
        record = evaluate(mid)
        if float(record.get("recall") or 0.0) >= target_recall:
            high = mid
        else:
            low = mid
    chosen = evaluate(high)
    return {
        "status": "ok",
        "route": "graph",
        "chosen_L": int(high),
        "chosen_prefilter_rerank_l": None,
        "achieved_recall": float(chosen.get("recall") or 0.0),
        "achieved_qps": float(chosen.get("qps") or 0.0),
        "evaluations": sorted(evaluations.values(), key=lambda row: int(row["L"])),
    }


def calibrate_prefilter_rerank_for_recall(
    *,
    search_binary: Path,
    manifest: dict[str, Any],
    bucket_info: dict[str, Any],
    threads: int,
    beamwidth: int,
    k: int,
    similarity: str,
    nbr_type: str,
    mem_l: int,
    target_recall: float,
    timeout: int,
    calib_dir: Path,
) -> dict[str, Any]:
    candidate_count = int(bucket_info["candidate_count"])
    total_points = int(manifest.get("npoints") or 0)
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate(rerank_l: int) -> dict[str, Any]:
        rerank_l = min(candidate_count, max(int(k), int(rerank_l)))
        if rerank_l not in evaluations:
            evaluations[rerank_l] = run_hybrid_search_record(
                search_binary=search_binary,
                index_type=manifest["index_type"],
                index_prefix=manifest["index_prefix"],
                threads=threads,
                beamwidth=beamwidth,
                query_bin=Path(bucket_info["query_bin"]),
                truthset_bin=bucket_info["truthset_bin"],
                k=k,
                similarity=similarity,
                nbr_type=nbr_type,
                selector_type=manifest["selector_type"],
                query_labels=Path(bucket_info["query_labels"]),
                route="prefilter",
                mem_l=mem_l,
                l_value=100,
                output_jsonl=calib_dir / f"prefilter_rerank{rerank_l}.jsonl",
                log_path=calib_dir / f"prefilter_rerank{rerank_l}.log",
                timeout=timeout,
                env_overrides={PREFILTER_RERANK_ENV: str(rerank_l)},
            )
        return evaluations[rerank_l]

    high = min(candidate_count, default_prefilter_rerank_l(k, candidate_count, total_points))
    high_eval = evaluate(high)
    while float(high_eval.get("recall") or 0.0) < target_recall and high < candidate_count:
        high = min(candidate_count, max(high + 1, high * 2))
        high_eval = evaluate(high)
    if float(high_eval.get("recall") or 0.0) < target_recall:
        return {
            "status": "unreachable_recall",
            "route": "prefilter",
            "chosen_L": 100,
            "chosen_prefilter_rerank_l": None,
            "achieved_recall": max(float(row.get("recall") or 0.0) for row in evaluations.values()),
            "evaluations": sorted(evaluations.values(), key=lambda row: int(row.get("prefilter_rerank_l") or 0)),
        }
    low = k - 1
    while high > low + 1:
        mid = (low + high) // 2
        record = evaluate(mid)
        if float(record.get("recall") or 0.0) >= target_recall:
            high = mid
        else:
            low = mid
    chosen = evaluate(high)
    return {
        "status": "ok",
        "route": "prefilter",
        "chosen_L": 100,
        "chosen_prefilter_rerank_l": int(high),
        "achieved_recall": float(chosen.get("recall") or 0.0),
        "achieved_qps": float(chosen.get("qps") or 0.0),
        "evaluations": sorted(evaluations.values(), key=lambda row: int(row.get("prefilter_rerank_l") or 0)),
    }


def calibrate_recall_plan(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    experiment_root = resolve_path(args.experiment_root)
    build_dir = resolve_path(args.build_dir)
    search_binary = build_dir / "tests" / "search_disk_index_hybrid"
    if not search_binary.exists():
        raise FileNotFoundError(f"missing search binary: {search_binary}")
    gt_threads = normalize_gt_threads(args.gt_threads)
    datasets = args.dataset or ["fashion_mnist784", "gist960", "glove100", "yfcc10m"]
    summary_dir = experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined_entries: list[dict[str, Any]] = []
    manifests: dict[str, str] = {}

    for dataset in datasets:
        manifest_file = resolve_path(args.manifest) if args.manifest and len(datasets) == 1 else default_manifest_for_dataset(data_root, experiment_root, dataset)
        manifest = read_json(manifest_file)
        base_bin = base_bin_for_dataset(data_root, experiment_root, dataset)
        _, _, base_rows = load_bin_matrix(base_bin, manifest["index_type"])
        sidecar = DenseBitsetSidecar.load(Path(f"{manifest['index_prefix']}_labels.densebit"))
        tags_by_id = load_tags_by_id(Path(manifest["index_prefix"]), sidecar.npoints)
        recall_buckets: list[dict[str, Any]] = []
        dataset_entries: list[dict[str, Any]] = []
        bucket_filter = set(args.bucket or [])

        for bucket in manifest.get("buckets", []):
            bucket_name = str(bucket["name"])
            if bucket_filter and bucket_name not in bucket_filter and str(bucket.get("label")) not in bucket_filter:
                continue
            bucket_info = ensure_recall_workload_and_truthset(
                dataset=dataset,
                manifest=manifest,
                bucket=bucket,
                data_root=data_root,
                experiment_root=experiment_root,
                query_count=args.recall_query_count,
                seed=args.seed,
                gt_threads=gt_threads,
                k=args.k,
                similarity=args.similarity,
                block_candidates=args.block_candidates,
                force_gt=args.force_gt,
                base_rows=base_rows,
                sidecar=sidecar,
                tags_by_id=tags_by_id,
            )
            recall_buckets.append(bucket_info["bucket"])
            calib_dir = dataset_experiment_dir(experiment_root, dataset) / "recall_calibration" / bucket_name
            calib_dir.mkdir(parents=True, exist_ok=True)

            route_probe = run_hybrid_search_record(
                search_binary=search_binary,
                index_type=manifest["index_type"],
                index_prefix=manifest["index_prefix"],
                threads=args.calibration_threads,
                beamwidth=args.beamwidth,
                query_bin=Path(bucket_info["query_bin"]),
                truthset_bin=bucket_info["truthset_bin"],
                k=args.k,
                similarity=args.similarity,
                nbr_type=args.nbr_type,
                selector_type=manifest["selector_type"],
                query_labels=Path(bucket_info["query_labels"]),
                route="auto",
                mem_l=args.mem_l,
                l_value=args.auto_probe_l,
                output_jsonl=calib_dir / "route_probe.jsonl",
                log_path=calib_dir / "route_probe.log",
                timeout=args.timeout,
            )
            route = route_from_record(route_probe)
            if route == "graph":
                calibration = calibrate_graph_l(
                    search_binary=search_binary,
                    manifest=manifest,
                    bucket_info=bucket_info,
                    threads=args.calibration_threads,
                    beamwidth=args.beamwidth,
                    k=args.k,
                    similarity=args.similarity,
                    nbr_type=args.nbr_type,
                    mem_l=args.mem_l,
                    target_recall=args.target_recall,
                    max_l=args.max_l,
                    timeout=args.timeout,
                    calib_dir=calib_dir,
                )
            else:
                calibration = calibrate_prefilter_rerank_for_recall(
                    search_binary=search_binary,
                    manifest=manifest,
                    bucket_info=bucket_info,
                    threads=args.calibration_threads,
                    beamwidth=args.beamwidth,
                    k=args.k,
                    similarity=args.similarity,
                    nbr_type=args.nbr_type,
                    mem_l=args.mem_l,
                    target_recall=args.target_recall,
                    timeout=args.timeout,
                    calib_dir=calib_dir,
                )
            entry = {
                "dataset": dataset,
                "bucket": bucket_name,
                "label": bucket.get("label"),
                "selectivity_midpoint": bucket.get("midpoint"),
                "route": route,
                "L": int(calibration.get("chosen_L") or args.auto_probe_l),
                "chosen_L": calibration.get("chosen_L"),
                "prefilter_rerank_l": calibration.get("chosen_prefilter_rerank_l"),
                "chosen_prefilter_rerank_l": calibration.get("chosen_prefilter_rerank_l"),
                "target_recall": float(args.target_recall),
                "achieved_recall": calibration.get("achieved_recall"),
                "achieved_qps": calibration.get("achieved_qps"),
                "candidate_count": bucket_info["candidate_count"],
                "query_count": bucket_info["query_count"],
                "calibration_threads": int(args.calibration_threads),
                "gt_threads": int(gt_threads),
                "status": calibration["status"],
            }
            write_json(calib_dir / "calibration_summary.json", {**entry, "route_probe": route_probe, "calibration": calibration})
            dataset_entries.append(entry)
            combined_entries.append(entry)
            log(f"[calib] {dataset}/{bucket_name} route={route} status={entry['status']} L={entry['chosen_L']} rerank={entry['chosen_prefilter_rerank_l']} recall={entry['achieved_recall']}")
            if entry["status"] != "ok":
                raise RuntimeError(f"recall calibration failed for {dataset}/{bucket_name}: {entry['status']}")

        recall_manifest = {
            **manifest,
            "format": "pipeann.hybrid.selectivity_manifest.v1",
            "buckets": recall_buckets,
            "recall_workload": {
                "format": "pipeann.qps4.recall_workload.v1",
                "source_manifest": str(manifest_file),
                "target_recall": float(args.target_recall),
                "sample_query_count": int(args.recall_query_count),
                "seed": int(args.seed),
                "gt_threads": int(gt_threads),
            },
        }
        manifest_out = dataset_experiment_dir(experiment_root, dataset) / "recall_workloads" / "recall_manifest.json"
        write_json(manifest_out, recall_manifest)
        manifests[dataset] = str(manifest_out)
        write_json(
            dataset_experiment_dir(experiment_root, dataset) / "recall_calibration" / "bucket_plan_recall98.json",
            {
                "format": "pipeann.qps4.recall98_bucket_plan.v1",
                "dataset": dataset,
                "manifest": str(manifest_out),
                "defaults": {"target_recall": float(args.target_recall), "calibration_threads": int(args.calibration_threads)},
                "entries": dataset_entries,
            },
        )

    combined_plan = {
        "format": "pipeann.qps4.recall98_bucket_plan.v1",
        "target_recall": float(args.target_recall),
        "calibration_threads": int(args.calibration_threads),
        "gt_threads": int(gt_threads),
        "manifests": manifests,
        "defaults": {"target_recall": float(args.target_recall), "calibration_threads": int(args.calibration_threads)},
        "entries": combined_entries,
    }
    plan_path = summary_dir / "bucket_plan_recall98.json"
    write_json(plan_path, combined_plan)
    return {"plan": str(plan_path), "manifests": manifests, "entries": len(combined_entries)}


def run_thread_sweep(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    experiment_root = resolve_path(args.experiment_root)
    manifest_file = resolve_path(args.manifest) if args.manifest else default_manifest_for_dataset(data_root, experiment_root, args.dataset)
    manifest = read_json(manifest_file)
    run_root = resolve_path(args.out_dir) if args.out_dir else sweep_root(experiment_root, args.dataset)
    run_root.mkdir(parents=True, exist_ok=True)

    bucket_filter = set(args.bucket or [])
    plan_entries, plan_defaults = load_bucket_plan(resolve_path(args.bucket_plan_json) if args.bucket_plan_json else None, args.dataset)
    requested_device = args.ssd_device or infer_block_device(Path(manifest["index_prefix"]))
    if requested_device is None:
        log("[warn] could not infer block device; CPU monitoring will still run")
    else:
        log(f"[info] requested monitor device={requested_device}")
    resolution = resolve_monitor_device(requested_device, manifest["index_prefix"]) if not args.dry_run else MonitorDeviceResolution(requested_device, requested_device, "ok", "dry-run")
    if resolution.disk_metrics_status == "ok":
        log(f"[info] monitor device={resolution.effective_device} ({resolution.reason})")
    else:
        log(f"[warn] disk metrics unavailable: status={resolution.disk_metrics_status} reason={resolution.reason}")

    rows: list[dict[str, Any]] = []
    for bucket in manifest.get("buckets", []):
        bucket_name = str(bucket.get("name"))
        bucket_label = str(bucket.get("label"))
        if bucket_filter and bucket_name not in bucket_filter and bucket_label not in bucket_filter:
            continue

        route, l_value = resolve_bucket_settings(bucket, plan_entries, plan_defaults, args.default_route, args.default_l)
        plan_entry = plan_entries.get(bucket_name) or plan_entries.get(bucket_label) or {}
        bucket_root = run_root / bucket_name
        bucket_root.mkdir(parents=True, exist_ok=True)
        single_manifest = bucket_root / "manifest.json"
        write_single_bucket_manifest(manifest, bucket, single_manifest)
        prefilter_rerank_json = args.prefilter_rerank_json or args.bucket_plan_json

        for thread in args.threads:
            run_dir = bucket_root / f"t{thread}"
            result_path = run_dir / "results.jsonl"
            monitor_path = run_dir / "monitor.json"
            stdout_path = run_dir / "run.stdout.log"
            stderr_path = run_dir / "run.stderr.log"
            iostat_path = run_dir / "iostat.log" if args.capture_iostat else None
            block_latency_path = run_dir / "block_latency.log" if args.capture_block_latency else None

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
                        "chosen_L": plan_entry.get("chosen_L") or l_value,
                        "chosen_prefilter_rerank_l": plan_entry.get("chosen_prefilter_rerank_l")
                        or plan_entry.get("prefilter_rerank_l"),
                        "recall_target": plan_entry.get("target_recall"),
                        "recall_calibration_threads": plan_entry.get("calibration_threads"),
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
            if prefilter_rerank_json:
                command.extend(["--prefilter-rerank-json", str(resolve_path(prefilter_rerank_json))])

            if args.dry_run:
                log(f"[dry-run] {shell_join(command)}")
                continue

            monitor_summary = run_with_monitor(
                command,
                REPO_ROOT,
                resolution,
                args.sample_interval_s,
                stdout_path,
                stderr_path,
                iostat_path=iostat_path,
                block_latency_path=block_latency_path,
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
                "chosen_L": plan_entry.get("chosen_L") or l_value,
                "chosen_prefilter_rerank_l": plan_entry.get("chosen_prefilter_rerank_l")
                or plan_entry.get("prefilter_rerank_l"),
                "recall_target": plan_entry.get("target_recall"),
                "recall_calibration_threads": plan_entry.get("calibration_threads"),
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
        "requested_device": requested_device,
        "device": resolution.effective_device,
        "disk_metrics_status": resolution.disk_metrics_status,
        "disk_metrics_reason": resolution.reason,
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

    recall_parser = subparsers.add_parser(
        "calibrate-recall-plan",
        help="Build sampled recall workloads, exact filtered GT, and per-bucket recall@10>=98 route budgets.",
    )
    recall_parser.add_argument("--dataset", action="append", choices=sorted(ALL_DATASETS))
    recall_parser.add_argument("--bucket", action="append", help="Specific bucket name or label to calibrate. May be repeated.")
    recall_parser.add_argument("--manifest", help="Override manifest path. Only valid with one dataset.")
    recall_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    recall_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    recall_parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    recall_parser.add_argument("--recall-query-count", type=int, default=DEFAULT_RECALL_QUERY_COUNT)
    recall_parser.add_argument("--target-recall", type=float, default=DEFAULT_RECALL_TARGET)
    recall_parser.add_argument("--calibration-threads", type=int, default=DEFAULT_RECALL_CALIBRATION_THREADS)
    recall_parser.add_argument("--gt-threads", type=int, default=0, help="0 means all available CPU cores.")
    recall_parser.add_argument("--seed", type=int, default=20260515)
    recall_parser.add_argument("--auto-probe-l", type=int, default=100)
    recall_parser.add_argument("--max-l", type=int, default=DEFAULT_RECALL_MAX_L)
    recall_parser.add_argument("--beamwidth", type=int, default=4)
    recall_parser.add_argument("--k", type=int, default=10)
    recall_parser.add_argument("--similarity", default="l2")
    recall_parser.add_argument("--nbr-type", default="pq")
    recall_parser.add_argument("--mem-l", type=int, default=0)
    recall_parser.add_argument("--block-candidates", type=int, default=100_000)
    recall_parser.add_argument("--timeout", type=int, default=7200)
    recall_parser.add_argument("--force-gt", action="store_true")

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
    sweep_parser.add_argument("--capture-iostat", action="store_true", help="Write per-run iostat -x logs next to monitor.json.")
    sweep_parser.add_argument("--capture-block-latency", action="store_true", help="Write a per-run block latency status log; percentile capture is best-effort.")
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
    summarize_parser.add_argument("--require-recall-at-least", type=float)

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
    if args.command == "calibrate-recall-plan":
        payload = calibrate_recall_plan(args)
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
