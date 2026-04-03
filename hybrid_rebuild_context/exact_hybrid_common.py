#!/usr/bin/env python3
"""Shared helpers for the SIFT1M exact hybrid rebuild workflow."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

CONTEXT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CONTEXT_ROOT.parent
BUILD_DIR = CONTEXT_ROOT / "build"
BUILD_BIN_DIR = BUILD_DIR / "bin"
ASSETS_DIR = CONTEXT_ROOT / "assets"

DEFAULT_SELECTIVITIES = (
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
)
DEFAULT_K = 10
DEFAULT_COARSE_NQ = 100
DEFAULT_RSS_NQ = 1
DENSEBIT_MAGIC = 0x54494245534E4544
DENSEBIT_VERSION = 1


def format_sel(value: float) -> str:
    return f"{float(value):.3f}"


def parse_selectivities(values: str | Sequence[float] | None) -> list[float]:
    if values is None:
        return list(DEFAULT_SELECTIVITIES)
    if isinstance(values, str):
        tokens = [token.strip() for token in values.split(",") if token.strip()]
        return [float(token) for token in tokens]
    return [float(value) for value in values]


def canonical_selectivities(values: str | Sequence[float] | None = None) -> list[float]:
    parsed = parse_selectivities(values)
    return [float(format_sel(value)) for value in parsed]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def context_path(*parts: str) -> Path:
    return CONTEXT_ROOT.joinpath(*parts)


def assets_path(*parts: str) -> Path:
    return ASSETS_DIR.joinpath(*parts)


def labels_dir() -> Path:
    return assets_path("labels")


def sift1m_dir() -> Path:
    return assets_path("sift1m")


def index_dir() -> Path:
    return assets_path("index")


def cache_dir() -> Path:
    return assets_path("cache")


def artifacts_dir() -> Path:
    return context_path("artifacts")


def artifacts_results_v2_dir() -> Path:
    return artifacts_dir() / "results_v2"


def artifacts_results_final_dir() -> Path:
    return artifacts_dir() / "results_final"


def artifacts_images_dir() -> Path:
    return artifacts_dir() / "images"


def source_sift1m_candidates() -> tuple[Path, ...]:
    return (
        repo_path("data", "sift1m"),
        repo_path("data", "sift"),
    )


def data_labels_path() -> Path:
    return labels_dir() / "data_labels_exact.spmat"


def densebit_path(spmat_path: str | Path) -> Path:
    path = resolve_path(spmat_path)
    return Path(f"{path}.densebit")


def selectivity_map_path() -> Path:
    return labels_dir() / "selectivity_map.tsv"


def query_labels_path(sel: float) -> Path:
    return labels_dir() / f"query_labels_exact_sel{format_sel(sel)}.spmat"


def gt_path(sel: float) -> Path:
    return sift1m_dir() / f"gt_exact_sel{format_sel(sel)}.bin"


def cache_query_bin_path(nq: int) -> Path:
    return cache_dir() / f"sift_query_n{int(nq)}.bin"


def cache_query_labels_path(sel: float, nq: int) -> Path:
    return cache_dir() / f"query_labels_exact_sel{format_sel(sel)}_n{int(nq)}.spmat"


def cache_gt_path(sel: float, nq: int) -> Path:
    return cache_dir() / f"gt_exact_sel{format_sel(sel)}_n{int(nq)}.bin"


def hybrid_build_help() -> str:
    return (
        "Run `cmake -S hybrid_rebuild_context -B hybrid_rebuild_context/build` and then "
        "`cmake --build hybrid_rebuild_context/build -j $(nproc) --target "
        "hybrid_build_disk_index hybrid_search_disk_index_filtered "
        "hybrid_search_disk_index_filtered_prefilter`."
    )


def require_binary(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"missing binary: {path}\n{hybrid_build_help()}")


def graph_search_bin_path() -> Path:
    return require_binary(BUILD_BIN_DIR / "hybrid_search_disk_index_filtered")


def prefilter_search_bin_path() -> Path:
    return require_binary(BUILD_BIN_DIR / "hybrid_search_disk_index_filtered_prefilter")


def build_index_bin_path() -> Path:
    return require_binary(BUILD_BIN_DIR / "hybrid_build_disk_index")


def pq16_exact_prefix() -> Path:
    return index_dir() / "sift_1m_pq16_exact"


def pq32_exact_prefix() -> Path:
    return index_dir() / "sift_1m_exact"


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sync_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def sync_many(src: Path, targets: Sequence[Path]) -> None:
    for target in targets:
        sync_file(src, target)


def ensure_source_sift1m_assets(force: bool = False) -> dict[str, Path]:
    targets = {
        "base": sift1m_dir() / "sift_base.bin",
        "query": sift1m_dir() / "sift_query.bin",
        "groundtruth": sift1m_dir() / "sift_groundtruth.bin",
    }
    source_names = {
        "base": "sift_base.bin",
        "query": "sift_query.bin",
        "groundtruth": "sift_groundtruth.bin",
    }

    for candidate_dir in source_sift1m_candidates():
        candidate_files = {key: candidate_dir / filename for key, filename in source_names.items()}
        if not all(path.exists() for path in candidate_files.values()):
            continue
        for key, src in candidate_files.items():
            dst = targets[key]
            if force or not dst.exists():
                sync_file(src, dst)
        return targets

    missing = ", ".join(str(path) for path in [candidate_dir / name for candidate_dir in source_sift1m_candidates() for name in source_names.values()])
    raise FileNotFoundError(f"could not locate source sift1m assets. Checked: {missing}")


def write_spmat_csr(path: Path, nrow: int, ncol: int, indptr: np.ndarray, indices: np.ndarray,
                    data: np.ndarray | None = None) -> None:
    ensure_parent(path)
    if data is None:
        data = np.ones(indices.shape[0], dtype=np.float32)
    with path.open("wb") as handle:
        handle.write(struct.pack("<qqq", int(nrow), int(ncol), int(indices.shape[0])))
        indptr.astype(np.int64, copy=False).tofile(handle)
        indices.astype(np.int32, copy=False).tofile(handle)
        data.astype(np.float32, copy=False).tofile(handle)


def load_spmat(path: Path) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        nrow, ncol, nnz = struct.unpack("<qqq", handle.read(24))
        indptr = np.fromfile(handle, dtype=np.int64, count=nrow + 1)
        indices = np.fromfile(handle, dtype=np.int32, count=nnz)
        data = np.fromfile(handle, dtype=np.float32, count=nnz)
    return int(nrow), int(ncol), indptr, indices, data


def load_spmat_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        nrow, ncol, nnz = struct.unpack("<qqq", handle.read(24))
    return int(nrow), int(ncol), int(nnz)


def dense_words_per_label(npoints: int) -> int:
    return (int(npoints) + 63) // 64


def ensure_densebit_sidecar(spmat_path: str | Path, force: bool = False) -> Path:
    src = resolve_path(spmat_path)
    dst = densebit_path(src)
    nrow, ncol, nnz = load_spmat_header(src)
    words_per_label = dense_words_per_label(nrow)
    expected_size = struct.calcsize("<6Q") + ncol * words_per_label * 8

    def matches_existing() -> bool:
        if force or not dst.exists():
            return False
        if dst.stat().st_mtime_ns < src.stat().st_mtime_ns:
            return False
        if dst.stat().st_size != expected_size:
            return False
        with dst.open("rb") as handle:
            header = struct.unpack("<6Q", handle.read(struct.calcsize("<6Q")))
        return header == (
            DENSEBIT_MAGIC,
            DENSEBIT_VERSION,
            nrow,
            ncol,
            words_per_label,
            nnz,
        )

    if matches_existing():
        return dst

    total_rows, total_cols, indptr, indices, data = load_spmat(src)
    if total_rows != nrow or total_cols != ncol:
        raise ValueError(f"spmat metadata changed while building densebit sidecar: {src}")

    row_ids = np.repeat(np.arange(nrow, dtype=np.uint32), np.diff(indptr).astype(np.int64, copy=False))
    keep = data != 0
    row_ids = row_ids[keep]
    label_ids = indices[keep].astype(np.intp, copy=False)
    word_ids = (row_ids >> 6).astype(np.intp, copy=False)
    masks = np.left_shift(np.uint64(1), (row_ids & 63).astype(np.uint64, copy=False))

    payload = np.zeros((ncol, words_per_label), dtype=np.uint64)
    np.bitwise_or.at(payload, (label_ids, word_ids), masks)
    if nrow > 0 and words_per_label > 0 and (nrow % 64) != 0:
      payload[:, -1] &= np.uint64((1 << (nrow % 64)) - 1)

    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    ensure_parent(tmp_path)
    with tmp_path.open("wb") as handle:
        handle.write(struct.pack("<6Q", DENSEBIT_MAGIC, DENSEBIT_VERSION, nrow, ncol, words_per_label, int(keep.sum())))
        payload.tofile(handle)
    tmp_path.replace(dst)
    return dst


def write_single_label_query_spmat(path: Path, nrow: int, label_id: int, ncol: int) -> None:
    indptr = np.arange(nrow + 1, dtype=np.int64)
    indices = np.full(nrow, int(label_id), dtype=np.int32)
    data = np.ones(nrow, dtype=np.float32)
    write_spmat_csr(path, nrow, ncol, indptr, indices, data)


def count_labels_by_id(indices: np.ndarray, ncol: int) -> np.ndarray:
    return np.bincount(indices.astype(np.int64, copy=False), minlength=ncol)


def invert_spmat_by_label(path: Path) -> dict[int, np.ndarray]:
    nrow, ncol, indptr, indices, _ = load_spmat(path)
    row_ids = np.repeat(np.arange(nrow, dtype=np.int32), np.diff(indptr))
    inverted: dict[int, np.ndarray] = {}
    for label_id in range(ncol):
        inverted[label_id] = row_ids[indices == label_id]
    return inverted


def load_bin_vectors(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    with path.open("rb") as handle:
        npts, dim = struct.unpack("<ii", handle.read(8))
        data = np.fromfile(handle, dtype=dtype, count=npts * dim)
    return data.reshape(npts, dim)


def write_bin_vectors(path: Path, vectors: np.ndarray, dtype: np.dtype = np.float32) -> None:
    ensure_parent(path)
    array = np.asarray(vectors, dtype=dtype)
    npts, dim = array.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", int(npts), int(dim)))
        array.tofile(handle)


def load_truthset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        nq, k = struct.unpack("<ii", handle.read(8))
        ids = np.fromfile(handle, dtype=np.uint32, count=nq * k).reshape(nq, k)
        dists = np.fromfile(handle, dtype=np.float32, count=nq * k).reshape(nq, k)
    return ids, dists


def write_truthset(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    ensure_parent(path)
    ids_u32 = np.asarray(ids, dtype=np.uint32)
    dists_f32 = np.asarray(dists, dtype=np.float32)
    nq, k = ids_u32.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", int(nq), int(k)))
        ids_u32.tofile(handle)
        dists_f32.tofile(handle)


def truncate_spmat_file(src: Path, dst: Path, nrow: int) -> None:
    total_rows, ncol, indptr, indices, data = load_spmat(src)
    nrow = min(int(nrow), total_rows)
    nnz = int(indptr[nrow])
    out_indptr = indptr[: nrow + 1].copy()
    out_indices = indices[:nnz].copy()
    out_data = data[:nnz].copy()
    write_spmat_csr(dst, nrow, ncol, out_indptr, out_indices, out_data)


def truncate_truthset_file(src: Path, dst: Path, nq: int) -> None:
    ids, dists = load_truthset(src)
    nq = min(int(nq), ids.shape[0])
    write_truthset(dst, ids[:nq], dists[:nq])


def truncate_query_file(src: Path, dst: Path, nq: int) -> None:
    queries = load_bin_vectors(src)
    nq = min(int(nq), queries.shape[0])
    write_bin_vectors(dst, queries[:nq])


def load_selectivity_map(path: Path | None = None) -> list[dict[str, float]]:
    target = path or selectivity_map_path()
    rows: list[dict[str, float]] = []
    with target.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            label_id, count, selectivity = line.split("\t")
            rows.append({
                "label_id": int(label_id),
                "count": int(count),
                "selectivity": float(selectivity),
            })
    return rows


def find_label_id_for_selectivity(target_sel: float, rows: Iterable[dict[str, float]]) -> int:
    needle = format_sel(target_sel)
    for row in rows:
        if format_sel(float(row["selectivity"])) == needle:
            return int(row["label_id"])
    raise KeyError(f"selectivity {needle} missing from selectivity map")


def ensure_query_subset_cache(selectivities: Sequence[float], coarse_nq: int = DEFAULT_COARSE_NQ,
                              rss_nq: int = DEFAULT_RSS_NQ, force: bool = False) -> None:
    ensure_source_sift1m_assets(force=False)
    cache_dir().mkdir(parents=True, exist_ok=True)
    full_query = sift1m_dir() / "sift_query.bin"
    subset_sizes = sorted({int(coarse_nq), int(rss_nq)})

    for nq in subset_sizes:
        dst_query = cache_query_bin_path(nq)
        if force or not dst_query.exists():
            truncate_query_file(full_query, dst_query, nq)

    for sel in selectivities:
        full_qlabel = query_labels_path(sel)
        if not full_qlabel.exists():
            continue

        for nq in subset_sizes:
            dst_qlabel = cache_query_labels_path(sel, nq)
            if force or not dst_qlabel.exists():
                truncate_spmat_file(full_qlabel, dst_qlabel, nq)

        full_gt = gt_path(sel)
        if full_gt.exists():
            dst_gt = cache_gt_path(sel, coarse_nq)
            if force or not dst_gt.exists():
                truncate_truthset_file(full_gt, dst_gt, coarse_nq)


def parse_search_output(output: str) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    metrics: list[dict[str, float]] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("METRICS_JSON "):
            try:
                metrics.append(json.loads(line.split(" ", 1)[1]))
            except json.JSONDecodeError:
                continue
            continue

        tokens = line.split()
        if len(tokens) < 7 or not re.fullmatch(r"\d+", tokens[0]):
            continue
        try:
            row = {
                "L": int(tokens[0]),
                "beamwidth": int(tokens[1]),
                "qps": float(tokens[2]),
                "latency_us": float(tokens[3]),
                "p99_latency_us": float(tokens[4]),
                "mean_hops": float(tokens[5]),
                "mean_ios": float(tokens[6]),
            }
            if len(tokens) >= 8:
                row["recall"] = float(tokens[7])
            rows.append(row)
        except ValueError:
            continue

    return rows, metrics


def extract_rss_mb(output: str) -> float | None:
    _, metrics = parse_search_output(output)
    if metrics and "rss_delta_mb" in metrics[-1]:
        return float(metrics[-1]["rss_delta_mb"])

    match = re.search(r"RSS Delta \(Query Memory\):\s+([\d.]+)\s+MB", output)
    if match:
        return float(match.group(1))
    return None


def extract_process_peak_rss_mb(output: str) -> float | None:
    _, metrics = parse_search_output(output)
    if metrics and "process_peak_rss_mb" in metrics[-1]:
        return float(metrics[-1]["process_peak_rss_mb"])

    match = re.search(r"Process Peak RSS:\s+([\d.]+)\s+MB", output)
    if match:
        return float(match.group(1))
    return None


def extract_time_peak_rss_mb(output: str) -> float | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", output)
    if match:
        return float(match.group(1)) / 1024.0
    return None


def run_command(cmd: Sequence[str], timeout: int = 600,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env.setdefault("PIPEANN_PQ_MMAP", "1")
    if env:
        merged_env.update(env)
    return subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=merged_env,
    )


def build_search_command(index_prefix: str | Path, query_bin: str | Path, truthset_bin: str | Path,
                         query_label_file: str | Path, prefilter_threshold: float, l_values: Sequence[int],
                         threads: int = 1, beamwidth: int = 4, data_labels: str | Path | None = None) -> list[str]:
    resolved_data_labels = resolve_path(data_labels or data_labels_path())
    resolved_truthset = str(resolve_path(truthset_bin)) if str(truthset_bin) != "null" else "null"
    resolved_index_prefix = str(resolve_path(index_prefix))
    resolved_query_bin = str(resolve_path(query_bin))
    resolved_query_label = str(resolve_path(query_label_file))
    use_prefilter = float(prefilter_threshold) > 0.0

    if use_prefilter:
        ensure_densebit_sidecar(resolved_data_labels)
        cmd = [
            str(prefilter_search_bin_path()),
            "float",
            resolved_index_prefix,
            str(int(threads)),
            str(int(beamwidth)),
            resolved_query_bin,
            resolved_truthset,
            str(DEFAULT_K),
            "l2",
            "pq",
            "intersect",
            resolved_query_label,
            str(resolved_data_labels),
        ]
    else:
        cmd = [
            str(graph_search_bin_path()),
            "float",
            resolved_index_prefix,
            str(int(threads)),
            str(int(beamwidth)),
            resolved_query_bin,
            resolved_truthset,
            str(DEFAULT_K),
            "l2",
            "pq",
            "intersect",
            resolved_query_label,
            "0",
            "0",
        ]
    cmd.extend(str(int(value)) for value in l_values)
    return cmd


def run_search_binary(index_prefix: str | Path, query_bin: str | Path, truthset_bin: str | Path,
                      query_label_file: str | Path, prefilter_threshold: float, l_values: Sequence[int],
                      threads: int = 1, beamwidth: int = 4, timeout: int = 600,
                      env: dict[str, str] | None = None) -> tuple[list[dict[str, float]], list[dict[str, float]], str, int]:
    cmd = build_search_command(
        index_prefix=index_prefix,
        query_bin=query_bin,
        truthset_bin=truthset_bin,
        query_label_file=query_label_file,
        prefilter_threshold=prefilter_threshold,
        l_values=l_values,
        threads=threads,
        beamwidth=beamwidth,
    )
    result = run_command(cmd, timeout=timeout, env=env)
    output = result.stdout + result.stderr
    rows, metrics = parse_search_output(output)
    return rows, metrics, output, result.returncode


def run_single_query_peak_rss(index_prefix: str | Path, query_bin: str | Path, query_label_file: str | Path,
                              prefilter_threshold: float, l_value: int, threads: int = 1, beamwidth: int = 4,
                              timeout: int = 600, env: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = build_search_command(
        index_prefix=index_prefix,
        query_bin=query_bin,
        truthset_bin="null",
        query_label_file=query_label_file,
        prefilter_threshold=prefilter_threshold,
        l_values=[l_value],
        threads=threads,
        beamwidth=beamwidth,
    )
    result = run_command(["/usr/bin/time", "-v", *cmd], timeout=timeout, env=env)
    output = result.stdout + result.stderr
    time_peak_rss_mb = extract_time_peak_rss_mb(output)
    return {
        "peak_rss_mb": time_peak_rss_mb,
        "process_peak_rss_mb": time_peak_rss_mb,
        "rss_before_mb": None,
        "rss_after_mb": None,
        "rss_delta_mb": None,
        "warning": None,
        "metrics": {},
        "output": output,
        "returncode": result.returncode,
    }
