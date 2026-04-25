#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_BUCKET_SPECS = (
    "s0:0,1e-5",
    "s1:1e-5,1e-4",
    "s2:1e-4,1e-3",
    "s3:1e-3,1e-2",
    "s4:1e-2,1e-1",
    "s5:1e-1,1.01",
)
DEFAULT_SINGLE_LABEL_TARGETS = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    1e-1,
)
DEFAULT_SYNTHETIC_HIGH_SELECTIVITY_SPECS = (
    ("s50", 0.50, 0),
    ("s75", 0.75, 1),
    ("s100", 1.00, 2),
)
DEFAULT_UNIFORM_EXACT_SELECTIVITY_SPECS = (
    ("u1e-05", 1e-5),
    ("u3e-05", 3e-5),
    ("u1e-04", 1e-4),
    ("u3e-04", 3e-4),
    ("u1e-03", 1e-3),
    ("u3e-03", 3e-3),
    ("u1e-02", 1e-2),
    ("u1e-01", 1e-1),
    ("u50", 0.50),
    ("u75", 0.75),
    ("u100", 1.00),
)
DEFAULT_EXTRA_REAL_HIGH_SELECTIVITY_LABELS = (8, 89, 29, 23)
DEFAULT_SYNTHETIC_QUERY_COUNT = 10_000
DEFAULT_SYNTHETIC_RANDOM_SEED = 20260424
DEFAULT_CALIBRATION_QUERY_COUNT = 200
DEFAULT_CALIBRATION_BLOCK_CANDIDATES = 16_384
DEFAULT_CALIBRATION_MAX_SELECTIVITY = 0.1
INDEX_PREFIX_CLONE_SUFFIXES = (
    "_disk.index",
    "_disk.index.tags",
    "_pq_compressed.bin",
    "_pq_pivots.bin",
    "_partition.bin.aligned",
    "_mem.index.tags",
)
BIN_HEADER = struct.Struct("<ii")
SPMAT_HEADER = struct.Struct("<qqq")
DENSEBIT_HEADER = struct.Struct("<QQQQQQ")
UINT64_ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
PREFILTER_RERANK_ENV = "PIPEANN_PREFILTER_RERANK_L"


@dataclass(frozen=True)
class BucketSpec:
    name: str
    lower: float
    upper: float

    def matches(self, value: float) -> bool:
        if value < self.lower:
            return False
        if math.isclose(value, self.upper):
            return True
        return value < self.upper

    @property
    def label(self) -> str:
        return f"[{self.lower:.1e}, {self.upper:.1e})"

    @property
    def midpoint(self) -> float:
        if self.lower <= 0:
            return self.upper / 2.0
        return math.sqrt(self.lower * self.upper)


@dataclass(frozen=True)
class SingleLabelTarget:
    selectivity: float

    @property
    def name(self) -> str:
        return f"t{self.selectivity:.0e}"


@dataclass
class SingleLabelStat:
    label_id: int
    candidate_count: int
    selectivity: float
    query_count: int


@dataclass(frozen=True)
class SyntheticSelectivitySpec:
    name: str
    selectivity: float
    label_id: int

    @property
    def label(self) -> str:
        if math.isclose(self.selectivity, 1.0) or self.selectivity >= 0.1:
            return f"{self.selectivity:.0%}"
        if self.selectivity >= 0.01:
            return f"{self.selectivity:.1%}"
        return f"{self.selectivity:.1e}"


@dataclass
class SpmatMatrix:
    path: Path
    nrow: int
    ncol: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SpmatMatrix":
        with path.open("rb") as reader:
            header = reader.read(SPMAT_HEADER.size)
            if len(header) != SPMAT_HEADER.size:
                raise ValueError(f"invalid spmat header: {path}")
            nrow, ncol, nnz = SPMAT_HEADER.unpack(header)
            indptr = np.fromfile(reader, dtype=np.int64, count=nrow + 1)
            indices = np.fromfile(reader, dtype=np.int32, count=nnz)
            data = np.fromfile(reader, dtype=np.float32, count=nnz)
        if indptr.size != nrow + 1 or indices.size != nnz or data.size != nnz:
            raise ValueError(f"incomplete spmat payload: {path}")
        return cls(path=path, nrow=int(nrow), ncol=int(ncol), indptr=indptr, indices=indices, data=data)

    def row_labels(self, row_id: int) -> list[int]:
        start = int(self.indptr[row_id])
        end = int(self.indptr[row_id + 1])
        if start == end:
            return []
        row_indices = self.indices[start:end]
        row_values = self.data[start:end]
        return [int(label) for label, value in zip(row_indices.tolist(), row_values.tolist()) if value != 0.0]


@dataclass
class DenseBitsetSidecar:
    path: Path
    npoints: int
    nlabels: int
    words_per_label: int
    nnz: int
    words: np.memmap

    @classmethod
    def load(cls, path: Path) -> "DenseBitsetSidecar":
        with path.open("rb") as reader:
            header = reader.read(DENSEBIT_HEADER.size)
            if len(header) != DENSEBIT_HEADER.size:
                raise ValueError(f"invalid densebit header: {path}")
            magic, version, npoints, nlabels, words_per_label, nnz = DENSEBIT_HEADER.unpack(header)
        if version != 1:
            raise ValueError(f"unsupported densebit version {version}: {path}")
        words = np.memmap(
            path,
            mode="r",
            dtype=np.uint64,
            offset=DENSEBIT_HEADER.size,
            shape=(int(nlabels), int(words_per_label)),
        )
        return cls(
            path=path,
            npoints=int(npoints),
            nlabels=int(nlabels),
            words_per_label=int(words_per_label),
            nnz=int(nnz),
            words=words,
        )

    @property
    def tail_mask(self) -> np.uint64:
        remainder = self.npoints % 64
        if remainder == 0:
            return UINT64_ALL_ONES
        return np.uint64((1 << remainder) - 1)

    def single_label_candidate_count(self, label: int) -> int:
        if label < 0 or label >= self.nlabels:
            raise ValueError(f"label {label} out of range for densebit sidecar")
        scratch = np.array(self.words[label], copy=True)
        if scratch.size > 0:
            scratch[-1] &= self.tail_mask
        return popcount_u64(scratch)

    def count_candidates(self, selector_type: str, labels: Sequence[int]) -> int:
        normalized = sorted(set(int(label) for label in labels))
        if selector_type == "subset":
            if not normalized:
                return self.npoints
            if any(label < 0 or label >= self.nlabels for label in normalized):
                return 0
            scratch = np.array(self.words[normalized[0]], copy=True)
            for label in normalized[1:]:
                np.bitwise_and(scratch, self.words[label], out=scratch)
        elif selector_type == "intersect":
            if not normalized:
                return 0
            scratch = np.zeros(self.words_per_label, dtype=np.uint64)
            for label in normalized:
                if label < 0 or label >= self.nlabels:
                    continue
                np.bitwise_or(scratch, self.words[label], out=scratch)
        else:
            raise ValueError(f"unsupported selector_type: {selector_type}")

        if scratch.size > 0:
            scratch[-1] &= self.tail_mask
        return popcount_u64(scratch)


def popcount_u64(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    scratch = np.array(values, copy=True, dtype=np.uint64)
    scratch -= (scratch >> np.uint64(1)) & np.uint64(0x5555555555555555)
    scratch = (scratch & np.uint64(0x3333333333333333)) + ((scratch >> np.uint64(2)) & np.uint64(0x3333333333333333))
    scratch = (scratch + (scratch >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((((scratch * np.uint64(0x0101010101010101)) >> np.uint64(56))).sum(dtype=np.uint64))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_output_path(path: Path, description: str) -> Path:
    normalized = path.resolve(strict=False)
    build_root = DEFAULT_BUILD_DIR.resolve(strict=False)
    if normalized == build_root or build_root in normalized.parents:
        experiments_root = DEFAULT_EXPERIMENTS_DIR.relative_to(REPO_ROOT)
        raise ValueError(
            f"{description} must not be written under {build_root}; "
            f"use {experiments_root} or another non-build directory instead: {normalized}"
        )
    return path


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def replace_file_with_link_or_copy(source_path: Path, dest_path: Path) -> None:
    ensure_parent(dest_path)
    if dest_path.exists() or dest_path.is_symlink():
        dest_path.unlink()
    try:
        os.link(source_path, dest_path)
    except OSError:
        shutil.copy2(source_path, dest_path)


def read_spmat_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as reader:
        header = reader.read(SPMAT_HEADER.size)
        if len(header) != SPMAT_HEADER.size:
            raise ValueError(f"invalid spmat header: {path}")
        nrow, ncol, nnz = SPMAT_HEADER.unpack(header)
    return int(nrow), int(ncol), int(nnz)


def find_binary(build_dir: Path, name: str) -> Path:
    candidates = sorted(
        path for path in build_dir.rglob(name)
        if path.is_file() and os.access(path, os.X_OK)
    )
    if not candidates:
        raise FileNotFoundError(
            f"missing built binary '{name}' under {build_dir}. "
            f"Run cmake --build {build_dir} --target {name}."
        )
    return candidates[0]


def parse_bucket_spec(spec: str) -> BucketSpec:
    try:
        name, bounds = spec.split(":", 1)
        lower_str, upper_str = bounds.split(",", 1)
        lower = float(lower_str)
        upper = float(upper_str)
    except ValueError as exc:
        raise ValueError(f"invalid bucket spec '{spec}', use name:lower,upper") from exc
    if lower < 0 or upper <= lower:
        raise ValueError(f"invalid bucket bounds '{spec}'")
    return BucketSpec(name=name, lower=lower, upper=upper)


def load_bin_matrix(path: Path, dtype_name: str) -> tuple[int, int, np.memmap]:
    dtype_map = {
        "float": np.float32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    with path.open("rb") as reader:
        header = reader.read(BIN_HEADER.size)
        if len(header) != BIN_HEADER.size:
            raise ValueError(f"invalid bin header: {path}")
        npts, dim = BIN_HEADER.unpack(header)
    data = np.memmap(path, mode="r", dtype=dtype_map[dtype_name], offset=BIN_HEADER.size, shape=(npts, dim))
    return int(npts), int(dim), data


def write_bin_subset(path: Path, rows: np.memmap, row_ids: Sequence[int]) -> None:
    ensure_parent(path)
    subset = np.asarray(rows[list(row_ids)])
    with path.open("wb") as writer:
        writer.write(BIN_HEADER.pack(subset.shape[0], subset.shape[1]))
        subset.tofile(writer)


def write_spmat_subset(path: Path, matrix: SpmatMatrix, row_ids: Sequence[int]) -> None:
    ensure_parent(path)
    selected_indices: list[np.ndarray] = []
    selected_values: list[np.ndarray] = []
    indptr = [0]
    nnz = 0
    for row_id in row_ids:
        start = int(matrix.indptr[row_id])
        end = int(matrix.indptr[row_id + 1])
        row_indices = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_indices.size:
          mask = row_values != 0.0
          row_indices = row_indices[mask]
        else:
          mask = row_values != 0.0
        row_values = np.ones(row_indices.shape[0], dtype=np.float32)
        selected_indices.append(np.asarray(row_indices, dtype=np.int32))
        selected_values.append(row_values)
        nnz += int(row_indices.shape[0])
        indptr.append(nnz)

    with path.open("wb") as writer:
        writer.write(SPMAT_HEADER.pack(len(row_ids), matrix.ncol, nnz))
        np.asarray(indptr, dtype=np.int64).tofile(writer)
        if nnz > 0:
            np.concatenate(selected_indices).astype(np.int32, copy=False).tofile(writer)
            np.concatenate(selected_values).astype(np.float32, copy=False).tofile(writer)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(record, sort_keys=True))
        writer.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def build_process_env(env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PIPEANN_PQ_MMAP"] = os.environ.get("PIPEANN_PQ_MMAP", "1")
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})
    return env


def format_logged_command(cmd: Sequence[str], env_overrides: dict[str, str] | None = None) -> str:
    env_prefix = ""
    if env_overrides:
        env_prefix = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in sorted(env_overrides.items())
        ) + " "
    return env_prefix + " ".join(shlex.quote(part) for part in cmd)


def run_command(
    cmd: Sequence[str],
    timeout: int,
    log_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=build_process_env(env_overrides),
    )
    if log_path is not None:
        ensure_parent(log_path)
        with log_path.open("w", encoding="utf-8") as writer:
            writer.write("$ ")
            writer.write(format_logged_command(cmd, env_overrides))
            writer.write("\n\n")
            writer.write(result.stdout)
            writer.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with code {result.returncode}: {' '.join(cmd)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def run_command_with_time(
    cmd: Sequence[str],
    timeout: int,
    log_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> int:
    time_binary = Path("/usr/bin/time")
    if not time_binary.exists():
        raise FileNotFoundError("/usr/bin/time is required for peak memory measurement")
    timed_cmd = [str(time_binary), "-v", *cmd]
    result = subprocess.run(
        timed_cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=build_process_env(env_overrides),
    )
    if log_path is not None:
        ensure_parent(log_path)
        with log_path.open("w", encoding="utf-8") as writer:
            writer.write("$ ")
            writer.write(format_logged_command(timed_cmd, env_overrides))
            writer.write("\n\n")
            writer.write(result.stdout)
            writer.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"timed command failed with code {result.returncode}: {' '.join(cmd)}\n"
            f"{result.stdout}{result.stderr}"
        )

    max_rss_kb = None
    for line in result.stderr.splitlines():
        if "Maximum resident set size" not in line:
            continue
        _, value = line.split(":", 1)
        max_rss_kb = int(value.strip())
        break
    if max_rss_kb is None:
        raise RuntimeError(f"failed to parse peak memory from /usr/bin/time output for: {' '.join(cmd)}")
    return max_rss_kb


def load_search_records(path: Path) -> list[dict[str, Any]]:
    return [record for record in read_jsonl(path) if record.get("format") == "pipeann.hybrid.search.v1"]


def load_workload_summary_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary["format"] == "pipeann.hybrid.random_single_label_workloads.v1":
        return list(summary["real_workloads"])
    return list(summary["workloads"])


def write_truthset_ids(path: Path, ids: np.ndarray) -> None:
    if ids.ndim != 2:
        raise ValueError(f"truthset ids must be rank-2, got shape {ids.shape}")
    ensure_parent(path)
    normalized = np.asarray(ids, dtype=np.uint32)
    with path.open("wb") as writer:
        writer.write(BIN_HEADER.pack(normalized.shape[0], normalized.shape[1]))
        normalized.tofile(writer)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def compute_exact_topk_ids(
    query_rows: np.ndarray,
    base_rows: np.memmap,
    candidate_ids: np.ndarray,
    *,
    k: int,
    similarity: str,
    block_candidates: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if block_candidates <= 0:
        raise ValueError(f"block_candidates must be positive, got {block_candidates}")
    if candidate_ids.size < k:
        raise ValueError(f"candidate count {candidate_ids.size} is smaller than k={k}")

    queries = np.asarray(query_rows, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError(f"query_rows must be rank-2, got shape {queries.shape}")

    if similarity == "cosine":
        queries = normalize_rows(queries)
    elif similarity not in {"l2", "mips"}:
        raise ValueError(f"unsupported similarity for calibration: {similarity}")

    query_count = queries.shape[0]
    best_scores = np.full((query_count, k), np.inf, dtype=np.float32)
    best_ids = np.full((query_count, k), np.iinfo(np.uint32).max, dtype=np.uint32)
    row_index = np.arange(query_count, dtype=np.int64)[:, None]
    query_norms = None
    if similarity == "l2":
        query_norms = np.sum(queries * queries, axis=1, dtype=np.float32)

    normalized_candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    for start in range(0, normalized_candidate_ids.shape[0], block_candidates):
        end = min(start + block_candidates, normalized_candidate_ids.shape[0])
        block_ids = normalized_candidate_ids[start:end]
        block = np.asarray(base_rows[block_ids], dtype=np.float32)
        if similarity == "cosine":
            block = normalize_rows(block)
            block_scores = 1.0 - (queries @ block.T)
        elif similarity == "mips":
            block_scores = -(queries @ block.T)
        else:
            block_norms = np.sum(block * block, axis=1, dtype=np.float32)
            block_scores = query_norms[:, None] + block_norms[None, :] - (2.0 * (queries @ block.T))

        block_id_matrix = np.broadcast_to(block_ids.astype(np.uint32)[None, :], (query_count, block_ids.shape[0]))
        combined_scores = np.concatenate((best_scores, block_scores), axis=1)
        combined_ids = np.concatenate((best_ids, block_id_matrix), axis=1)
        keep = np.argpartition(combined_scores, kth=k - 1, axis=1)[:, :k]
        best_scores = combined_scores[row_index, keep]
        best_ids = combined_ids[row_index, keep]
        order = np.argsort(best_scores, axis=1, kind="stable")
        best_scores = best_scores[row_index, order]
        best_ids = best_ids[row_index, order]

    return best_ids


def load_prefilter_rerank_overrides(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = json.loads(require_file(path, "prefilter rerank json").read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") == "pipeann.hybrid.prefilter_rerank_calibration.v1":
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"invalid overrides payload in {path}")
        return {str(bucket_name): int(value) for bucket_name, value in overrides.items()}
    if isinstance(payload, dict):
        return {str(bucket_name): int(value) for bucket_name, value in payload.items()}
    raise ValueError(f"unsupported prefilter rerank json format: {path}")


def default_prefilter_rerank_l(k_search: int, candidate_count: int, total_points: int) -> int:
    if candidate_count <= 0:
        return 0
    selectivity = 1.0 if total_points <= 0 else candidate_count / total_points
    target = 192
    if selectivity <= 0.005:
        target = 96
    elif selectivity <= 0.02:
        target = 128
    elif selectivity <= 0.1:
        target = 160
    target = max(int(k_search), target)
    return min(target, int(candidate_count))


def clone_index_prefix_files(source_prefix: Path, dest_prefix: Path) -> list[Path]:
    copied_paths: list[Path] = []
    for suffix in INDEX_PREFIX_CLONE_SUFFIXES:
        source_path = Path(f"{source_prefix}{suffix}")
        dest_path = Path(f"{dest_prefix}{suffix}")
        if not source_path.exists():
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            continue
        replace_file_with_link_or_copy(source_path, dest_path)
        copied_paths.append(dest_path)
    return copied_paths


def write_densebit_sidecar_from_spmat(label_path: Path, sidecar_path: Path, expected_npoints: int) -> Path:
    matrix = SpmatMatrix.load(label_path)
    if matrix.nrow != expected_npoints:
        raise ValueError(
            f"label row count mismatch: {label_path} has {matrix.nrow}, expected {expected_npoints}"
        )

    nlabels = int(matrix.ncol)
    words_per_label = (expected_npoints + 63) // 64
    payload = np.zeros((nlabels, words_per_label), dtype=np.uint64)
    kept_nnz = 0
    for row_id in range(matrix.nrow):
        start = int(matrix.indptr[row_id])
        end = int(matrix.indptr[row_id + 1])
        if start == end:
            continue
        row_indices = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        valid_mask = row_values != 0.0
        if not np.any(valid_mask):
            continue
        for label_id in row_indices[valid_mask].tolist():
            label_index = int(label_id)
            if label_index < 0 or label_index >= nlabels:
                raise ValueError(f"label id {label_index} out of range for {label_path}")
            word_index = row_id // 64
            bit_index = row_id % 64
            payload[label_index, word_index] |= np.uint64(1) << np.uint64(bit_index)
            kept_nnz += 1

    ensure_parent(sidecar_path)
    with sidecar_path.open("wb") as writer:
        writer.write(
            DENSEBIT_HEADER.pack(
                0x54494245534E4544,
                1,
                expected_npoints,
                nlabels,
                words_per_label,
                kept_nnz,
            )
        )
        payload.tofile(writer)
    return sidecar_path


def create_index_prefix_for_labels(args: argparse.Namespace) -> Path:
    source_prefix = resolve_path(args.source_prefix)
    dest_prefix = resolve_path(args.dest_prefix)
    label_path = require_file(resolve_path(args.label_file), "label file")

    disk_index_path = Path(f"{source_prefix}_disk.index")
    require_file(disk_index_path, "source disk index")
    npoints, _, _ = read_spmat_header(label_path)

    copied_paths = clone_index_prefix_files(source_prefix, dest_prefix)
    sidecar_path = write_densebit_sidecar_from_spmat(label_path, Path(f"{dest_prefix}_labels.densebit"), npoints)
    meta_path = Path(f"{dest_prefix}_hybrid.meta")
    if meta_path.exists() or meta_path.is_symlink():
        meta_path.unlink()

    summary_path = resolve_path(args.summary_json) if args.summary_json else dest_prefix.parent / f"{dest_prefix.name}_label_runtime.json"
    summary = {
        "format": "pipeann.hybrid.label_runtime.v1",
        "source_prefix": str(source_prefix),
        "dest_prefix": str(dest_prefix),
        "label_file": str(label_path),
        "copied_files": [str(path) for path in copied_paths],
        "sidecar_path": str(sidecar_path),
        "hybrid_meta_path": str(meta_path),
        "npoints": npoints,
    }
    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] cloned index prefix from {source_prefix} to {dest_prefix}")
    print(f"[ok] wrote densebit sidecar to {sidecar_path}")
    print(f"[ok] removed stale hybrid metadata at {meta_path}")
    print(f"[ok] wrote runtime summary to {summary_path}")
    return summary_path


def load_workload_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        summary = json.load(reader)
    supported_formats = {
        "pipeann.hybrid.random_single_label_workloads.v1",
        "pipeann.hybrid.synthetic_high_selectivity.v1",
        "pipeann.hybrid.uniform_exact_selectivity.v1",
    }
    if summary.get("format") not in supported_formats:
        raise ValueError(f"unsupported workload summary format: {path}")
    return summary


def build_manifest_from_workload_summary(args: argparse.Namespace) -> Path:
    summary_path = require_file(resolve_path(args.summary_json), "workload summary")
    summary = load_workload_summary(summary_path)
    manifest_path = validate_output_path(resolve_path(args.manifest), "manifest output") if args.manifest else summary_path.with_name(f"{summary_path.stem}_manifest.json")
    manifest_path = validate_output_path(manifest_path, "manifest output")
    index_prefix = resolve_path(args.index_prefix)

    if summary["format"] == "pipeann.hybrid.random_single_label_workloads.v1":
        workloads = summary["real_workloads"]
        query_bin = summary.get("query_bin")
    else:
        workloads = summary["workloads"]
        query_bin = summary.get("query_bin")

    buckets: list[dict[str, Any]] = []
    for item in workloads:
        buckets.append(
            {
                "name": item["bucket_name"],
                "label": item["bucket_label"],
                "lower": float(item["selectivity"]),
                "upper": float(item["selectivity"]),
                "midpoint": float(item["selectivity"]),
                "query_count": int(item["query_count"]),
                "query_bin": str(item["query_bin"]),
                "query_labels": str(item["query_labels"]),
                "probe_query_bin": str(item["probe_query_bin"]),
                "probe_query_labels": str(item["probe_query_labels"]),
            }
        )

    manifest = {
        "format": "pipeann.hybrid.selectivity_manifest.v1",
        "index_prefix": str(index_prefix),
        "index_type": args.index_type,
        "selector_type": args.selector_type,
        "query_bin": None if query_bin is None else str(query_bin),
        "query_labels": None,
        "sidecar_path": str(Path(f"{index_prefix}_labels.densebit")),
        "npoints": int(summary["npoints"]),
        "scanned_queries": sum(int(item["query_count"]) for item in workloads),
        "queries_per_bucket": None,
        "buckets": buckets,
    }
    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] wrote manifest to {manifest_path}")
    print(f"[summary] buckets={len(buckets)} index_prefix={index_prefix}")
    return manifest_path


def build_default_bucket_specs() -> list[BucketSpec]:
    return [parse_bucket_spec(spec) for spec in DEFAULT_BUCKET_SPECS]


def build_default_single_label_targets() -> list[SingleLabelTarget]:
    return [SingleLabelTarget(selectivity=value) for value in DEFAULT_SINGLE_LABEL_TARGETS]


def build_default_synthetic_high_selectivity_specs() -> list[SyntheticSelectivitySpec]:
    return [
        SyntheticSelectivitySpec(name=name, selectivity=selectivity, label_id=label_id)
        for name, selectivity, label_id in DEFAULT_SYNTHETIC_HIGH_SELECTIVITY_SPECS
    ]


def build_default_uniform_exact_selectivity_specs() -> list[SyntheticSelectivitySpec]:
    return [
        SyntheticSelectivitySpec(name=name, selectivity=selectivity, label_id=label_id)
        for label_id, (name, selectivity) in enumerate(DEFAULT_UNIFORM_EXACT_SELECTIVITY_SPECS)
    ]


def build_default_extra_real_high_selectivity_labels() -> list[int]:
    return list(DEFAULT_EXTRA_REAL_HIGH_SELECTIVITY_LABELS)


def parse_single_label_target(value: str) -> SingleLabelTarget:
    selectivity = float(value)
    if selectivity <= 0.0 or selectivity >= 1.0:
        raise ValueError(f"single-label target must be in (0, 1), got {value}")
    return SingleLabelTarget(selectivity=selectivity)


def parse_synthetic_selectivity_spec(value: str, label_id: int) -> SyntheticSelectivitySpec:
    name, separator, selectivity_text = value.partition(":")
    if not separator:
        raise ValueError(
            f"synthetic selectivity spec must be formatted as name:selectivity, got {value!r}"
        )
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError(f"synthetic selectivity spec name must be non-empty, got {value!r}")
    selectivity = float(selectivity_text)
    if selectivity <= 0.0 or selectivity > 1.0:
        raise ValueError(f"synthetic selectivity must be in (0, 1], got {value!r}")
    return SyntheticSelectivitySpec(name=normalized_name, selectivity=selectivity, label_id=label_id)


def resolve_synthetic_selectivity_specs(values: Sequence[str] | None) -> list[SyntheticSelectivitySpec]:
    if not values:
        return build_default_uniform_exact_selectivity_specs()

    specs = [parse_synthetic_selectivity_spec(value, label_id) for label_id, value in enumerate(values)]
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"duplicate synthetic selectivity bucket name: {spec.name}")
        seen_names.add(spec.name)
    return specs


def write_one_hot_spmat(path: Path, nrows: int, nlabels: int, label_id: int) -> None:
    ensure_parent(path)
    if nrows < 0:
        raise ValueError(f"nrows must be non-negative, got {nrows}")
    if label_id < 0 or label_id >= nlabels:
        raise ValueError(f"label_id {label_id} out of range for {nlabels} labels")

    indptr = np.arange(nrows + 1, dtype=np.int64)
    indices = np.full(nrows, label_id, dtype=np.int32)
    data = np.ones(nrows, dtype=np.float32)
    with path.open("wb") as writer:
        writer.write(SPMAT_HEADER.pack(nrows, nlabels, nrows))
        indptr.tofile(writer)
        indices.tofile(writer)
        data.tofile(writer)


def write_spmat_from_membership_masks(
    path: Path,
    masks: Sequence[np.ndarray],
    *,
    chunk_rows: int,
) -> None:
    if not masks:
        raise ValueError("write_spmat_from_membership_masks requires at least one label mask")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

    nrows = int(masks[0].shape[0])
    normalized_masks: list[np.ndarray] = []
    for label_id, mask in enumerate(masks):
        if mask.ndim != 1 or int(mask.shape[0]) != nrows:
            raise ValueError(f"mask {label_id} shape mismatch: expected ({nrows},), got {mask.shape}")
        normalized_masks.append(np.asarray(mask, dtype=np.bool_))

    nnz = sum(int(np.count_nonzero(mask)) for mask in normalized_masks)
    ensure_parent(path)
    with path.open("wb") as writer:
        writer.write(SPMAT_HEADER.pack(nrows, len(normalized_masks), nnz))

        np.asarray([0], dtype=np.int64).tofile(writer)
        running_nnz = 0
        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            row_counts = np.zeros(end - start, dtype=np.int16)
            for mask in normalized_masks:
                row_counts += mask[start:end].astype(np.int16, copy=False)
            indptr_chunk = running_nnz + np.cumsum(row_counts, dtype=np.int64)
            indptr_chunk.tofile(writer)
            if indptr_chunk.size > 0:
                running_nnz = int(indptr_chunk[-1])

        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            row_counts = np.zeros(end - start, dtype=np.int16)
            for mask in normalized_masks:
                row_counts += mask[start:end].astype(np.int16, copy=False)
            chunk_nnz = int(row_counts.sum(dtype=np.int64))
            if chunk_nnz == 0:
                continue

            local_indptr = np.empty((end - start) + 1, dtype=np.int64)
            local_indptr[0] = 0
            np.cumsum(row_counts, dtype=np.int64, out=local_indptr[1:])
            cursor = local_indptr[:-1].copy()
            chunk_indices = np.empty(chunk_nnz, dtype=np.int32)
            for label_id, mask in enumerate(normalized_masks):
                selected_rows = np.flatnonzero(mask[start:end])
                if selected_rows.size == 0:
                    continue
                positions = cursor[selected_rows]
                chunk_indices[positions] = label_id
                cursor[selected_rows] += 1
            chunk_indices.tofile(writer)

        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            row_counts = np.zeros(end - start, dtype=np.int16)
            for mask in normalized_masks:
                row_counts += mask[start:end].astype(np.int16, copy=False)
            chunk_nnz = int(row_counts.sum(dtype=np.int64))
            if chunk_nnz == 0:
                continue
            np.ones(chunk_nnz, dtype=np.float32).tofile(writer)


def load_scan_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        summary = json.load(reader)
    if summary.get("format") != "pipeann.hybrid.single_label_scan.v1":
        raise ValueError(f"unsupported single-label scan summary format: {path}")
    return summary


def load_label_stats_map(path: Path, label_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(label_id) for label_id in label_ids}
    found: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            label_id = int(record["label_id"])
            if label_id not in wanted:
                continue
            found[label_id] = record
            if len(found) == len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"missing label stats for labels: {missing}")
    return found


def collect_label_row_ids_from_spmat(
    path: Path,
    label_ids: Sequence[int],
    *,
    chunk_nnz: int,
) -> tuple[int, int, dict[int, np.ndarray]]:
    if chunk_nnz <= 0:
        raise ValueError(f"chunk_nnz must be positive, got {chunk_nnz}")

    nrow, ncol, nnz = read_spmat_header(path)
    target_labels = np.asarray(sorted({int(label_id) for label_id in label_ids}), dtype=np.int32)
    if target_labels.size == 0:
        return nrow, ncol, {}

    indptr_offset = SPMAT_HEADER.size
    indices_offset = indptr_offset + (nrow + 1) * np.dtype(np.int64).itemsize
    data_offset = indices_offset + nnz * np.dtype(np.int32).itemsize
    indptr = np.memmap(path, mode="r", dtype=np.int64, offset=indptr_offset, shape=(nrow + 1,))
    indices = np.memmap(path, mode="r", dtype=np.int32, offset=indices_offset, shape=(nnz,))
    data = np.memmap(path, mode="r", dtype=np.float32, offset=data_offset, shape=(nnz,))

    collected: dict[int, list[np.ndarray]] = {int(label_id): [] for label_id in target_labels.tolist()}
    for start in range(0, nnz, chunk_nnz):
        end = min(start + chunk_nnz, nnz)
        chunk_indices = indices[start:end]
        chunk_data = data[start:end]
        mask = np.isin(chunk_indices, target_labels) & (chunk_data != 0.0)
        if not np.any(mask):
            continue

        matched_positions = np.arange(start, end, dtype=np.int64)[mask]
        matched_labels = np.asarray(chunk_indices[mask], dtype=np.int32)
        matched_rows = np.searchsorted(indptr, matched_positions, side="right") - 1
        order = np.argsort(matched_labels, kind="stable")
        sorted_labels = matched_labels[order]
        sorted_rows = matched_rows[order]
        unique_labels, counts = np.unique(sorted_labels, return_counts=True)
        cursor = 0
        for unique_label, count in zip(unique_labels.tolist(), counts.tolist()):
            collected[int(unique_label)].append(np.asarray(sorted_rows[cursor:cursor + count], dtype=np.int32))
            cursor += count

    row_ids_by_label: dict[int, np.ndarray] = {}
    for label_id, chunks in collected.items():
        if not chunks:
            row_ids_by_label[label_id] = np.empty(0, dtype=np.int32)
            continue
        row_ids_by_label[label_id] = np.concatenate(chunks).astype(np.int32, copy=False)
    return nrow, ncol, row_ids_by_label


def create_random_single_label_workloads(args: argparse.Namespace) -> Path:
    base_bin_path = require_file(resolve_path(args.base_bin), "base bin")
    base_labels_path = require_file(resolve_path(args.base_labels), "base labels")
    query_bin_path = require_file(resolve_path(args.query_bin), "query bin")
    scan_summary_path = require_file(resolve_path(args.scan_summary), "single-label scan summary")
    out_dir = validate_output_path(resolve_path(args.out_dir), "random workload output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_scan_summary(scan_summary_path)
    label_stats_path = resolve_path(args.label_stats_jsonl) if args.label_stats_jsonl else scan_summary_path.with_name("single_label_label_stats.jsonl")
    label_stats_path = require_file(label_stats_path, "single-label label stats")

    selected_specs: list[dict[str, Any]] = []
    seen_label_ids: set[int] = set()
    for item in summary.get("recommendations", []):
        primary = item.get("primary")
        if primary is None:
            continue
        label_id = int(primary["label_id"])
        if label_id in seen_label_ids:
            continue
        seen_label_ids.add(label_id)
        selected_specs.append(
            {
                "bucket_name": f"real_{item['target_name']}_l{label_id}",
                "bucket_label": item["target_name"],
                "label_id": label_id,
                "selectivity": float(primary["selectivity"]),
                "candidate_count": int(primary["candidate_count"]),
                "source": "scan-primary",
                "target_name": item["target_name"],
            }
        )

    extra_labels = [int(value) for value in args.extra_real_label] if args.extra_real_label else build_default_extra_real_high_selectivity_labels()
    stats_map = load_label_stats_map(label_stats_path, [spec["label_id"] for spec in selected_specs] + extra_labels)
    for label_id in extra_labels:
        if label_id in seen_label_ids:
            continue
        stat = stats_map[label_id]
        seen_label_ids.add(label_id)
        selected_specs.append(
            {
                "bucket_name": f"real_extra_l{label_id}",
                "bucket_label": f"label {label_id}",
                "label_id": label_id,
                "selectivity": float(stat["selectivity"]),
                "candidate_count": int(stat["candidate_count"]),
                "source": "extra-high-selectivity",
                "target_name": None,
            }
        )

    npoints_from_labels, nlabels, _ = read_spmat_header(base_labels_path)
    npoints_from_bin, dim, _ = load_bin_matrix(base_bin_path, args.index_type)
    if npoints_from_labels != npoints_from_bin:
        raise ValueError(
            f"base point count mismatch: {base_labels_path} has {npoints_from_labels}, {base_bin_path} has {npoints_from_bin}"
        )
    query_npts, query_dim, query_rows = load_bin_matrix(query_bin_path, args.index_type)
    if query_dim != dim:
        raise ValueError(
            f"query dim mismatch: {query_bin_path} has dim {query_dim}, but {base_bin_path} has dim {dim}"
        )
    if query_npts == 0:
        raise ValueError(f"query bin has no rows: {query_bin_path}")

    real_out_dir = out_dir / "real_selected_labels"
    real_out_dir.mkdir(parents=True, exist_ok=True)
    real_workloads: list[dict[str, Any]] = []
    for spec in selected_specs:
        label_id = int(spec["label_id"])

        bucket_dir = real_out_dir / spec["bucket_name"]
        bucket_dir.mkdir(parents=True, exist_ok=True)
        query_labels_path = bucket_dir / "queries.spmat"
        probe_query_bin_path = bucket_dir / "probe_query.bin"
        probe_query_labels_path = bucket_dir / "probe_query.spmat"

        write_one_hot_spmat(query_labels_path, query_npts, nlabels, label_id)
        write_bin_subset(probe_query_bin_path, query_rows, [0])
        write_one_hot_spmat(probe_query_labels_path, 1, nlabels, label_id)

        actual_selectivity = float(spec["candidate_count"]) / npoints_from_bin if npoints_from_bin > 0 else 0.0
        real_workloads.append(
            {
                "bucket_name": spec["bucket_name"],
                "bucket_label": spec["bucket_label"],
                "label_id": label_id,
                "source": spec["source"],
                "target_name": spec["target_name"],
                "candidate_count": int(spec["candidate_count"]),
                "selectivity": actual_selectivity,
                "query_count": query_npts,
                "query_generation_mode": "original-query-bin-relabeled",
                "query_source_pool_size": query_npts,
                "sampled_with_replacement": False,
                "unique_source_count": query_npts,
                "query_bin": str(query_bin_path),
                "query_labels": str(query_labels_path),
                "probe_query_bin": str(probe_query_bin_path),
                "probe_query_labels": str(probe_query_labels_path),
            }
        )

    synthetic_summary_path = None
    if not args.skip_synthetic_high:
        synthetic_args = argparse.Namespace(
            base_bin=str(base_bin_path),
            query_bin=str(query_bin_path),
            out_dir=str(out_dir / "synthetic_high_selectivity"),
            index_type=args.index_type,
            selector_type=args.selector_type,
            index_prefix=None,
            chunk_rows=args.chunk_rows,
        )
        synthetic_summary_path = create_synthetic_high_selectivity_workload(synthetic_args)

    summary_path = out_dir / "random_single_label_workloads_summary.json"
    combined_summary = {
        "format": "pipeann.hybrid.random_single_label_workloads.v1",
        "base_bin": str(base_bin_path),
        "base_labels": str(base_labels_path),
        "query_bin": str(query_bin_path),
        "scan_summary": str(scan_summary_path),
        "label_stats_jsonl": str(label_stats_path),
        "npoints": npoints_from_bin,
        "dim": dim,
        "nlabels": nlabels,
        "queries_per_label": query_npts,
        "real_workloads": real_workloads,
        "synthetic_high_summary": None if synthetic_summary_path is None else str(synthetic_summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(combined_summary, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] wrote random single-label workload summary to {summary_path}")
    for item in real_workloads:
        print(
            f"[real] {item['bucket_name']} label={item['label_id']} "
            f"selectivity={item['selectivity']:.3e} queries={item['query_count']} source={item['query_generation_mode']}"
        )
    if synthetic_summary_path is not None:
        print(f"[ok] synthetic high-selectivity workloads live under {synthetic_summary_path.parent}")
    return summary_path


def create_synthetic_high_selectivity_workload(args: argparse.Namespace) -> Path:
    base_bin_path = require_file(resolve_path(args.base_bin), "base bin")
    query_bin_path = require_file(resolve_path(args.query_bin), "query bin")
    out_dir = validate_output_path(resolve_path(args.out_dir), "synthetic workload output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_default_synthetic_high_selectivity_specs()

    npoints, dim, _ = load_bin_matrix(base_bin_path, args.index_type)
    query_npts, query_dim, query_rows = load_bin_matrix(query_bin_path, args.index_type)
    if query_dim != dim:
        raise ValueError(
            f"query dim mismatch: {query_bin_path} has dim {query_dim}, but {base_bin_path} has dim {dim}"
        )
    if query_npts == 0:
        raise ValueError(f"query bin has no rows: {query_bin_path}")

    rng = np.random.default_rng(DEFAULT_SYNTHETIC_RANDOM_SEED)
    permutation = np.arange(npoints, dtype=np.int32)
    rng.shuffle(permutation)

    masks: list[np.ndarray] = []
    candidate_counts: dict[str, int] = {}
    for spec in specs:
        candidate_count = int(round(spec.selectivity * npoints))
        candidate_count = min(max(candidate_count, 0), npoints)
        if candidate_count == npoints:
            mask = np.ones(npoints, dtype=np.bool_)
        else:
            mask = np.zeros(npoints, dtype=np.bool_)
            mask[permutation[:candidate_count]] = True
        masks.append(mask)
        candidate_counts[spec.name] = candidate_count

    base_labels_path = out_dir / "base.synthetic_high_selectivity.spmat"
    write_spmat_from_membership_masks(base_labels_path, masks, chunk_rows=args.chunk_rows)

    manifest_buckets: list[dict[str, Any]] = []
    workload_summary: list[dict[str, Any]] = []
    for spec in specs:
        bucket_dir = out_dir / spec.name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        query_labels_path = bucket_dir / "queries.spmat"
        probe_query_bin_path = bucket_dir / "probe_query.bin"
        probe_query_labels_path = bucket_dir / "probe_query.spmat"

        write_one_hot_spmat(query_labels_path, query_npts, len(specs), spec.label_id)
        write_bin_subset(probe_query_bin_path, query_rows, [0])
        write_one_hot_spmat(probe_query_labels_path, 1, len(specs), spec.label_id)

        candidate_count = candidate_counts[spec.name]
        selectivity = 0.0 if npoints == 0 else candidate_count / npoints
        manifest_buckets.append(
            {
                "name": spec.name,
                "label": spec.label,
                "lower": selectivity,
                "upper": selectivity,
                "midpoint": selectivity,
                "query_count": query_npts,
                "candidate_counts": [candidate_count] * query_npts,
                "selectivities": [selectivity] * query_npts,
                "query_bin": str(query_bin_path),
                "query_labels": str(query_labels_path),
                "probe_query_bin": str(probe_query_bin_path),
                "probe_query_labels": str(probe_query_labels_path),
            }
        )
        workload_summary.append(
            {
                "bucket_name": spec.name,
                "bucket_label": spec.label,
                "label_id": spec.label_id,
                "candidate_count": candidate_count,
                "selectivity": selectivity,
                "query_count": query_npts,
                "query_generation_mode": "original-query-bin-relabeled",
                "query_source_pool_size": query_npts,
                "query_bin": str(query_bin_path),
                "query_labels": str(query_labels_path),
                "probe_query_bin": str(probe_query_bin_path),
                "probe_query_labels": str(probe_query_labels_path),
            }
        )

    manifest_path = None
    if args.index_prefix:
        index_prefix = resolve_path(args.index_prefix)
        manifest_path = out_dir / "synthetic_high_selectivity_manifest.json"
        manifest = {
            "format": "pipeann.hybrid.selectivity_manifest.v1",
            "index_prefix": str(index_prefix),
            "index_type": args.index_type,
            "selector_type": args.selector_type,
            "query_bin": str(query_bin_path),
            "query_labels": None,
            "sidecar_path": str(Path(f"{index_prefix}_labels.densebit")),
            "npoints": npoints,
            "scanned_queries": len(specs) * query_npts,
            "queries_per_bucket": query_npts,
            "buckets": manifest_buckets,
        }
        with manifest_path.open("w", encoding="utf-8") as writer:
            json.dump(manifest, writer, indent=2, sort_keys=True)
            writer.write("\n")

    summary_path = out_dir / "synthetic_high_selectivity_summary.json"
    summary = {
        "format": "pipeann.hybrid.synthetic_high_selectivity.v1",
        "base_bin": str(base_bin_path),
        "base_labels": str(base_labels_path),
        "query_bin": str(query_bin_path),
        "index_type": args.index_type,
        "selector_type": args.selector_type,
        "npoints": npoints,
        "dim": dim,
        "queries_per_selectivity": query_npts,
        "manifest": None if manifest_path is None else str(manifest_path),
        "workloads": workload_summary,
    }
    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] wrote synthetic base labels to {base_labels_path}")
    for item in workload_summary:
        print(
            f"[bucket] {item['bucket_name']} label={item['bucket_label']} "
            f"selectivity={item['selectivity']:.2%} queries={item['query_count']}"
        )
    if manifest_path is not None:
        print(f"[ok] wrote synthetic manifest to {manifest_path}")
    print(f"[ok] wrote synthetic workload summary to {summary_path}")
    return summary_path


def create_uniform_exact_selectivity_workloads(args: argparse.Namespace) -> Path:
    base_bin_path = require_file(resolve_path(args.base_bin), "base bin")
    query_bin_path = require_file(resolve_path(args.query_bin), "query bin")
    out_dir = validate_output_path(resolve_path(args.out_dir), "uniform workload output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_synthetic_selectivity_specs(args.selectivity_spec)

    npoints, dim, _ = load_bin_matrix(base_bin_path, args.index_type)
    query_npts, query_dim, query_rows = load_bin_matrix(query_bin_path, args.index_type)
    if query_dim != dim:
        raise ValueError(
            f"query dim mismatch: {query_bin_path} has dim {query_dim}, but {base_bin_path} has dim {dim}"
        )
    if query_npts == 0:
        raise ValueError(f"query bin has no rows: {query_bin_path}")

    rng = np.random.default_rng(args.seed)
    masks: list[np.ndarray] = []
    candidate_counts: dict[str, int] = {}
    for spec in specs:
        candidate_count = int(round(spec.selectivity * npoints))
        candidate_count = min(max(candidate_count, 0), npoints)
        if candidate_count == 0:
            raise ValueError(
                f"selectivity {spec.selectivity:.3e} for bucket {spec.name} rounds to zero candidates over {npoints} points"
            )
        if candidate_count == npoints:
            mask = np.ones(npoints, dtype=np.bool_)
        else:
            mask = np.zeros(npoints, dtype=np.bool_)
            selected_rows = rng.choice(npoints, size=candidate_count, replace=False)
            mask[selected_rows] = True
        masks.append(mask)
        candidate_counts[spec.name] = candidate_count

    base_labels_path = out_dir / "base.uniform_exact_selectivity.spmat"
    write_spmat_from_membership_masks(base_labels_path, masks, chunk_rows=args.chunk_rows)

    manifest_buckets: list[dict[str, Any]] = []
    workload_summary: list[dict[str, Any]] = []
    for spec in specs:
        bucket_dir = out_dir / spec.name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        query_labels_path = bucket_dir / "queries.spmat"
        probe_query_bin_path = bucket_dir / "probe_query.bin"
        probe_query_labels_path = bucket_dir / "probe_query.spmat"

        write_one_hot_spmat(query_labels_path, query_npts, len(specs), spec.label_id)
        write_bin_subset(probe_query_bin_path, query_rows, [0])
        write_one_hot_spmat(probe_query_labels_path, 1, len(specs), spec.label_id)

        candidate_count = candidate_counts[spec.name]
        selectivity = 0.0 if npoints == 0 else candidate_count / npoints
        manifest_buckets.append(
            {
                "name": spec.name,
                "label": spec.label,
                "lower": selectivity,
                "upper": selectivity,
                "midpoint": selectivity,
                "query_count": query_npts,
                "candidate_counts": [candidate_count] * query_npts,
                "selectivities": [selectivity] * query_npts,
                "query_bin": str(query_bin_path),
                "query_labels": str(query_labels_path),
                "probe_query_bin": str(probe_query_bin_path),
                "probe_query_labels": str(probe_query_labels_path),
            }
        )
        workload_summary.append(
            {
                "bucket_name": spec.name,
                "bucket_label": spec.label,
                "label_id": spec.label_id,
                "requested_selectivity": spec.selectivity,
                "candidate_count": candidate_count,
                "selectivity": selectivity,
                "query_count": query_npts,
                "query_generation_mode": "original-query-bin-relabeled",
                "query_source_pool_size": query_npts,
                "query_bin": str(query_bin_path),
                "query_labels": str(query_labels_path),
                "probe_query_bin": str(probe_query_bin_path),
                "probe_query_labels": str(probe_query_labels_path),
            }
        )

    manifest_path = None
    if args.index_prefix:
        index_prefix = resolve_path(args.index_prefix)
        manifest_path = out_dir / "uniform_exact_selectivity_manifest.json"
        manifest = {
            "format": "pipeann.hybrid.selectivity_manifest.v1",
            "index_prefix": str(index_prefix),
            "index_type": args.index_type,
            "selector_type": args.selector_type,
            "query_bin": str(query_bin_path),
            "query_labels": None,
            "sidecar_path": str(Path(f"{index_prefix}_labels.densebit")),
            "npoints": npoints,
            "scanned_queries": len(specs) * query_npts,
            "queries_per_bucket": query_npts,
            "buckets": manifest_buckets,
        }
        with manifest_path.open("w", encoding="utf-8") as writer:
            json.dump(manifest, writer, indent=2, sort_keys=True)
            writer.write("\n")

    summary_path = out_dir / "uniform_exact_selectivity_summary.json"
    summary = {
        "format": "pipeann.hybrid.uniform_exact_selectivity.v1",
        "base_bin": str(base_bin_path),
        "base_labels": str(base_labels_path),
        "query_bin": str(query_bin_path),
        "index_type": args.index_type,
        "selector_type": args.selector_type,
        "random_seed": args.seed,
        "npoints": npoints,
        "dim": dim,
        "queries_per_selectivity": query_npts,
        "manifest": None if manifest_path is None else str(manifest_path),
        "workloads": workload_summary,
    }
    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] wrote uniform exact-selectivity base labels to {base_labels_path}")
    for item in workload_summary:
        print(
            f"[bucket] {item['bucket_name']} label={item['bucket_label']} "
            f"selectivity={item['selectivity']:.3e} candidates={item['candidate_count']} queries={item['query_count']}"
        )
    if manifest_path is not None:
        print(f"[ok] wrote uniform exact-selectivity manifest to {manifest_path}")
    print(f"[ok] wrote uniform exact-selectivity workload summary to {summary_path}")
    return summary_path


def scan_query_label_counts(matrix: SpmatMatrix, nlabels: int, max_rows: int = 0) -> tuple[np.ndarray, int]:
    counts = np.zeros(nlabels, dtype=np.int64)
    scanned_rows = matrix.nrow if max_rows == 0 else min(matrix.nrow, max_rows)
    for row_id in range(scanned_rows):
        labels = set(matrix.row_labels(row_id))
        for label in labels:
            if 0 <= label < nlabels:
                counts[label] += 1
    return counts, scanned_rows


def scan_base_label_counts_from_spmat(path: Path, chunk_nnz: int = 10_000_000) -> tuple[int, int, np.ndarray]:
    with path.open("rb") as reader:
        header = reader.read(SPMAT_HEADER.size)
        if len(header) != SPMAT_HEADER.size:
            raise ValueError(f"invalid spmat header: {path}")
        nrow, ncol, nnz = SPMAT_HEADER.unpack(header)

    indptr_bytes = (int(nrow) + 1) * np.dtype(np.int64).itemsize
    indices_offset = SPMAT_HEADER.size + indptr_bytes
    data_offset = indices_offset + int(nnz) * np.dtype(np.int32).itemsize
    indices = np.memmap(path, mode="r", dtype=np.int32, offset=indices_offset, shape=(int(nnz),))
    data = np.memmap(path, mode="r", dtype=np.float32, offset=data_offset, shape=(int(nnz),))

    counts = np.zeros(int(ncol), dtype=np.int64)
    for start in range(0, int(nnz), chunk_nnz):
        end = min(start + chunk_nnz, int(nnz))
        mask = data[start:end] != 0.0
        if not np.any(mask):
            continue
        chunk_counts = np.bincount(indices[start:end][mask], minlength=int(ncol))
        counts += chunk_counts.astype(np.int64, copy=False)
    return int(nrow), int(ncol), counts


def build_single_label_stats_from_counts(npoints: int, label_counts: np.ndarray, query_counts: np.ndarray) -> list[SingleLabelStat]:
    if label_counts.shape[0] != query_counts.shape[0]:
        raise ValueError(
            f"label count mismatch: base labels={label_counts.shape[0]} query labels={query_counts.shape[0]}"
        )

    label_stats: list[SingleLabelStat] = []
    for label_id in range(label_counts.shape[0]):
        candidate_count = int(label_counts[label_id])
        selectivity = 0.0 if npoints == 0 else candidate_count / npoints
        label_stats.append(
            SingleLabelStat(
                label_id=label_id,
                candidate_count=candidate_count,
                selectivity=selectivity,
                query_count=int(query_counts[label_id]),
            )
        )
    return label_stats


def build_single_label_stats(sidecar: DenseBitsetSidecar, query_counts: np.ndarray) -> list[SingleLabelStat]:
    label_counts = np.zeros(sidecar.nlabels, dtype=np.int64)
    for label_id in range(sidecar.nlabels):
        label_counts[label_id] = sidecar.single_label_candidate_count(label_id)
    return build_single_label_stats_from_counts(sidecar.npoints, label_counts, query_counts)


def single_label_distance(lhs: float, rhs: float) -> float:
    lhs_safe = max(lhs, 1e-12)
    rhs_safe = max(rhs, 1e-12)
    return abs(math.log10(lhs_safe) - math.log10(rhs_safe))


def recommend_single_labels(
    label_stats: Sequence[SingleLabelStat],
    targets: Sequence[SingleLabelTarget],
    min_query_count: int,
) -> list[dict[str, Any]]:
    eligible = [stat for stat in label_stats if stat.candidate_count > 0 and stat.query_count >= min_query_count]
    recommendations: list[dict[str, Any]] = []
    for target in targets:
        ranked = sorted(
            eligible,
            key=lambda stat: (
                single_label_distance(stat.selectivity, target.selectivity),
                -stat.query_count,
                stat.candidate_count,
                stat.label_id,
            ),
        )
        primary = ranked[0] if ranked else None
        backup = ranked[1] if len(ranked) > 1 else None
        recommendations.append(
            {
                "target_selectivity": target.selectivity,
                "target_name": target.name,
                "primary": None if primary is None else {
                    "label_id": primary.label_id,
                    "candidate_count": primary.candidate_count,
                    "selectivity": primary.selectivity,
                    "query_count": primary.query_count,
                    "log10_distance": single_label_distance(primary.selectivity, target.selectivity),
                },
                "backup": None if backup is None else {
                    "label_id": backup.label_id,
                    "candidate_count": backup.candidate_count,
                    "selectivity": backup.selectivity,
                    "query_count": backup.query_count,
                    "log10_distance": single_label_distance(backup.selectivity, target.selectivity),
                },
            }
        )
    return recommendations


def scan_single_label_distribution(args: argparse.Namespace) -> Path:
    query_labels_path = require_file(resolve_path(args.query_labels), "query labels")
    out_dir = validate_output_path(resolve_path(args.out_dir), "scan output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [parse_single_label_target(value) for value in args.target] if args.target else build_default_single_label_targets()
    query_labels = SpmatMatrix.load(query_labels_path)

    index_prefix = resolve_path(args.index_prefix) if args.index_prefix else None
    sidecar_path = Path(f"{index_prefix}_labels.densebit") if index_prefix is not None else None
    base_labels_path = resolve_path(args.base_labels) if args.base_labels else None

    if base_labels_path is not None:
        base_labels_path = require_file(base_labels_path, "base labels")
        npoints, nlabels, label_counts = scan_base_label_counts_from_spmat(base_labels_path)
        query_counts, scanned_queries = scan_query_label_counts(query_labels, nlabels, args.max_scanned_queries)
        label_stats = build_single_label_stats_from_counts(npoints, label_counts, query_counts)
        scan_source = "base-labels"
    else:
        if sidecar_path is None:
            raise ValueError("scan-single-label requires either --base-labels or --index-prefix")
        sidecar_path = require_file(sidecar_path, "densebit sidecar")
        sidecar = DenseBitsetSidecar.load(sidecar_path)
        query_counts, scanned_queries = scan_query_label_counts(query_labels, sidecar.nlabels, args.max_scanned_queries)
        label_stats = build_single_label_stats(sidecar, query_counts)
        npoints = sidecar.npoints
        nlabels = sidecar.nlabels
        scan_source = "densebit-sidecar"

    recommendations = recommend_single_labels(label_stats, targets, args.min_query_count)

    eligible_count = sum(1 for stat in label_stats if stat.query_count >= args.min_query_count and stat.candidate_count > 0)
    summary = {
        "format": "pipeann.hybrid.single_label_scan.v1",
        "index_prefix": None if index_prefix is None else str(index_prefix),
        "base_labels": None if base_labels_path is None else str(base_labels_path),
        "query_labels": str(query_labels_path),
        "sidecar_path": None if sidecar_path is None else str(sidecar_path),
        "scan_source": scan_source,
        "npoints": npoints,
        "nlabels": nlabels,
        "scanned_queries": scanned_queries,
        "min_query_count": args.min_query_count,
        "eligible_label_count": eligible_count,
        "target_selectivities": [target.selectivity for target in targets],
        "recommendations": recommendations,
    }

    stats_path = out_dir / "single_label_label_stats.jsonl"
    if stats_path.exists():
        stats_path.unlink()
    for stat in label_stats:
        write_jsonl(
            stats_path,
            {
                "label_id": stat.label_id,
                "candidate_count": stat.candidate_count,
                "selectivity": stat.selectivity,
                "query_count": stat.query_count,
            },
        )

    summary_path = validate_output_path(resolve_path(args.summary_json), "scan summary output") if args.summary_json else out_dir / "single_label_scan_summary.json"
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
        writer.write("\n")

    print(f"[ok] wrote single-label summary to {summary_path}")
    print(f"[ok] wrote single-label label stats to {stats_path}")
    print(
        f"[summary] source={scan_source} nlabels={nlabels} scanned_queries={scanned_queries} "
        f"eligible_labels={eligible_count} min_query_count={args.min_query_count}"
    )
    for item in recommendations:
        primary = item["primary"]
        backup = item["backup"]
        if primary is None:
            print(f"[target] {item['target_selectivity']:.1e} primary=none backup=none")
            continue
        backup_str = "none"
        if backup is not None:
            backup_str = (
                f"label={backup['label_id']} selectivity={backup['selectivity']:.3e} "
                f"queries={backup['query_count']}"
            )
        print(
            f"[target] {item['target_selectivity']:.1e} "
            f"primary=label={primary['label_id']} selectivity={primary['selectivity']:.3e} queries={primary['query_count']} "
            f"backup={backup_str}"
        )

    return summary_path


def prepare_manifest(args: argparse.Namespace) -> Path:
    index_prefix = resolve_path(args.index_prefix)
    query_bin = require_file(resolve_path(args.query_bin), "query bin")
    query_labels_path = require_file(resolve_path(args.query_labels), "query labels")
    sidecar_path = require_file(Path(f"{index_prefix}_labels.densebit"), "densebit sidecar")
    out_dir = validate_output_path(resolve_path(args.out_dir), "prepare output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_specs = [parse_bucket_spec(spec) for spec in (args.bucket or DEFAULT_BUCKET_SPECS)]
    query_labels = SpmatMatrix.load(query_labels_path)
    query_npts, query_dim, query_rows = load_bin_matrix(query_bin, args.index_type)
    if query_npts != query_labels.nrow:
        raise ValueError(
            f"query count mismatch: {query_bin} has {query_npts}, {query_labels_path} has {query_labels.nrow}"
        )

    sidecar = DenseBitsetSidecar.load(sidecar_path)
    selections: dict[str, list[dict[str, Any]]] = {bucket.name: [] for bucket in bucket_specs}
    selected_counts = {bucket.name: 0 for bucket in bucket_specs}
    scan_limit = query_labels.nrow if args.max_scanned_queries == 0 else min(query_labels.nrow, args.max_scanned_queries)

    for query_id in range(scan_limit):
        labels = query_labels.row_labels(query_id)
        candidate_count = sidecar.count_candidates(args.selector_type, labels)
        selectivity = 0.0 if sidecar.npoints == 0 else candidate_count / sidecar.npoints
        for bucket in bucket_specs:
            if not bucket.matches(selectivity):
                continue
            if selected_counts[bucket.name] >= args.queries_per_bucket:
                break
            selections[bucket.name].append(
                {
                    "query_id": query_id,
                    "candidate_count": candidate_count,
                    "selectivity": selectivity,
                }
            )
            selected_counts[bucket.name] += 1
            break
        if all(count >= args.queries_per_bucket for count in selected_counts.values()):
            break

    manifest_buckets: list[dict[str, Any]] = []
    for bucket in bucket_specs:
        rows = selections[bucket.name]
        if not rows:
            continue
        bucket_dir = out_dir / bucket.name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        query_ids = [int(row["query_id"]) for row in rows]
        probe_ids = query_ids[:1]
        bucket_query_bin = bucket_dir / "queries.bin"
        bucket_query_labels = bucket_dir / "queries.spmat"
        probe_query_bin = bucket_dir / "probe_query.bin"
        probe_query_labels = bucket_dir / "probe_query.spmat"
        write_bin_subset(bucket_query_bin, query_rows, query_ids)
        write_spmat_subset(bucket_query_labels, query_labels, query_ids)
        write_bin_subset(probe_query_bin, query_rows, probe_ids)
        write_spmat_subset(probe_query_labels, query_labels, probe_ids)
        manifest_buckets.append(
            {
                "name": bucket.name,
                "label": bucket.label,
                "lower": bucket.lower,
                "upper": bucket.upper,
                "midpoint": bucket.midpoint,
                "query_count": len(query_ids),
                "query_ids": query_ids,
                "candidate_counts": [int(row["candidate_count"]) for row in rows],
                "selectivities": [float(row["selectivity"]) for row in rows],
                "query_bin": str(bucket_query_bin),
                "query_labels": str(bucket_query_labels),
                "probe_query_bin": str(probe_query_bin),
                "probe_query_labels": str(probe_query_labels),
            }
        )

    manifest_path = validate_output_path(resolve_path(args.manifest), "manifest output") if args.manifest else out_dir / "selectivity_manifest.json"
    ensure_parent(manifest_path)
    manifest = {
        "format": "pipeann.hybrid.selectivity_manifest.v1",
        "index_prefix": str(index_prefix),
        "index_type": args.index_type,
        "selector_type": args.selector_type,
        "query_bin": str(query_bin),
        "query_labels": str(query_labels_path),
        "sidecar_path": str(sidecar_path),
        "npoints": sidecar.npoints,
        "scanned_queries": scan_limit,
        "queries_per_bucket": args.queries_per_bucket,
        "buckets": manifest_buckets,
    }
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, indent=2, sort_keys=True)
        writer.write("\n")
    print(f"[ok] wrote manifest to {manifest_path}")
    for bucket in manifest_buckets:
        print(
            f"[bucket] {bucket['name']} count={bucket['query_count']} "
            f"range={bucket['lower']:.1e}..{bucket['upper']:.1e}"
        )
    return manifest_path


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        manifest = json.load(reader)
    if manifest.get("format") != "pipeann.hybrid.selectivity_manifest.v1":
        raise ValueError(f"unsupported manifest format: {path}")
    return manifest


def extract_single_search_record(path: Path) -> dict[str, Any]:
    records = load_search_records(path)
    if len(records) != 1:
        raise ValueError(f"expected exactly one search record in {path}, found {len(records)}")
    return records[0]


def calibrate_prefilter_rerank(args: argparse.Namespace) -> Path:
    summary_path = require_file(resolve_path(args.summary_json), "workload summary")
    summary = load_workload_summary(summary_path)
    if "base_bin" not in summary or "base_labels" not in summary:
        raise ValueError(f"summary does not include base_bin/base_labels required for calibration: {summary_path}")

    out_dir = validate_output_path(resolve_path(args.out_dir), "calibration output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = validate_output_path(resolve_path(args.output_json), "calibration output json") if args.output_json else out_dir / "prefilter_rerank_calibration.json"
    ensure_parent(output_json)

    index_prefix = resolve_path(args.index_prefix)
    build_dir = resolve_path(args.build_dir)
    search_binary = find_binary(build_dir, "search_disk_index_hybrid")
    selector_type = str(summary["selector_type"])
    index_type = str(summary["index_type"])
    total_points = int(summary["npoints"])

    workloads = load_workload_summary_items(summary)
    workloads_by_name = {str(item["bucket_name"]): item for item in workloads}
    selected_bucket_names = [str(value) for value in args.bucket] if args.bucket else [str(item["bucket_name"]) for item in workloads]
    missing_bucket_names = [name for name in selected_bucket_names if name not in workloads_by_name]
    if missing_bucket_names:
        raise ValueError(f"unknown workload buckets in {summary_path}: {missing_bucket_names}")
    selected_workloads = [workloads_by_name[name] for name in selected_bucket_names]

    base_bin_path = require_file(resolve_path(summary["base_bin"]), "base bin")
    base_labels_path = require_file(resolve_path(summary["base_labels"]), "base labels")
    _, _, base_rows = load_bin_matrix(base_bin_path, index_type)

    needed_label_ids = [
        int(item["label_id"])
        for item in selected_workloads
        if float(item["selectivity"]) <= args.max_selectivity
    ]
    _, _, candidate_ids_by_label = collect_label_row_ids_from_spmat(
        base_labels_path,
        needed_label_ids,
        chunk_nnz=args.chunk_nnz,
    )

    calibration_results: list[dict[str, Any]] = []
    overrides: dict[str, int] = {}
    for workload in selected_workloads:
        bucket_name = str(workload["bucket_name"])
        selectivity = float(workload["selectivity"])
        candidate_count = int(workload["candidate_count"])
        result_record: dict[str, Any] = {
            "bucket_name": bucket_name,
            "bucket_label": str(workload["bucket_label"]),
            "label_id": int(workload["label_id"]),
            "selectivity": selectivity,
            "candidate_count": candidate_count,
        }

        if selectivity > args.max_selectivity:
            result_record.update(
                {
                    "status": "skipped",
                    "skip_reason": f"selectivity_above_max:{args.max_selectivity:.3e}",
                    "prefilter_rerank_l": None,
                }
            )
            calibration_results.append(result_record)
            print(
                f"[skip] bucket={bucket_name} selectivity={selectivity:.3e} exceeds calibration cap {args.max_selectivity:.3e}"
            )
            continue

        bucket_query_bin_path = require_file(resolve_path(workload["query_bin"]), f"query bin for {bucket_name}")
        bucket_query_labels_path = require_file(resolve_path(workload["query_labels"]), f"query labels for {bucket_name}")
        query_total, _, query_rows = load_bin_matrix(bucket_query_bin_path, index_type)
        query_labels = SpmatMatrix.load(bucket_query_labels_path)
        if query_total != query_labels.nrow:
            raise ValueError(
                f"query count mismatch for {bucket_name}: {bucket_query_bin_path} has {query_total}, {bucket_query_labels_path} has {query_labels.nrow}"
            )
        calibration_query_count = min(args.calibration_queries, query_total)
        if calibration_query_count <= 0:
            raise ValueError(f"bucket {bucket_name} has no queries available for calibration")

        bucket_dir = out_dir / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        calibration_row_ids = list(range(calibration_query_count))
        calibration_query_bin = bucket_dir / "queries.bin"
        calibration_query_labels = bucket_dir / "queries.spmat"
        truthset_path = bucket_dir / "truthset.bin"
        write_bin_subset(calibration_query_bin, query_rows, calibration_row_ids)
        write_spmat_subset(calibration_query_labels, query_labels, calibration_row_ids)

        candidate_ids = candidate_ids_by_label[int(workload["label_id"])]
        if int(candidate_ids.shape[0]) != candidate_count:
            raise ValueError(
                f"candidate count mismatch for {bucket_name}: summary says {candidate_count}, labels resolve to {candidate_ids.shape[0]}"
            )

        exact_topk_ids = compute_exact_topk_ids(
            np.asarray(query_rows[calibration_row_ids], dtype=np.float32),
            base_rows,
            candidate_ids,
            k=args.k,
            similarity=args.similarity,
            block_candidates=args.block_candidates,
        )
        write_truthset_ids(truthset_path, exact_topk_ids)

        evaluations: dict[int, dict[str, Any]] = {}

        def evaluate_rerank(rerank_l: int) -> dict[str, Any]:
            normalized_rerank = max(int(rerank_l), args.k)
            cached = evaluations.get(normalized_rerank)
            if cached is not None:
                return cached

            raw_jsonl_path = bucket_dir / f"rerank_{normalized_rerank}.jsonl"
            if raw_jsonl_path.exists():
                raw_jsonl_path.unlink()
            query_cmd = [
                str(search_binary),
                index_type,
                str(index_prefix),
                str(args.threads),
                str(args.beamwidth),
                str(calibration_query_bin),
                str(truthset_path),
                str(args.k),
                args.similarity,
                args.nbr_type,
                selector_type,
                str(calibration_query_labels),
                "prefilter",
                "0",
                str(args.mem_l),
                str(args.search_l),
                "--jsonl-output",
                str(raw_jsonl_path),
            ]
            run_command(
                query_cmd,
                timeout=args.timeout,
                log_path=bucket_dir / f"rerank_{normalized_rerank}.log",
                env_overrides={PREFILTER_RERANK_ENV: str(normalized_rerank)},
            )
            record = extract_single_search_record(raw_jsonl_path)
            if record.get("recall") is None:
                raise RuntimeError(f"calibration run for {bucket_name} missing recall in {raw_jsonl_path}")
            evaluation = {
                "rerank_l": normalized_rerank,
                "recall": float(record["recall"]),
                "avg_latency_us": float(record["avg_latency_us"]),
                "qps": float(record["qps"]),
            }
            evaluations[normalized_rerank] = evaluation
            print(
                f"[calibrate] bucket={bucket_name} rerank_l={normalized_rerank} recall={evaluation['recall']:.2f} latency_us={evaluation['avg_latency_us']:.2f}"
            )
            return evaluation

        upper_bound = default_prefilter_rerank_l(args.k, candidate_count, total_points)
        upper_eval = evaluate_rerank(upper_bound)
        while upper_eval["recall"] < args.target_recall and upper_bound < candidate_count:
            next_upper_bound = min(candidate_count, max(upper_bound + 1, upper_bound * 2))
            if next_upper_bound == upper_bound:
                break
            upper_bound = next_upper_bound
            upper_eval = evaluate_rerank(upper_bound)

        if upper_eval["recall"] < args.target_recall:
            result_record.update(
                {
                    "status": "failed",
                    "skip_reason": "target_recall_unreachable_up_to_candidate_count",
                    "calibration_query_count": calibration_query_count,
                    "prefilter_rerank_l": None,
                    "evaluations": sorted(evaluations.values(), key=lambda item: int(item["rerank_l"])),
                }
            )
            calibration_results.append(result_record)
            continue

        lower_bound = args.k
        while lower_bound < upper_bound:
            midpoint = (lower_bound + upper_bound) // 2
            midpoint_eval = evaluate_rerank(midpoint)
            if midpoint_eval["recall"] >= args.target_recall:
                upper_bound = midpoint
            else:
                lower_bound = midpoint + 1

        best_eval = evaluate_rerank(lower_bound)
        overrides[bucket_name] = int(best_eval["rerank_l"])
        result_record.update(
            {
                "status": "calibrated",
                "calibration_query_count": calibration_query_count,
                "target_recall": args.target_recall,
                "prefilter_rerank_l": int(best_eval["rerank_l"]),
                "achieved_recall": float(best_eval["recall"]),
                "achieved_avg_latency_us": float(best_eval["avg_latency_us"]),
                "achieved_qps": float(best_eval["qps"]),
                "evaluations": sorted(evaluations.values(), key=lambda item: int(item["rerank_l"])),
            }
        )
        calibration_results.append(result_record)

    payload = {
        "format": "pipeann.hybrid.prefilter_rerank_calibration.v1",
        "summary_json": str(summary_path),
        "index_prefix": str(index_prefix),
        "index_type": index_type,
        "selector_type": selector_type,
        "similarity": args.similarity,
        "nbr_type": args.nbr_type,
        "k": args.k,
        "threads": args.threads,
        "beamwidth": args.beamwidth,
        "mem_l": args.mem_l,
        "search_l": args.search_l,
        "target_recall": args.target_recall,
        "calibration_queries": args.calibration_queries,
        "max_selectivity": args.max_selectivity,
        "block_candidates": args.block_candidates,
        "overrides": overrides,
        "results": calibration_results,
    }
    with output_json.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True)
        writer.write("\n")
    print(f"[ok] wrote prefilter rerank calibration to {output_json}")
    return output_json


def run_experiment(args: argparse.Namespace) -> Path:
    manifest_path = require_file(resolve_path(args.manifest), "selectivity manifest")
    manifest = load_manifest(manifest_path)
    build_dir = resolve_path(args.build_dir)
    search_binary = find_binary(build_dir, "search_disk_index_hybrid")
    out_dir = validate_output_path(resolve_path(args.out_dir), "run output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = validate_output_path(resolve_path(args.results_jsonl), "results jsonl output") if args.results_jsonl else out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    prefilter_rerank_path = resolve_path(args.prefilter_rerank_json) if args.prefilter_rerank_json else None
    bucket_rerank_overrides = load_prefilter_rerank_overrides(prefilter_rerank_path)

    routes = [route.strip() for route in args.routes.split(",") if route.strip()]
    l_values = [int(value) for value in args.l_values.split(",") if value.strip()]
    if not l_values:
        raise ValueError("provide at least one L value")
    max_l = max(l_values)

    for bucket in manifest["buckets"]:
        bucket_dir = out_dir / bucket["name"]
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bucket_rerank = bucket_rerank_overrides.get(str(bucket["name"]))
        env_overrides = None if bucket_rerank is None else {PREFILTER_RERANK_ENV: str(bucket_rerank)}
        for route in routes:
            raw_jsonl_path = bucket_dir / f"{route}.jsonl"
            if raw_jsonl_path.exists():
                raw_jsonl_path.unlink()

            query_cmd = [
                str(search_binary),
                manifest["index_type"],
                manifest["index_prefix"],
                str(args.threads),
                str(args.beamwidth),
                bucket["query_bin"],
                "null",
                str(args.k),
                args.similarity,
                args.nbr_type,
                manifest["selector_type"],
                bucket["query_labels"],
                route,
                "0",
                str(args.mem_l),
                *[str(value) for value in l_values],
                "--jsonl-output",
                str(raw_jsonl_path),
            ]
            run_command(
                query_cmd,
                timeout=args.timeout,
                log_path=bucket_dir / f"{route}.log",
                env_overrides=env_overrides,
            )

            memory_cmd = [
                str(search_binary),
                manifest["index_type"],
                manifest["index_prefix"],
                str(args.threads),
                str(args.beamwidth),
                bucket["probe_query_bin"],
                "null",
                str(args.k),
                args.similarity,
                args.nbr_type,
                manifest["selector_type"],
                bucket["probe_query_labels"],
                route,
                "0",
                str(args.mem_l),
                str(max_l),
            ]
            peak_memory_kb = run_command_with_time(
                memory_cmd,
                timeout=args.timeout,
                log_path=bucket_dir / f"{route}.memory.log",
                env_overrides=env_overrides,
            )

            for record in load_search_records(raw_jsonl_path):
                record.update(
                    {
                        "dataset": args.dataset_name,
                        "bucket_name": bucket["name"],
                        "bucket_label": bucket["label"],
                        "selectivity_lower": bucket["lower"],
                        "selectivity_upper": bucket["upper"],
                        "selectivity_midpoint": bucket["midpoint"],
                        "bucket_query_count": bucket["query_count"],
                        "peak_memory_kb": peak_memory_kb,
                        "prefilter_rerank_l": bucket_rerank,
                        "prefilter_rerank_source": "calibration" if bucket_rerank is not None else "default-heuristic",
                    }
                )
                write_jsonl(results_path, record)
            print(
                f"[ok] bucket={bucket['name']} route={route} queries={bucket['query_count']} peak_memory_kb={peak_memory_kb} rerank={bucket_rerank if bucket_rerank is not None else 'default'}"
            )

    print(f"[ok] wrote results to {results_path}")
    return results_path


def format_decimal_selectivity(value: float) -> str:
    text = f"{value:.5f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


def format_plot_bucket_label(selectivity_lower: float, selectivity_upper: float) -> str:
    if math.isclose(selectivity_lower, selectivity_upper):
        return format_decimal_selectivity(selectivity_lower)
    return (
        f"{format_decimal_selectivity(selectivity_lower)}"
        f"-{format_decimal_selectivity(selectivity_upper)}"
    )


def format_plot_rerank_label(record: dict[str, Any]) -> str | None:
    rerank_value = record.get("prefilter_rerank_l")
    if rerank_value is not None:
        return str(int(rerank_value))

    bucket_query_count = int(record.get("bucket_query_count", 0) or 0)
    prefilter_count = int(record.get("prefilter_count", 0) or 0)
    graph_count = int(record.get("graph_count", 0) or 0)
    if bucket_query_count > 0 and prefilter_count == 0 and graph_count == bucket_query_count:
        return "0"
    return None


def plot_results(args: argparse.Namespace) -> Path:
    results_path = require_file(resolve_path(args.results_jsonl), "results jsonl")
    records = [record for record in read_jsonl(results_path) if record.get("format") == "pipeann.hybrid.search.v1"]
    if not records:
        raise ValueError(f"no search records found in {results_path}")

    route_filter = {route.strip() for route in args.routes.split(",") if route.strip()} if args.routes else None
    if route_filter is not None:
        records = [record for record in records if record.get("route") in route_filter]
    if not records:
        raise ValueError("no records left after route filter")

    requested_l = args.plot_l
    if requested_l is None:
        available_l = sorted({int(record["L"]) for record in records})
        if len(available_l) != 1:
            raise ValueError(f"multiple L values present {available_l}, set --plot-l")
        requested_l = available_l[0]
    records = [record for record in records if int(record["L"]) == requested_l]
    if not records:
        raise ValueError(f"no records found for L={requested_l}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    route_names = sorted({str(record["route"]) for record in records})
    bucket_order = sorted(
        {
            (
                str(record["bucket_name"]),
                str(record["bucket_label"]),
                float(record["selectivity_lower"]),
                float(record["selectivity_upper"]),
            )
            for record in records
        },
        key=lambda item: (item[2], item[3], item[0]),
    )
    bucket_names = [item[0] for item in bucket_order]
    bucket_labels = [format_plot_bucket_label(item[2], item[3]) for item in bucket_order]
    x_positions = np.arange(len(bucket_order), dtype=float)

    metric_specs = (
        ("avg_latency_us", "Latency (ms)", lambda value: value / 1000.0),
        ("qps", "QPS", lambda value: value),
        ("peak_memory_kb", "Peak Memory (GB)", lambda value: value / (1024.0 * 1024.0)),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    width = 0.8 / max(len(route_names), 1)
    color_cycle = ["#0b6e4f", "#c84c09", "#0b4f8c", "#7c2d12"]

    for axis, (metric_key, title, transform) in zip(axes, metric_specs):
        for route_index, route in enumerate(route_names):
            y_values = []
            annotation_labels = []
            for bucket_name in bucket_names:
                matching = [
                    record for record in records
                    if str(record["route"]) == route and str(record["bucket_name"]) == bucket_name
                ]
                if len(matching) != 1:
                    raise ValueError(f"expected exactly one record for route={route}, bucket={bucket_name}, L={requested_l}")
                matching_record = matching[0]
                y_values.append(transform(float(matching_record[metric_key])))
                annotation_labels.append(format_plot_rerank_label(matching_record))
            offset = (route_index - (len(route_names) - 1) / 2.0) * width
            bars = axis.bar(
                x_positions + offset,
                y_values,
                width=width,
                label=route,
                color=color_cycle[route_index % len(color_cycle)],
            )
            if metric_key == "avg_latency_us":
                y_max = max(y_values) if y_values else 0.0
                text_offset = max(y_max * 0.02, 0.15)
                for bar, label in zip(bars, annotation_labels):
                    if label is None:
                        continue
                    axis.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + text_offset,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=0,
                    )

        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(bucket_labels, rotation=25, ha="right")
        axis.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    axes[0].set_ylabel("Value")
    axes[1].legend(loc="best")
    figure_title = args.title or f"{records[0]['dataset']} hybrid search by selectivity (L={requested_l})"
    fig.suptitle(figure_title, fontsize=14)

    output_path = validate_output_path(resolve_path(args.output), "plot output")
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote figure to {output_path}")
    return output_path


def run_all(args: argparse.Namespace) -> int:
    prepare_args = argparse.Namespace(
        index_prefix=args.index_prefix,
        query_bin=args.query_bin,
        query_labels=args.query_labels,
        selector_type=args.selector_type,
        index_type=args.index_type,
        out_dir=args.work_dir,
        manifest=args.manifest,
        bucket=args.bucket,
        queries_per_bucket=args.queries_per_bucket,
        max_scanned_queries=args.max_scanned_queries,
    )
    manifest_path = prepare_manifest(prepare_args)
    run_args = argparse.Namespace(
        manifest=str(manifest_path),
        build_dir=args.build_dir,
        out_dir=args.work_dir,
        results_jsonl=args.results_jsonl,
        dataset_name=args.dataset_name,
        threads=args.threads,
        beamwidth=args.beamwidth,
        k=args.k,
        similarity=args.similarity,
        nbr_type=args.nbr_type,
        mem_l=args.mem_l,
        routes=args.routes,
        l_values=args.l_values,
        timeout=args.timeout,
        prefilter_rerank_json=args.prefilter_rerank_json,
    )
    results_path = run_experiment(run_args)
    plot_args = argparse.Namespace(
        results_jsonl=str(results_path),
        output=args.output,
        routes=args.plot_routes,
        plot_l=args.plot_l,
        title=args.title,
    )
    plot_results(plot_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run PipeANN-only hybrid selectivity experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan-single-label",
        help="Scan single-label selectivity and recommend representative labels.",
    )
    scan_parser.add_argument("--index-prefix")
    scan_parser.add_argument("--base-labels")
    scan_parser.add_argument("--query-labels", required=True)
    scan_parser.add_argument("--out-dir", required=True)
    scan_parser.add_argument("--summary-json")
    scan_parser.add_argument(
        "--target",
        action="append",
        help="Target single-label selectivity. May be repeated. Defaults to a denser 8-point grid.",
    )
    scan_parser.add_argument("--min-query-count", type=int, default=20)
    scan_parser.add_argument("--max-scanned-queries", type=int, default=0)

    synth_parser = subparsers.add_parser(
        "generate-synthetic-high-selectivity",
        help="Generate 50%/75%/100% synthetic labels and relabeled workloads over the full original query bin.",
    )
    synth_parser.add_argument("--base-bin", required=True)
    synth_parser.add_argument("--query-bin", required=True)
    synth_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    synth_parser.add_argument("--selector-type", default="intersect", choices=["intersect", "subset"])
    synth_parser.add_argument("--out-dir", required=True)
    synth_parser.add_argument("--index-prefix")
    synth_parser.add_argument("--chunk-rows", type=int, default=1_000_000)

    exact_uniform_parser = subparsers.add_parser(
        "generate-uniform-exact-selectivity-workloads",
        help="Generate exact-selectivity synthetic labels by uniformly sampling the requested number of base vectors per label, then relabel the full original query bin.",
    )
    exact_uniform_parser.add_argument("--base-bin", required=True)
    exact_uniform_parser.add_argument("--query-bin", required=True)
    exact_uniform_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    exact_uniform_parser.add_argument("--selector-type", default="intersect", choices=["intersect", "subset"])
    exact_uniform_parser.add_argument("--out-dir", required=True)
    exact_uniform_parser.add_argument(
        "--selectivity-spec",
        action="append",
        help="Exact-selectivity bucket spec in the form name:selectivity. May be repeated. Defaults to an 11-point grid from 1e-5 to 1.0.",
    )
    exact_uniform_parser.add_argument("--index-prefix")
    exact_uniform_parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    exact_uniform_parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_RANDOM_SEED)

    random_single_label_parser = subparsers.add_parser(
        "generate-random-single-label-workloads",
        help="Generate relabeled workloads over the full original query bin for all selected real labels, plus optional synthetic high-selectivity labels.",
    )
    random_single_label_parser.add_argument("--base-bin", required=True)
    random_single_label_parser.add_argument("--base-labels", required=True)
    random_single_label_parser.add_argument("--query-bin", required=True)
    random_single_label_parser.add_argument("--scan-summary", required=True)
    random_single_label_parser.add_argument("--label-stats-jsonl")
    random_single_label_parser.add_argument("--out-dir", required=True)
    random_single_label_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    random_single_label_parser.add_argument("--selector-type", default="intersect", choices=["intersect", "subset"])
    random_single_label_parser.add_argument("--chunk-nnz", type=int, default=10_000_000)
    random_single_label_parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    random_single_label_parser.add_argument("--extra-real-label", action="append", help="Additional real label id to include.")
    random_single_label_parser.add_argument("--skip-synthetic-high", action="store_true")

    runtime_parser = subparsers.add_parser(
        "prepare-index-prefix-for-labels",
        help="Clone an existing graph index prefix and generate a densebit sidecar for a new label file.",
    )
    runtime_parser.add_argument("--source-prefix", required=True)
    runtime_parser.add_argument("--dest-prefix", required=True)
    runtime_parser.add_argument("--label-file", required=True)
    runtime_parser.add_argument("--summary-json")

    manifest_from_summary_parser = subparsers.add_parser(
        "build-manifest-from-summary",
        help="Convert a workload summary JSON into a runnable selectivity manifest.",
    )
    manifest_from_summary_parser.add_argument("--summary-json", required=True)
    manifest_from_summary_parser.add_argument("--index-prefix", required=True)
    manifest_from_summary_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    manifest_from_summary_parser.add_argument("--selector-type", default="intersect", choices=["intersect", "subset"])
    manifest_from_summary_parser.add_argument("--manifest")

    prepare_parser = subparsers.add_parser("prepare", help="Bucket queries by real selectivity and write subset files.")
    prepare_parser.add_argument("--index-prefix", required=True)
    prepare_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    prepare_parser.add_argument("--query-bin", required=True)
    prepare_parser.add_argument("--query-labels", required=True)
    prepare_parser.add_argument("--selector-type", required=True, choices=["intersect", "subset"])
    prepare_parser.add_argument("--out-dir", required=True)
    prepare_parser.add_argument("--manifest")
    prepare_parser.add_argument("--bucket", action="append", help="Bucket spec in the form name:lower,upper.")
    prepare_parser.add_argument("--queries-per-bucket", type=int, default=200)
    prepare_parser.add_argument("--max-scanned-queries", type=int, default=0)

    run_parser = subparsers.add_parser("run", help="Run hybrid search over prepared selectivity buckets.")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--results-jsonl")
    run_parser.add_argument("--dataset-name", default="yfcc10m")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--beamwidth", type=int, default=4)
    run_parser.add_argument("--k", type=int, default=10)
    run_parser.add_argument("--similarity", default="l2")
    run_parser.add_argument("--nbr-type", default="pq")
    run_parser.add_argument("--mem-l", type=int, default=0)
    run_parser.add_argument("--routes", default="auto")
    run_parser.add_argument("--l-values", default="100")
    run_parser.add_argument("--timeout", type=int, default=3600)
    run_parser.add_argument("--prefilter-rerank-json")

    calibrate_parser = subparsers.add_parser(
        "calibrate-rerank",
        help="Find the minimum prefilter rerank count per selectivity bucket that reaches the target recall.",
    )
    calibrate_parser.add_argument("--summary-json", required=True)
    calibrate_parser.add_argument("--index-prefix", required=True)
    calibrate_parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    calibrate_parser.add_argument("--out-dir", required=True)
    calibrate_parser.add_argument("--output-json")
    calibrate_parser.add_argument("--bucket", action="append", help="Specific bucket name to calibrate. May be repeated.")
    calibrate_parser.add_argument("--threads", type=int, default=52)
    calibrate_parser.add_argument("--beamwidth", type=int, default=4)
    calibrate_parser.add_argument("--k", type=int, default=10)
    calibrate_parser.add_argument("--similarity", default="l2")
    calibrate_parser.add_argument("--nbr-type", default="pq")
    calibrate_parser.add_argument("--mem-l", type=int, default=0)
    calibrate_parser.add_argument("--search-l", type=int, default=100)
    calibrate_parser.add_argument("--target-recall", type=float, default=98.0)
    calibrate_parser.add_argument("--calibration-queries", type=int, default=DEFAULT_CALIBRATION_QUERY_COUNT)
    calibrate_parser.add_argument("--max-selectivity", type=float, default=DEFAULT_CALIBRATION_MAX_SELECTIVITY)
    calibrate_parser.add_argument("--block-candidates", type=int, default=DEFAULT_CALIBRATION_BLOCK_CANDIDATES)
    calibrate_parser.add_argument("--chunk-nnz", type=int, default=10_000_000)
    calibrate_parser.add_argument("--timeout", type=int, default=3600)

    plot_parser = subparsers.add_parser("plot", help="Plot latency, QPS and peak memory across selectivity buckets.")
    plot_parser.add_argument("--results-jsonl", required=True)
    plot_parser.add_argument("--output", required=True)
    plot_parser.add_argument("--routes")
    plot_parser.add_argument("--plot-l", type=int)
    plot_parser.add_argument("--title")

    all_parser = subparsers.add_parser("all", help="Run prepare, run and plot in sequence.")
    all_parser.add_argument("--index-prefix", required=True)
    all_parser.add_argument("--index-type", default="uint8", choices=["float", "int8", "uint8"])
    all_parser.add_argument("--query-bin", required=True)
    all_parser.add_argument("--query-labels", required=True)
    all_parser.add_argument("--selector-type", required=True, choices=["intersect", "subset"])
    all_parser.add_argument("--work-dir", required=True)
    all_parser.add_argument("--manifest")
    all_parser.add_argument("--results-jsonl")
    all_parser.add_argument("--output", required=True)
    all_parser.add_argument("--title")
    all_parser.add_argument("--dataset-name", default="yfcc10m")
    all_parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    all_parser.add_argument("--bucket", action="append", help="Bucket spec in the form name:lower,upper.")
    all_parser.add_argument("--queries-per-bucket", type=int, default=200)
    all_parser.add_argument("--max-scanned-queries", type=int, default=0)
    all_parser.add_argument("--threads", type=int, default=1)
    all_parser.add_argument("--beamwidth", type=int, default=4)
    all_parser.add_argument("--k", type=int, default=10)
    all_parser.add_argument("--similarity", default="l2")
    all_parser.add_argument("--nbr-type", default="pq")
    all_parser.add_argument("--mem-l", type=int, default=0)
    all_parser.add_argument("--routes", default="auto")
    all_parser.add_argument("--plot-routes")
    all_parser.add_argument("--l-values", default="100")
    all_parser.add_argument("--plot-l", type=int)
    all_parser.add_argument("--timeout", type=int, default=3600)
    all_parser.add_argument("--prefilter-rerank-json")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "scan-single-label":
        scan_single_label_distribution(args)
        return 0
    if args.command == "generate-synthetic-high-selectivity":
        create_synthetic_high_selectivity_workload(args)
        return 0
    if args.command == "generate-uniform-exact-selectivity-workloads":
        create_uniform_exact_selectivity_workloads(args)
        return 0
    if args.command == "generate-random-single-label-workloads":
        create_random_single_label_workloads(args)
        return 0
    if args.command == "prepare-index-prefix-for-labels":
        create_index_prefix_for_labels(args)
        return 0
    if args.command == "build-manifest-from-summary":
        build_manifest_from_workload_summary(args)
        return 0
    if args.command == "prepare":
        prepare_manifest(args)
        return 0
    if args.command == "calibrate-rerank":
        calibrate_prefilter_rerank(args)
        return 0
    if args.command == "run":
        run_experiment(args)
        return 0
    if args.command == "plot":
        plot_results(args)
        return 0
    if args.command == "all":
        return run_all(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)