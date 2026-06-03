#!/usr/bin/env python3
"""Shared helpers for feature-only graph/prefilter threshold prediction.

This module intentionally avoids reading query latency, calibration curves, or
known thresholds.  It is for prediction-time observable dataset/vector/label/
layout feature extraction and for writing small auditable artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BIN_HEADER = struct.Struct("<ii")
SPMAT_HEADER = struct.Struct("<qqq")
DEFAULT_SELECTIVITIES = (0.001, 0.003, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50)
DEFAULT_LABEL_FAMILIES = (
    "random_uniform",
    "zipf",
    "head_heavy",
    "tail_heavy",
    "contiguous_front",
    "contiguous_back",
    "projection_clustered",
    "projection_anti_clustered",
    "insertion_order_front",
    "insertion_order_back",
)
LEAKAGE_BLOCKLIST = {
    "s_exp",
    "oracle_threshold",
    "threshold_status",
    "graph_avg_ms",
    "prefilter_avg_ms",
    "latency",
    "recall",
    "calibration",
    "sparse_probe",
    "query_sweep",
}


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"malformed JSONL at {path}:{line_no}: {exc}") from exc
            continue
        if strict and not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_no} is {type(row).__name__}, expected object")
        if isinstance(row, dict):
            rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(sanitize_json(value), sort_keys=True, allow_nan=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(BIN_HEADER.size)
    if len(raw) != BIN_HEADER.size:
        raise ValueError(f"invalid bin header: {path}")
    npoints, dim = BIN_HEADER.unpack(raw)
    return int(npoints), int(dim)


def bin_memmap(path: Path, dtype: str = "float32") -> tuple[int, int, np.memmap]:
    dtype_map = {
        "float": np.float32,
        "float32": np.float32,
        "uint8": np.uint8,
        "int8": np.int8,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype}")
    npoints, dim = read_bin_header(path)
    data = np.memmap(path, mode="r", dtype=dtype_map[dtype], offset=BIN_HEADER.size, shape=(npoints, dim))
    return npoints, dim, data


def sample_ids(npoints: int, limit: int, seed: int) -> np.ndarray:
    if npoints <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(limit), int(npoints))
    rng = np.random.default_rng(seed)
    if count == npoints:
        return np.arange(npoints, dtype=np.int64)
    return np.sort(rng.choice(npoints, size=count, replace=False).astype(np.int64))


def fmean(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else default


def fstd(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.pstdev(finite)) if len(finite) > 1 else default


def percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


def gini(values: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in values if float(value) >= 0.0], dtype=np.float64)
    if arr.size == 0 or float(arr.sum()) == 0.0:
        return 0.0
    arr.sort()
    n = arr.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * arr) / (n * np.sum(arr))) - ((n + 1.0) / n))


def entropy_from_counts(counts: Sequence[int]) -> float:
    arr = np.asarray([int(value) for value in counts if int(value) > 0], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-np.sum(probs * np.log2(probs)))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def vector_global_features(base_bin: Path, *, dtype: str = "float32", sample_size: int = 2048, seed: int = 17) -> dict[str, Any]:
    npoints, dim, data = bin_memmap(base_bin, dtype)
    ids = sample_ids(npoints, sample_size, seed)
    sample = np.asarray(data[ids], dtype=np.float32) if ids.size else np.empty((0, dim), dtype=np.float32)
    if sample.size == 0:
        return {"npoints": npoints, "dim": dim, "vector_sample_size": 0}
    norms = np.linalg.norm(sample, axis=1)
    features: dict[str, Any] = {
        "npoints": npoints,
        "dim": dim,
        "vector_dtype": dtype,
        "bytes_per_vector": dim * np.dtype(np.float32 if dtype in {"float", "float32"} else np.uint8).itemsize,
        "vector_sample_size": int(sample.shape[0]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    pair_count = min(4096, max(0, sample.shape[0] // 2))
    if pair_count > 0:
        rng = np.random.default_rng(seed + 1)
        lhs = rng.integers(0, sample.shape[0], size=pair_count)
        rhs = rng.integers(0, sample.shape[0], size=pair_count)
        dists = np.linalg.norm(sample[lhs] - sample[rhs], axis=1)
        features.update({
            "pairwise_l2_mean": float(dists.mean()),
            "pairwise_l2_std": float(dists.std()),
            "pairwise_l2_cv": float(dists.std() / dists.mean()) if float(dists.mean()) > 0 else 0.0,
            "pairwise_l2_p10": float(np.percentile(dists, 10)),
            "pairwise_l2_p50": float(np.percentile(dists, 50)),
            "pairwise_l2_p90": float(np.percentile(dists, 90)),
        })
    try:
        from sklearn.neighbors import NearestNeighbors

        nn_count = min(512, sample.shape[0])
        nn = NearestNeighbors(n_neighbors=min(8, nn_count), algorithm="auto").fit(sample[:nn_count])
        distances, _ = nn.kneighbors(sample[:nn_count])
        if distances.shape[1] >= 3:
            d1 = np.maximum(distances[:, 1], 1e-12)
            d2 = np.maximum(distances[:, 2], 1e-12)
            ratio = d2 / d1
            logs = np.log(np.maximum(distances[:, 1:], 1e-12) / np.maximum(distances[:, [1]], 1e-12))
            lid = 1.0 / np.maximum(logs[:, 1:].mean(axis=1), 1e-12)
            features.update({
                "nn_d2_d1_ratio_mean": float(ratio.mean()),
                "nn_d2_d1_ratio_std": float(ratio.std()),
                "lid_mean": float(lid.mean()),
                "lid_std": float(lid.std()),
            })
    except Exception:
        features.update({"nn_d2_d1_ratio_mean": 0.0, "nn_d2_d1_ratio_std": 0.0, "lid_mean": 0.0, "lid_std": 0.0})
    try:
        from sklearn.cluster import MiniBatchKMeans

        cluster_count = min(16, max(2, int(math.sqrt(sample.shape[0] / 8))))
        kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=seed, n_init=3, batch_size=512)
        labels = kmeans.fit_predict(sample)
        counts = np.bincount(labels, minlength=cluster_count)
        features.update({
            "cluster_count": int(cluster_count),
            "cluster_inertia_per_point": float(kmeans.inertia_ / max(1, sample.shape[0])),
            "cluster_size_entropy": entropy_from_counts(counts.tolist()),
            "cluster_size_gini": gini(counts.tolist()),
            "cluster_max_fraction": float(counts.max() / max(1, counts.sum())),
        })
    except Exception:
        features.update({
            "cluster_count": 0,
            "cluster_inertia_per_point": 0.0,
            "cluster_size_entropy": 0.0,
            "cluster_size_gini": 0.0,
            "cluster_max_fraction": 0.0,
        })
    return features


def label_global_features(label_counts: Sequence[int], npoints: int) -> dict[str, Any]:
    counts = [int(value) for value in label_counts]
    selectivities = [count / npoints for count in counts] if npoints else [0.0 for _ in counts]
    nonzero_counts = [count for count in counts if count > 0]
    sorted_counts = sorted(counts, reverse=True)
    top = lambda k: sum(sorted_counts[:k]) / max(1, npoints)
    return {
        "label_cardinality": len(counts),
        "label_nonempty_count": len(nonzero_counts),
        "label_empty_ratio": 0.0 if not counts else (len(counts) - len(nonzero_counts)) / len(counts),
        "label_singleton_ratio": 0.0 if not counts else sum(1 for count in counts if count == 1) / len(counts),
        "label_selectivity_min": min(selectivities) if selectivities else 0.0,
        "label_selectivity_max": max(selectivities) if selectivities else 0.0,
        "label_selectivity_mean": fmean(selectivities),
        "label_selectivity_median": percentile(selectivities, 50),
        "label_selectivity_p90": percentile(selectivities, 90),
        "label_selectivity_p95": percentile(selectivities, 95),
        "label_selectivity_p99": percentile(selectivities, 99),
        "label_entropy": entropy_from_counts(counts),
        "label_gini": gini(counts),
        "label_top1_mass": top(1),
        "label_top5_mass": top(5),
        "label_top10_mass": top(10),
        "label_tail_mass": 1.0 - top(10),
    }


def target_label_features(point_ids: np.ndarray, npoints: int, global_features: dict[str, Any],
                          base_bin: Path, *, dtype: str, sample_size: int, seed: int,
                          nodes_per_page: int = 1) -> dict[str, Any]:
    ids = np.asarray(point_ids, dtype=np.int64)
    count = int(ids.size)
    selectivity = 0.0 if npoints == 0 else count / npoints
    features: dict[str, Any] = {
        "target_label_count": count,
        "target_selectivity": selectivity,
        "target_id_span_fraction": 0.0,
        "target_id_gap_mean": 0.0,
        "target_id_gap_std": 0.0,
        "target_contiguous_run_mean": 0.0,
        "target_page_spread_fraction": 0.0,
    }
    if count > 0:
        sorted_ids = np.sort(ids)
        span = int(sorted_ids[-1] - sorted_ids[0] + 1)
        features["target_id_span_fraction"] = span / max(1, npoints)
        if count > 1:
            gaps = np.diff(sorted_ids)
            features["target_id_gap_mean"] = float(gaps.mean())
            features["target_id_gap_std"] = float(gaps.std())
            run_breaks = np.where(gaps != 1)[0]
            run_lengths = np.diff(np.concatenate(([-1], run_breaks, [count - 1])))
            features["target_contiguous_run_mean"] = float(run_lengths.mean())
        page_ids = sorted_ids // max(1, nodes_per_page)
        features["target_page_spread_fraction"] = float(np.unique(page_ids).size / max(1, math.ceil(npoints / max(1, nodes_per_page))))
    if count > 0:
        _, dim, data = bin_memmap(base_bin, dtype)
        rng = np.random.default_rng(seed)
        picked = ids if count <= sample_size else rng.choice(ids, size=sample_size, replace=False)
        sample = np.asarray(data[np.asarray(picked, dtype=np.int64)], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        features.update({
            "target_norm_mean": float(norms.mean()),
            "target_norm_std": float(norms.std()),
            "target_norm_global_delta": float(norms.mean() - float(global_features.get("norm_mean", 0.0))),
        })
        if sample.shape[0] >= 2:
            centroid = sample.mean(axis=0)
            features["target_centroid_norm"] = float(np.linalg.norm(centroid))
            global_centroid_norm_proxy = float(global_features.get("norm_mean", 0.0))
            features["target_centroid_norm_ratio"] = float(features["target_centroid_norm"] / global_centroid_norm_proxy) if global_centroid_norm_proxy > 0 else 0.0
            centered = sample - centroid
            features["target_cov_trace"] = float((centered * centered).sum(axis=1).mean())
        else:
            features["target_centroid_norm"] = float(norms[0]) if norms.size else 0.0
            features["target_centroid_norm_ratio"] = 0.0
            features["target_cov_trace"] = 0.0
    else:
        features.update({
            "target_norm_mean": 0.0,
            "target_norm_std": 0.0,
            "target_norm_global_delta": 0.0,
            "target_centroid_norm": 0.0,
            "target_centroid_norm_ratio": 0.0,
            "target_cov_trace": 0.0,
        })
    return features


def write_spmat_from_memberships(path: Path, npoints: int, nlabels: int, memberships: Sequence[np.ndarray]) -> None:
    rows: list[list[int]] = [[] for _ in range(npoints)]
    for label_id, ids in enumerate(memberships):
        for point_id in np.asarray(ids, dtype=np.int64).tolist():
            if 0 <= point_id < npoints:
                rows[point_id].append(int(label_id))
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        unique = sorted(set(row))
        indices.extend(unique)
        indptr.append(len(indices))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(SPMAT_HEADER.pack(npoints, nlabels, len(indices)))
        np.asarray(indptr, dtype=np.int64).tofile(handle)
        if indices:
            np.asarray(indices, dtype=np.int32).tofile(handle)
            np.ones(len(indices), dtype=np.float32).tofile(handle)


def projection_scores(base_bin: Path, *, dtype: str, sample_ids_for_fit: np.ndarray, seed: int) -> np.ndarray:
    npoints, dim, data = bin_memmap(base_bin, dtype)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction /= norm
    chunk = 262_144
    scores = np.empty(npoints, dtype=np.float32)
    for start in range(0, npoints, chunk):
        end = min(npoints, start + chunk)
        scores[start:end] = np.asarray(data[start:end], dtype=np.float32) @ direction
    return scores


def synthetic_membership_ids(family: str, npoints: int, count: int, seed: int,
                             projection_order: np.ndarray | None = None) -> np.ndarray:
    count = max(0, min(int(count), int(npoints)))
    if count == 0:
        return np.empty(0, dtype=np.uint32)
    rng = np.random.default_rng(seed)
    if family in {"contiguous_front", "insertion_order_front"}:
        return np.arange(count, dtype=np.uint32)
    if family in {"contiguous_back", "insertion_order_back"}:
        return np.arange(npoints - count, npoints, dtype=np.uint32)
    if family == "projection_clustered" and projection_order is not None:
        return np.sort(projection_order[:count].astype(np.uint32))
    if family == "projection_anti_clustered" and projection_order is not None:
        if count >= npoints:
            return np.arange(npoints, dtype=np.uint32)
        positions = np.linspace(0, npoints - 1, count, dtype=np.int64)
        return np.sort(projection_order[positions].astype(np.uint32))
    if family == "zipf":
        weights = 1.0 / np.maximum(np.arange(1, npoints + 1, dtype=np.float64), 1.0)
        weights /= weights.sum()
        return np.sort(rng.choice(npoints, size=count, replace=False, p=weights).astype(np.uint32))
    if family == "head_heavy":
        pool = max(count, min(npoints, max(count * 4, int(0.2 * npoints))))
        return np.sort(rng.choice(pool, size=count, replace=False).astype(np.uint32))
    if family == "tail_heavy":
        pool = max(count, min(npoints, max(count * 4, int(0.2 * npoints))))
        offset = npoints - pool
        return np.sort((offset + rng.choice(pool, size=count, replace=False)).astype(np.uint32))
    return np.sort(rng.choice(npoints, size=count, replace=False).astype(np.uint32))


def load_hardware_constants(path: Path | None) -> dict[str, Any]:
    defaults = {
        "hw_4kb_random_read_avg_us": 0.0,
        "hw_4kb_random_read_p95_us": 0.0,
        "hw_sequential_read_mb_s": 0.0,
        "hw_cpu_distance_mops": 0.0,
        "hw_pq_decode_mops": 0.0,
        "hw_label_lookup_mops": 0.0,
        "hw_thread_count": os.cpu_count() or 0,
    }
    if path is None or not path.exists():
        return defaults
    payload = read_json_if_exists(path)
    for key in defaults:
        if key in payload:
            defaults[key] = payload[key]
    return defaults


def static_index_features(meta: dict[str, Any], index_prefix: Path | None = None, npoints: int = 0) -> dict[str, Any]:
    build_r = int(meta.get("build_R", meta.get("R", 0)) or 0)
    pq_bytes = int(meta.get("pq_bytes", 0) or 0)
    dim = int(meta.get("dim", 0) or 0)
    page_size = 4096
    node_record_size = int(meta.get("node_record_size", 0) or 0)
    if node_record_size <= 0 and dim > 0:
        node_record_size = dim * 4 + build_r * 4 + pq_bytes
    nodes_per_page = max(1, page_size // max(1, node_record_size))
    features = {
        "index_build_R": build_r,
        "index_build_L": int(meta.get("build_L", meta.get("Lbuild", 0)) or 0),
        "index_alpha": float(meta.get("alpha", 0.0) or 0.0),
        "index_pq_bytes": pq_bytes,
        "index_node_record_size": node_record_size,
        "index_page_size": page_size,
        "index_nodes_per_page": nodes_per_page,
        "index_cross_page_node_ratio": 0.0 if node_record_size <= page_size else 1.0,
        "index_packing_utilization": (nodes_per_page * node_record_size / page_size) if node_record_size > 0 else 0.0,
        "index_adjacency_bytes_per_node": build_r * 4,
        "index_expected_4kb_reads_per_hop": 1 if node_record_size <= page_size else math.ceil(node_record_size / page_size),
        "label_sidecar_bytes_per_vector": 0.0,
        "metadata_idmap_bytes_per_vector": 4.0,
    }
    if index_prefix:
        densebit = Path(str(index_prefix) + "_labels.densebit")
        if densebit.exists() and npoints > 0:
            features["label_sidecar_bytes_per_vector"] = densebit.stat().st_size / npoints
        disk = Path(str(index_prefix) + "_disk.index")
        if disk.exists() and npoints > 0:
            features["index_disk_bytes_per_vector"] = disk.stat().st_size / npoints
    return features


def leakage_audit_for_features(feature_names: Iterable[str]) -> list[dict[str, Any]]:
    findings = []
    for name in feature_names:
        lowered = name.lower()
        hits = sorted(token for token in LEAKAGE_BLOCKLIST if token in lowered)
        if hits:
            findings.append({"feature": name, "status": "REJECT", "reason": f"name contains leakage token(s): {', '.join(hits)}"})
    return findings
