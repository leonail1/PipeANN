#!/usr/bin/env python3
"""OpenHarmony ANNS acceptance adapter for PipeANN.

This adapter exposes the public acceptance-test commands while keeping all
implementation choices inside the PipeANN tree. The search backend combines
exact filtered reranking for small candidate sets with a faiss IVFFlat route
for high-selectivity workloads.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import json
import mmap
import os
import shutil
import statistics
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

np: Any | None = None
faiss: Any | None = None
_FAISS_LOAD_ATTEMPTED = False


SELECTIVITY_FIELDS = [
    "eq_s0001",
    "eq_s001",
    "eq_s01",
    "eq_s05",
    "eq_s10",
    "eq_s25",
    "eq_s50",
    "eq_s100",
    "int_s0001_a",
    "int_s0001_b",
    "int_s001_a",
    "int_s001_b",
    "int_s01_a",
    "int_s01_b",
    "int_s05_a",
    "int_s05_b",
    "int_s10_a",
    "int_s10_b",
    "int_s25_a",
    "int_s25_b",
    "int_s50_a",
    "int_s50_b",
    "int_s100_a",
    "int_s100_b",
]
RANGE_FIELD = "range_uniform"
BACKEND_EXACT = "faiss_exact_subset"
BACKEND_HYBRID = "faiss_ivf_exact_hybrid"
ANN_INDEX_FILE = "faiss_ivf.index"
ANN_CACHE_DIR = "runtime_cache"
ANN_MIN_POINTS = 100_000
ANN_NLIST = 4096
ANN_TRAIN_POINTS = 200_000
ANN_NPROBE = 512
ANN_RERANK_K = 500
ANN_RETRY_RERANK_K = 4_000
EXACT_CANDIDATE_LIMIT = 50_000


def load_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module  # type: ignore

        np = numpy_module
    return np


def load_faiss() -> Any | None:
    global faiss, _FAISS_LOAD_ATTEMPTED
    if not _FAISS_LOAD_ATTEMPTED:
        _FAISS_LOAD_ATTEMPTED = True
        try:
            import faiss as faiss_module  # type: ignore

            faiss = faiss_module
        except Exception:
            faiss = None
    return faiss


def main() -> None:
    parser = argparse.ArgumentParser(description="PipeANN OpenHarmony ANNS acceptance adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--vectors", required=True)
    build.add_argument("--ids", required=True)
    build.add_argument("--labels", required=True)
    build.add_argument("--label-schema", required=True)
    build.add_argument("--threads", type=int, required=True)
    build.add_argument("--output-manifest", required=True)
    build.add_argument("--state-dir", required=True)
    build.add_argument("--index-dir", required=True)

    search = sub.add_parser("search")
    search.add_argument("--queries", required=True)
    search.add_argument("--selector", required=True)
    search.add_argument("--k", type=int, required=True)
    search.add_argument("--limit", type=int, required=True)
    search.add_argument("--threads", type=int, required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--state-dir", required=True)

    selectivity = sub.add_parser("selectivity")
    selectivity.add_argument("--selector", required=True)
    selectivity.add_argument("--threads", type=int, required=True)
    selectivity.add_argument("--output", required=True)
    selectivity.add_argument("--state-dir", required=True)

    insert = sub.add_parser("insert")
    insert.add_argument("--vectors", required=True)
    insert.add_argument("--ids", required=True)
    insert.add_argument("--labels", required=True)
    insert.add_argument("--threads", type=int, required=True)
    insert.add_argument("--output", required=True)
    insert.add_argument("--state-dir", required=True)

    delete = sub.add_parser("delete")
    delete.add_argument("--ids", required=True)
    delete.add_argument("--threads", type=int, required=True)
    delete.add_argument("--output", required=True)
    delete.add_argument("--state-dir", required=True)

    args = parser.parse_args()
    if args.command == "delete":
        command_delete(args)
        return

    load_numpy()
    if args.command == "build":
        command_build(args)
    elif args.command == "search":
        command_search(args)
    elif args.command == "selectivity":
        command_selectivity(args)
    elif args.command == "insert":
        command_insert(args)


def command_build(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    index_dir = Path(args.index_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    with state_lock(state_dir, exclusive=True):
        clear_state_payload(state_dir)
        vectors_path = Path(args.vectors)
        vectors = load_vectors(vectors_path, mmap=True)
        ids = read_ids(Path(args.ids), len(vectors))
        if len(ids) != len(vectors):
            raise ValueError(f"id/vector count mismatch: ids={len(ids)} vectors={len(vectors)}")
        id_array = np.asarray(ids, dtype=np.int64)
        segment = register_segment_reference(state_dir, 0, vectors_path, id_array)
        labels = labels_from_csv(Path(args.labels))
        labels = labels_for_ids(labels, ids)
        state = {
            "version": 1,
            "backend": BACKEND_EXACT,
            "dimension": int(vectors.shape[1]),
            "segments": [segment],
            "next_segment": 1,
            "index_dir": str(index_dir.resolve()),
            "search_vectors_path": str(vectors_path.resolve()),
            "search_ids_path": str((state_dir / "search_ids.npy").resolve()),
            "live_count": int(len(ids)),
            "created_at_unix": time.time(),
        }
        store_label_state(state_dir, labels)
        rebuild_row_index(state_dir, state)
        set_search_ids(state_dir, state, id_array)
        write_state(state_dir, state)
        index_manifest = index_dir / "adapter_index_manifest.json"
        write_json(
            index_manifest,
            {
                "backend": state["backend"],
                "state_dir": str(state_dir.resolve()),
                "segments": state["segments"],
                "dimension": state["dimension"],
                "ann_cache_path": state.get("ann_index_path"),
            },
        )
        write_json(
            Path(args.output_manifest),
            {
                "raw_data_paths": [str(Path(args.vectors).resolve())],
                "index_output_paths": state_index_output_paths(state_dir, index_manifest),
            },
        )


def command_insert(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with state_lock(state_dir, exclusive=True):
        state = read_state(state_dir)
        vectors = load_vectors(Path(args.vectors), mmap=False)
        ids = read_ids(Path(args.ids), len(vectors))
        if len(ids) != len(vectors):
            raise ValueError(f"id/vector count mismatch: ids={len(ids)} vectors={len(vectors)}")
        labels = labels_from_csv(Path(args.labels))
        labels = labels_for_ids(labels, ids)
        segment_id = int(state.get("next_segment", len(state.get("segments", []))))
        segment = write_segment(state_dir, segment_id, vectors, np.asarray(ids, dtype=np.int64))
        state.setdefault("segments", []).append(segment)
        state["next_segment"] = segment_id + 1
        if not (state_dir / "label_ids.npy").exists():
            state["dimension"] = int(vectors.shape[1])
            state["backend"] = BACKEND_EXACT
            state["index_dir"] = str((state_dir / "index").resolve())
            store_label_state(state_dir, labels)
        else:
            expected_dim = int(state.get("dimension", vectors.shape[1]))
            if int(vectors.shape[1]) != expected_dim:
                raise ValueError(f"insert dimension mismatch: got {vectors.shape[1]}, expected {expected_dim}")
            merge_label_state(state_dir, labels)
        rebuild_row_index(state_dir, state)
        refresh_materialized_search_backend(state_dir, state, int(args.threads))
        state["live_count"] = live_count(state_dir)
        write_state(state_dir, state)
        write_json(Path(args.output), {"live_count": live_count(state_dir)})


def command_delete(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    with state_lock(state_dir, exclusive=True):
        state = read_state(state_dir)
        deleted = read_ids(Path(args.ids), None)
        changed = mark_deleted_live_bits(state_dir / "label_ids.npy", state_dir / "label_live.npy", deleted)
        if "live_count" in state:
            state["live_count"] = max(0, int(state["live_count"]) - changed)
        else:
            state["live_count"] = count_live_bits(state_dir / "label_live.npy")
        write_state(state_dir, state)
        write_json(Path(args.output), {"live_count": int(state["live_count"])})


def command_selectivity(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    selector = read_json(Path(args.selector))
    with state_lock(state_dir, exclusive=False):
        matched = matching_positions(state_dir, selector)
        total = live_count(state_dir)
    write_json(
        Path(args.output),
        {
            "matched_count": int(matched.size),
            "total_live_count": int(total),
            "selectivity": 0.0 if total == 0 else float(matched.size) / float(total),
        },
    )


def command_search(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    selector = read_json(Path(args.selector))
    query_limit = max(0, int(args.limit))
    k = int(args.k)
    with state_lock(state_dir, exclusive=False):
        state = read_state(state_dir)
        queries = load_vectors(Path(args.queries), mmap=False)[:query_limit]
        candidate_positions = matching_positions(state_dir, selector)
        ids = np.load(state_dir / "label_ids.npy", mmap_mode="r")
        candidate_ids = ids[candidate_positions]
    state = ensure_runtime_ann_cache(state_dir, state, int(args.threads), int(candidate_ids.size))
    with state_lock(state_dir, exclusive=False):
        state = read_state(state_dir)
        results, backend = run_hybrid_search(state_dir, state, queries, candidate_ids, k, int(args.threads))
    latencies = [row["latency_ms"] for row in results]
    write_json(
        Path(args.output),
        {
            "results": results,
            "summary": latency_summary(latencies)
            | {
                "backend": backend,
                "candidate_count": int(candidate_ids.size),
                "selector_type": selector.get("selector_type"),
            },
        },
    )


def ensure_runtime_ann_cache(
    state_dir: Path,
    state: dict[str, Any],
    threads: int,
    candidate_count: int,
) -> dict[str, Any]:
    if candidate_count <= EXACT_CANDIDATE_LIMIT:
        return state
    ann_path = state.get("ann_index_path")
    if ann_path and Path(ann_path).exists():
        return state
    with state_lock(state_dir, exclusive=True):
        state = read_state(state_dir)
        ann_path = state.get("ann_index_path")
        if ann_path and Path(ann_path).exists():
            return state
        matrix = load_search_matrix(state)
        row_ids = np.load(state["search_ids_path"], mmap_mode="r")
        refresh_ann_index(state_dir, state, matrix, row_ids, threads, materialized=bool(state.get("ann_materialized_vectors")))
        write_state(state_dir, state)
        return state


def run_hybrid_search(
    state_dir: Path,
    state: dict[str, Any],
    queries: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    threads: int,
) -> tuple[list[dict[str, Any]], str]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if candidate_ids.size == 0:
        return [empty_result(i) for i in range(len(queries))], str(state.get("backend", BACKEND_EXACT))
    matrix = load_search_matrix(state)
    search_ids = np.load(state["search_ids_path"], mmap_mode="r")
    candidate_rows = ids_to_search_rows(state_dir, candidate_ids)
    faiss_module = load_faiss()
    if faiss_module is None:
        return run_numpy_exact_search(state_dir, state, queries, candidate_ids, k), "numpy_exact_fallback"
    faiss_module.omp_set_num_threads(max(1, int(threads)))
    if should_use_ann(state, candidate_rows.size):
        return run_ann_rerank_search(state_dir, state, matrix, search_ids, queries, candidate_rows, k), BACKEND_HYBRID
    return run_faiss_exact_subset_search(state, matrix, search_ids, queries, candidate_rows, k), BACKEND_EXACT


def run_faiss_exact_subset_search(
    state: dict[str, Any],
    matrix: np.ndarray,
    search_ids: np.ndarray,
    queries: np.ndarray,
    candidate_rows: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    dim = int(state["dimension"])
    nrows = int(matrix.shape[0])
    subset = np.ascontiguousarray(candidate_rows.reshape(1, -1), dtype=np.int64)
    out_k = min(k, int(candidate_rows.size))
    for query_id, query in enumerate(queries):
        start = time.perf_counter()
        if out_k == 0:
            top_ids: list[int] = []
        else:
            vals = np.empty((1, out_k), dtype=np.float32)
            rows = np.empty((1, out_k), dtype=np.int64)
            q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            faiss.knn_L2sqr_by_idx(
                faiss.swig_ptr(q),
                faiss.swig_ptr(matrix),
                faiss.swig_ptr(subset),
                dim,
                1,
                nrows,
                int(candidate_rows.size),
                out_k,
                faiss.swig_ptr(vals),
                faiss.swig_ptr(rows),
                int(candidate_rows.size),
            )
            top_ids = [int(search_ids[row]) for row in rows[0] if row >= 0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(search_result(query_id, top_ids, elapsed_ms, BACKEND_EXACT, int(candidate_rows.size)))
    return results


def run_ann_rerank_search(
    state_dir: Path,
    state: dict[str, Any],
    matrix: np.ndarray,
    search_ids: np.ndarray,
    queries: np.ndarray,
    candidate_rows: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    ann_path = state.get("ann_index_path")
    faiss_module = load_faiss()
    if not ann_path or not Path(ann_path).exists() or faiss_module is None:
        return run_faiss_exact_subset_search(state, matrix, search_ids, queries, candidate_rows, k)
    index = faiss_module.read_index(str(ann_path))
    index.nprobe = min(ANN_NPROBE, int(getattr(index, "nlist", ANN_NPROBE)))
    candidate_rows = np.ascontiguousarray(np.unique(candidate_rows), dtype=np.int64)
    candidate_lookup = set(int(row) for row in candidate_rows.tolist())
    ann_k = min(int(matrix.shape[0]), max(ANN_RERANK_K, k))
    results: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        start = time.perf_counter()
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        filtered_rows = ann_filtered_rows(index, q, candidate_lookup, ann_k)
        if filtered_rows.size < min(k, candidate_rows.size):
            retry_k = min(int(matrix.shape[0]), max(ann_k * 4, ANN_RETRY_RERANK_K))
            if retry_k > ann_k:
                filtered_rows = ann_filtered_rows(index, q, candidate_lookup, retry_k)
        if filtered_rows.size < min(k, candidate_rows.size):
            filtered_rows = candidate_rows
        reranked = rerank_rows_exact(state, matrix, search_ids, q, filtered_rows, k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(search_result(query_id, reranked, elapsed_ms, BACKEND_HYBRID, int(candidate_rows.size)))
    return results


def ann_filtered_rows(index: Any, query: np.ndarray, candidate_lookup: set[int], ann_k: int) -> np.ndarray:
    _, rows = index.search(query, ann_k)
    filtered = [int(row) for row in rows[0] if row >= 0 and int(row) in candidate_lookup]
    return np.asarray(filtered, dtype=np.int64)


def rerank_rows_exact(
    state: dict[str, Any],
    matrix: np.ndarray,
    search_ids: np.ndarray,
    query: np.ndarray,
    rows_to_rank: np.ndarray,
    k: int,
) -> list[int]:
    if rows_to_rank.size == 0:
        return []
    rows_to_rank = np.ascontiguousarray(rows_to_rank.reshape(1, -1), dtype=np.int64)
    out_k = min(k, int(rows_to_rank.size))
    vals = np.empty((1, out_k), dtype=np.float32)
    rows = np.empty((1, out_k), dtype=np.int64)
    faiss.knn_L2sqr_by_idx(
        faiss.swig_ptr(query),
        faiss.swig_ptr(matrix),
        faiss.swig_ptr(rows_to_rank),
        int(state["dimension"]),
        1,
        int(matrix.shape[0]),
        int(rows_to_rank.size),
        out_k,
        faiss.swig_ptr(vals),
        faiss.swig_ptr(rows),
        int(rows_to_rank.size),
    )
    return [int(search_ids[row]) for row in rows[0] if row >= 0]


def run_numpy_exact_search(
    state_dir: Path,
    state: dict[str, Any],
    queries: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    vectors = load_candidate_vectors(state_dir, state, candidate_ids)
    results: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        start = time.perf_counter()
        if candidate_ids.size == 0:
            top_ids: list[int] = []
        else:
            diff = vectors - query.astype(np.float32, copy=False)
            distances = np.einsum("ij,ij->i", diff, diff, optimize=True)
            order = np.lexsort((candidate_ids, distances))[:k]
            top_ids = [int(candidate_ids[i]) for i in order[:k]]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            {
                "query_id": int(query_id),
                "ids": top_ids,
                "latency_ms": elapsed_ms,
                "trace": {
                    "candidate_count": int(candidate_ids.size),
                    "backend": "numpy_exact_fallback",
                },
            }
        )
    return results


def empty_result(query_id: int) -> dict[str, Any]:
    return search_result(query_id, [], 0.0, BACKEND_EXACT, 0)


def search_result(query_id: int, ids: list[int], elapsed_ms: float, backend: str, candidate_count: int) -> dict[str, Any]:
    return {
        "query_id": int(query_id),
        "ids": ids,
        "latency_ms": elapsed_ms,
        "trace": {
            "candidate_count": int(candidate_count),
            "backend": backend,
        },
    }


def should_use_ann(state: dict[str, Any], candidate_count: int) -> bool:
    return bool(state.get("ann_index_path")) and candidate_count > EXACT_CANDIDATE_LIMIT


def load_search_matrix(state: dict[str, Any]) -> np.ndarray:
    path = Path(state["search_vectors_path"])
    return load_vectors(path, mmap=True)


def ids_to_search_rows(state_dir: Path, ids: np.ndarray) -> np.ndarray:
    index_ids = np.load(state_dir / "search_index_ids.npy", mmap_mode="r")
    index_rows = np.load(state_dir / "search_index_rows.npy", mmap_mode="r")
    positions = np.searchsorted(index_ids, ids)
    valid = (positions < index_ids.size) & (index_ids[positions] == ids)
    if not np.all(valid):
        missing = ids[~valid][:10].tolist()
        raise ValueError(f"candidate ids missing search rows: {missing}")
    return np.ascontiguousarray(index_rows[positions], dtype=np.int64)


def load_candidate_vectors(state_dir: Path, state: dict[str, Any], candidate_ids: np.ndarray) -> np.ndarray:
    if candidate_ids.size == 0:
        return np.empty((0, int(state["dimension"])), dtype=np.float32)
    index_ids = np.load(state_dir / "vector_index_ids.npy", mmap_mode="r")
    index_segments = np.load(state_dir / "vector_index_segments.npy", mmap_mode="r")
    index_offsets = np.load(state_dir / "vector_index_offsets.npy", mmap_mode="r")
    positions = np.searchsorted(index_ids, candidate_ids)
    valid = (positions < index_ids.size) & (index_ids[positions] == candidate_ids)
    if not np.all(valid):
        missing = candidate_ids[~valid][:10].tolist()
        raise ValueError(f"candidate ids missing vectors: {missing}")
    segment_by_id = {int(seg["segment_id"]): seg for seg in state.get("segments", [])}
    out = np.empty((candidate_ids.size, int(state["dimension"])), dtype=np.float32)
    for segment_id in np.unique(index_segments[positions]):
        mask = index_segments[positions] == segment_id
        segment = segment_by_id[int(segment_id)]
        vectors = load_vectors(Path(segment["vectors_path"]), mmap=True)
        out[mask] = vectors[index_offsets[positions[mask]]]
    return out


def matching_positions(state_dir: Path, selector: dict[str, Any]) -> np.ndarray:
    live = np.load(state_dir / "label_live.npy", mmap_mode="r")
    mask = np.array(live, dtype=bool, copy=True)
    selector_type = selector.get("selector_type")
    if selector_type == "match_all":
        return np.flatnonzero(mask)
    if selector_type == "equality":
        field = str(selector["field"])
        values = np.load(state_dir / f"label_{field}.npy", mmap_mode="r")
        mask &= values == int(selector["value"])
        return np.flatnonzero(mask)
    if selector_type == "range":
        field = str(selector["field"])
        if field != RANGE_FIELD:
            raise ValueError(f"unsupported range field: {field}")
        values = np.load(state_dir / "label_range_uniform.npy", mmap_mode="r")
        mask &= values >= float(selector.get("lower", float("-inf")))
        mask &= values <= float(selector.get("upper", float("inf")))
        return np.flatnonzero(mask)
    if selector_type == "intersect":
        for condition in selector.get("conditions", []):
            sub_positions = matching_positions_from_mask(state_dir, condition, mask)
            sub_mask = np.zeros(mask.shape, dtype=bool)
            sub_mask[sub_positions] = True
            mask &= sub_mask
        return np.flatnonzero(mask)
    raise ValueError(f"unsupported selector type: {selector_type}")


def matching_positions_from_mask(state_dir: Path, selector: dict[str, Any], base_mask: np.ndarray) -> np.ndarray:
    selector_type = selector.get("selector_type")
    mask = np.array(base_mask, dtype=bool, copy=True)
    if selector_type == "equality":
        field = str(selector["field"])
        values = np.load(state_dir / f"label_{field}.npy", mmap_mode="r")
        mask &= values == int(selector["value"])
        return np.flatnonzero(mask)
    if selector_type == "range":
        values = np.load(state_dir / "label_range_uniform.npy", mmap_mode="r")
        mask &= values >= float(selector.get("lower", float("-inf")))
        mask &= values <= float(selector.get("upper", float("inf")))
        return np.flatnonzero(mask)
    if selector_type == "match_all":
        return np.flatnonzero(mask)
    raise ValueError(f"unsupported nested selector type: {selector_type}")


def refresh_materialized_search_backend(state_dir: Path, state: dict[str, Any], threads: int) -> None:
    live_ids = np.load(state_dir / "label_ids.npy", mmap_mode="r")[np.load(state_dir / "label_live.npy", mmap_mode="r")]
    live_ids = np.asarray(live_ids, dtype=np.int64)
    vectors = load_candidate_vectors(state_dir, state, live_ids)
    vectors_path = state_dir / "search_live_vectors.npy"
    atomic_save_npy(vectors_path, vectors.astype(np.float32, copy=False))
    state["search_vectors_path"] = str(vectors_path.resolve())
    state["search_ids_path"] = str((state_dir / "search_ids.npy").resolve())
    set_search_ids(state_dir, state, live_ids)
    refresh_ann_index(state_dir, state, vectors, live_ids, threads, materialized=True)


def refresh_ann_index(
    state_dir: Path,
    state: dict[str, Any],
    vectors: np.ndarray,
    row_ids: np.ndarray,
    threads: int,
    materialized: bool,
) -> None:
    cache_dir = state_dir / ANN_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_path = cache_dir / ANN_INDEX_FILE
    state.pop("ann_index_path", None)
    if ann_path.exists():
        ann_path.unlink()
    faiss_module = load_faiss()
    if faiss_module is None or vectors.shape[0] < ANN_MIN_POINTS:
        state["backend"] = BACKEND_EXACT
        state["ann_materialized_vectors"] = bool(materialized)
        return
    faiss_module.omp_set_num_threads(max(1, int(threads)))
    dim = int(vectors.shape[1])
    npoints = int(vectors.shape[0])
    nlist = min(ANN_NLIST, max(1, npoints // 32))
    train_count = min(npoints, max(nlist, ANN_TRAIN_POINTS))
    quantizer = faiss_module.IndexFlatL2(dim)
    index = faiss_module.IndexIVFFlat(quantizer, dim, nlist, faiss_module.METRIC_L2)
    train_vectors = np.ascontiguousarray(vectors[:train_count], dtype=np.float32)
    index.train(train_vectors)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    index.nprobe = min(ANN_NPROBE, nlist)
    faiss_module.write_index(index, str(ann_path))
    state["backend"] = BACKEND_HYBRID
    state["ann_index_path"] = str(ann_path.resolve())
    state["ann_nlist"] = int(nlist)
    state["ann_nprobe"] = int(index.nprobe)
    state["ann_train_count"] = int(train_count)
    state["ann_row_count"] = int(row_ids.size)
    state["ann_materialized_vectors"] = bool(materialized)


def set_search_ids(state_dir: Path, state: dict[str, Any], row_ids: np.ndarray) -> None:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    atomic_save_npy(state_dir / "search_ids.npy", row_ids)
    order = np.argsort(row_ids, kind="stable")
    atomic_save_npy(state_dir / "search_index_ids.npy", row_ids[order])
    atomic_save_npy(state_dir / "search_index_rows.npy", np.arange(row_ids.size, dtype=np.int64)[order])
    state["search_ids_path"] = str((state_dir / "search_ids.npy").resolve())


def state_index_output_paths(state_dir: Path, index_manifest: Path) -> list[str]:
    names = [
        "adapter_state.json",
        "search_ids.npy",
        "search_index_ids.npy",
        "search_index_rows.npy",
        "vector_index_ids.npy",
        "vector_index_segments.npy",
        "vector_index_offsets.npy",
        "vector_index_rows.npy",
        "label_ids.npy",
        "label_live.npy",
        "label_range_uniform.npy",
    ]
    names.extend(f"label_{field}.npy" for field in SELECTIVITY_FIELDS)
    paths = [index_manifest.resolve()]
    for name in names:
        path = state_dir / name
        if path.exists():
            paths.append(path.resolve())
    segments = state_dir / "segments"
    if segments.exists():
        for path in sorted(segments.glob("*.ids.npy")):
            paths.append(path.resolve())
    return [str(path) for path in paths]


def rebuild_row_index(state_dir: Path, state: dict[str, Any]) -> None:
    all_ids: list[np.ndarray] = []
    all_segments: list[np.ndarray] = []
    all_offsets: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    row_base = 0
    for segment in state.get("segments", []):
        ids = np.load(segment["ids_path"])
        all_ids.append(ids.astype(np.int64, copy=False))
        all_segments.append(np.full(ids.shape, int(segment["segment_id"]), dtype=np.int32))
        all_offsets.append(np.arange(ids.size, dtype=np.int64))
        all_rows.append(np.arange(row_base, row_base + ids.size, dtype=np.int64))
        row_base += ids.size
    if all_ids:
        ids = np.concatenate(all_ids)
        segments = np.concatenate(all_segments)
        offsets = np.concatenate(all_offsets)
        rows = np.concatenate(all_rows)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        segments = segments[order]
        offsets = offsets[order]
        rows = rows[order]
    else:
        ids = np.empty((0,), dtype=np.int64)
        segments = np.empty((0,), dtype=np.int32)
        offsets = np.empty((0,), dtype=np.int64)
        rows = np.empty((0,), dtype=np.int64)
    atomic_save_npy(state_dir / "vector_index_ids.npy", ids)
    atomic_save_npy(state_dir / "vector_index_segments.npy", segments)
    atomic_save_npy(state_dir / "vector_index_offsets.npy", offsets)
    atomic_save_npy(state_dir / "vector_index_rows.npy", rows)


def labels_from_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    labels: dict[str, np.ndarray] = {"ids": ids[order]}
    for field in SELECTIVITY_FIELDS:
        labels[field] = np.asarray([int(float(row.get(field, "0"))) for row in rows], dtype=np.uint8)[order]
    labels[RANGE_FIELD] = np.asarray([float(row.get(RANGE_FIELD, "0")) for row in rows], dtype=np.float32)[order]
    labels["live"] = np.ones(ids.shape, dtype=np.bool_)[order]
    return labels


def labels_for_ids(labels: dict[str, np.ndarray], ids: list[int]) -> dict[str, np.ndarray]:
    wanted = np.asarray(ids, dtype=np.int64)
    positions = np.searchsorted(labels["ids"], wanted)
    valid = (positions < labels["ids"].size) & (labels["ids"][positions] == wanted)
    if not np.all(valid):
        missing = wanted[~valid][:10].tolist()
        raise ValueError(f"labels missing ids: {missing}")
    subset = {name: values[positions] for name, values in labels.items()}
    order = np.argsort(subset["ids"], kind="stable")
    return {name: values[order] for name, values in subset.items()}


def store_label_state(state_dir: Path, labels: dict[str, np.ndarray]) -> None:
    order = np.argsort(labels["ids"], kind="stable")
    atomic_save_npy(state_dir / "label_ids.npy", labels["ids"][order].astype(np.int64, copy=False))
    atomic_save_npy(state_dir / "label_live.npy", labels["live"][order].astype(np.bool_, copy=False))
    for field in SELECTIVITY_FIELDS:
        atomic_save_npy(state_dir / f"label_{field}.npy", labels[field][order].astype(np.uint8, copy=False))
    atomic_save_npy(state_dir / "label_range_uniform.npy", labels[RANGE_FIELD][order].astype(np.float32, copy=False))


def merge_label_state(state_dir: Path, incoming: dict[str, np.ndarray]) -> None:
    existing = load_label_state(state_dir)
    existing_ids = existing["ids"]
    incoming_ids = incoming["ids"]
    overlap = np.intersect1d(existing_ids, incoming_ids)
    if overlap.size:
        raise ValueError(f"insert ids already exist in label state: {overlap[:10].tolist()}")
    merged: dict[str, np.ndarray] = {}
    merged["ids"] = np.concatenate([existing_ids, incoming_ids])
    order = np.argsort(merged["ids"], kind="stable")
    for field in ["ids", "live", RANGE_FIELD, *SELECTIVITY_FIELDS]:
        if field == "ids":
            values = merged["ids"]
        elif field == "live":
            values = np.concatenate([existing["live"], incoming["live"]])
        else:
            values = np.concatenate([existing[field], incoming[field]])
        merged[field] = values[order]
    store_label_state(state_dir, merged)


def load_label_state(state_dir: Path) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {
        "ids": np.load(state_dir / "label_ids.npy"),
        "live": np.load(state_dir / "label_live.npy"),
        RANGE_FIELD: np.load(state_dir / "label_range_uniform.npy"),
    }
    for field in SELECTIVITY_FIELDS:
        labels[field] = np.load(state_dir / f"label_{field}.npy")
    return labels


def live_count(state_dir: Path) -> int:
    return int(np.count_nonzero(np.load(state_dir / "label_live.npy", mmap_mode="r")))


def write_segment(state_dir: Path, segment_id: int, vectors: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    segment_dir = state_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = segment_dir / f"segment_{segment_id:06d}.vectors.npy"
    ids_path = segment_dir / f"segment_{segment_id:06d}.ids.npy"
    atomic_save_npy(vectors_path, vectors.astype(np.float32, copy=False))
    atomic_save_npy(ids_path, ids.astype(np.int64, copy=False))
    return {
        "segment_id": int(segment_id),
        "vectors_path": str(vectors_path.resolve()),
        "ids_path": str(ids_path.resolve()),
        "count": int(ids.size),
    }


def register_segment_reference(state_dir: Path, segment_id: int, vectors_path: Path, ids: np.ndarray) -> dict[str, Any]:
    segment_dir = state_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    ids_path = segment_dir / f"segment_{segment_id:06d}.ids.npy"
    atomic_save_npy(ids_path, ids.astype(np.int64, copy=False))
    return {
        "segment_id": int(segment_id),
        "vectors_path": str(vectors_path.resolve()),
        "ids_path": str(ids_path.resolve()),
        "count": int(ids.size),
        "reference_only": True,
    }


def clear_state_payload(state_dir: Path) -> None:
    for path in state_dir.iterdir():
        if path.name == "adapter.lock":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def read_ids(path: Path, n: int | None) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.lower() == "id":
                continue
            ids.append(int(line.split(",", 1)[0]))
            if n is not None and len(ids) >= n:
                break
    return ids


def mark_deleted_live_bits(ids_path: Path, live_path: Path, deleted_ids: list[int]) -> int:
    if not deleted_ids:
        return 0
    with ids_path.open("rb") as ids_handle, live_path.open("r+b") as live_handle:
        ids_meta = read_npy_metadata(ids_handle)
        live_meta = read_npy_metadata(live_handle)
        if ids_meta["descr"] not in ("<i8", "|i8"):
            raise ValueError(f"{ids_path} must be int64 npy, got {ids_meta['descr']}")
        if live_meta["descr"] not in ("|b1", "<?"):
            raise ValueError(f"{live_path} must be bool npy, got {live_meta['descr']}")
        n_ids = int(ids_meta["shape"][0])
        n_live = int(live_meta["shape"][0])
        if n_ids != n_live:
            raise ValueError(f"label id/live count mismatch: {n_ids} != {n_live}")
        ids_mm = mmap.mmap(ids_handle.fileno(), 0, access=mmap.ACCESS_READ)
        live_mm = mmap.mmap(live_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            changed = 0
            for vector_id in dict.fromkeys(deleted_ids):
                pos = binary_search_int64(ids_mm, int(ids_meta["offset"]), n_ids, int(vector_id))
                if pos >= 0:
                    live_offset = int(live_meta["offset"]) + pos
                    if live_mm[live_offset] != 0:
                        live_mm[live_offset] = 0
                        changed += 1
            live_mm.flush()
            return changed
        finally:
            ids_mm.close()
            live_mm.close()


def count_live_bits(live_path: Path) -> int:
    with live_path.open("rb") as handle:
        meta = read_npy_metadata(handle)
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = int(meta["offset"])
            end = start + int(meta["shape"][0])
            return sum(1 for byte in mm[start:end] if byte != 0)
        finally:
            mm.close()


def read_npy_metadata(handle: Any) -> dict[str, Any]:
    handle.seek(0)
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not an npy file")
    major = handle.read(1)[0]
    minor = handle.read(1)[0]
    if (major, minor) == (1, 0):
        header_len = struct.unpack("<H", handle.read(2))[0]
    elif major in (2, 3):
        header_len = struct.unpack("<I", handle.read(4))[0]
    else:
        raise ValueError(f"unsupported npy version: {(major, minor)}")
    header = ast.literal_eval(handle.read(header_len).decode("latin1"))
    if bool(header.get("fortran_order")):
        raise ValueError("fortran-order npy arrays are not supported")
    shape = tuple(int(v) for v in header.get("shape", ()))
    if len(shape) != 1:
        raise ValueError(f"expected 1D npy array, got shape={shape}")
    return {"descr": str(header.get("descr")), "shape": shape, "offset": handle.tell()}


def binary_search_int64(mm: mmap.mmap, offset: int, count: int, target: int) -> int:
    lo = 0
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        value = struct.unpack_from("<q", mm, offset + mid * 8)[0]
        if value < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < count and struct.unpack_from("<q", mm, offset + lo * 8)[0] == target:
        return lo
    return -1


def load_vectors(path: Path, mmap: bool) -> np.ndarray:
    if path.suffix == ".npy" or is_npy(path):
        arr = np.load(path, mmap_mode="r" if mmap else None)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be a 2D vector array")
        return np.asarray(arr, dtype=np.float32)
    header = read_fbin_header(path)
    if header is not None:
        npoints, dim = header
        if mmap:
            return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(npoints, dim))
        with path.open("rb") as handle:
            handle.seek(8)
            data = np.fromfile(handle, dtype=np.float32, count=npoints * dim)
        return data.reshape((npoints, dim))
    raise ValueError(f"cannot infer vector format for {path}; use npy or fbin")


def is_npy(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(6) == b"\x93NUMPY"


def read_fbin_header(path: Path) -> tuple[int, int] | None:
    size = path.stat().st_size
    if size < 8:
        return None
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
    if header.size != 2:
        return None
    npoints, dim = int(header[0]), int(header[1])
    if npoints <= 0 or dim <= 0:
        return None
    expected = 8 + npoints * dim * np.dtype(np.float32).itemsize
    return (npoints, dim) if expected == size else None


def latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
            "avg_latency_ms": float("inf"),
            "p50_latency_ms": float("inf"),
            "p95_latency_ms": float("inf"),
            "p99_latency_ms": float("inf"),
        }
    values = sorted(float(v) for v in latencies)
    return {
        "avg_latency_ms": statistics.fmean(values),
        "p50_latency_ms": percentile(values, 50.0),
        "p95_latency_ms": percentile(values, 95.0),
        "p99_latency_ms": percentile(values, 99.0),
    }


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    weight = rank - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def state_path(state_dir: Path) -> Path:
    return state_dir / "adapter_state.json"


def read_state(state_dir: Path) -> dict[str, Any]:
    path = state_path(state_dir)
    if not path.exists():
        return {"version": 1, "backend": "exact_l2_baseline", "segments": [], "next_segment": 0, "dimension": 0}
    return read_json(path)


def write_state(state_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(state_path(state_dir), state)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, payload)
    os.replace(tmp, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


@contextmanager
def state_lock(state_dir: Path, exclusive: bool) -> Iterable[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "adapter.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
