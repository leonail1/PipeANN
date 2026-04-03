#!/usr/bin/env python3
"""Build exact filtered GT files for the SIFT1M hybrid exact experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from exact_hybrid_common import (
    DEFAULT_K,
    canonical_selectivities,
    data_labels_path,
    ensure_source_sift1m_assets,
    find_label_id_for_selectivity,
    format_sel,
    gt_path,
    invert_spmat_by_label,
    load_bin_vectors,
    load_selectivity_map,
    load_truthset,
    sift1m_dir,
    write_truthset,
)


def compute_exact_topk(candidate_ids: np.ndarray, candidate_vectors: np.ndarray, queries: np.ndarray, k: int,
                       max_matrix_mb: int) -> tuple[np.ndarray, np.ndarray]:
    nq = queries.shape[0]
    n_candidates = candidate_vectors.shape[0]
    ids_out = np.zeros((nq, k), dtype=np.uint32)
    dists_out = np.full((nq, k), np.inf, dtype=np.float32)

    if n_candidates == 0:
        return ids_out, dists_out

    candidate_norms = np.sum(candidate_vectors * candidate_vectors, axis=1, dtype=np.float32)
    query_norms = np.sum(queries * queries, axis=1, dtype=np.float32)
    max_matrix_bytes = max(1, int(max_matrix_mb)) * 1024 * 1024
    batch_size = max(1, min(128, max_matrix_bytes // max(4, n_candidates * 4)))

    for start in range(0, nq, batch_size):
        end = min(nq, start + batch_size)
        batch = queries[start:end]
        dist_matrix = (
            candidate_norms[:, None]
            + query_norms[start:end][None, :]
            - 2.0 * candidate_vectors @ batch.T
        )
        dist_matrix = np.asarray(dist_matrix, dtype=np.float32)

        if n_candidates <= k:
            top_idx = np.argsort(dist_matrix, axis=0)
        else:
            top_idx = np.argpartition(dist_matrix, kth=k - 1, axis=0)[:k, :]
            top_dist = np.take_along_axis(dist_matrix, top_idx, axis=0)
            order = np.argsort(top_dist, axis=0)
            top_idx = np.take_along_axis(top_idx, order, axis=0)

        picked = top_idx[:k, :]
        picked_dist = np.take_along_axis(dist_matrix, picked, axis=0)
        ids_out[start:end, :picked.shape[0]] = candidate_ids[picked].T
        dists_out[start:end, :picked.shape[0]] = picked_dist.T

    return ids_out, dists_out


def build_one_gt(sel: float, label_id: int, postings: dict[int, np.ndarray], base_vectors: np.ndarray,
                 queries: np.ndarray, k: int, max_matrix_mb: int, out_file, reuse_unfiltered: bool) -> None:
    if reuse_unfiltered:
        ids, dists = load_truthset(sift1m_dir() / "sift_groundtruth.bin")
        if ids.shape[1] < k:
            raise ValueError("sift_groundtruth.bin does not contain enough neighbors")
        write_truthset(out_file, ids[:, :k], dists[:, :k])
        print(f"[reuse] sel={format_sel(sel)} -> {out_file}")
        return

    candidate_ids = postings[label_id]
    candidate_vectors = np.ascontiguousarray(base_vectors[candidate_ids], dtype=np.float32)
    ids, dists = compute_exact_topk(candidate_ids, candidate_vectors, queries, k, max_matrix_mb=max_matrix_mb)
    write_truthset(out_file, ids, dists)
    print(f"[build] sel={format_sel(sel)} label={label_id} candidates={candidate_ids.shape[0]} -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(sift1m_dir() / "sift_base.bin"))
    parser.add_argument("--queries", default=str(sift1m_dir() / "sift_query.bin"))
    parser.add_argument("--data-labels", default=str(data_labels_path()))
    parser.add_argument("--selectivities", default=None,
                        help="Comma-separated selectivities. Defaults to the canonical 9-point exact set.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-matrix-mb", type=int, default=128,
                        help="Upper bound for the candidate x query-batch distance matrix.")
    parser.add_argument("--force", action="store_true", help="Recompute GT files even if they already exist.")
    args = parser.parse_args()

    ensure_source_sift1m_assets(force=False)
    selectivities = canonical_selectivities(args.selectivities)
    selectivity_rows = load_selectivity_map()
    postings = invert_spmat_by_label(Path(args.data_labels))
    queries = np.ascontiguousarray(load_bin_vectors(Path(args.queries)), dtype=np.float32)
    base_vectors = np.ascontiguousarray(load_bin_vectors(Path(args.data)), dtype=np.float32)

    for sel in selectivities:
        out_file = gt_path(sel)
        if out_file.exists() and not args.force:
            print(f"[skip] sel={format_sel(sel)} -> {out_file}")
            continue

        label_id = find_label_id_for_selectivity(sel, selectivity_rows)
        reuse_unfiltered = math.isclose(sel, 1.0, rel_tol=0.0, abs_tol=1e-9) and (sift1m_dir() / "sift_groundtruth.bin").exists()
        build_one_gt(sel, label_id, postings, base_vectors, queries, args.k, args.max_matrix_mb, out_file,
                     reuse_unfiltered=reuse_unfiltered)


if __name__ == "__main__":
    main()
