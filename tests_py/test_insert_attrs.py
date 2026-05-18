"""Compare filtered recall of full build vs half-build-plus-insert with attrs.

The dataset is large enough to make the recall gap obvious:
1. build all vectors at once with attrs
2. build the first half, insert the second half with attrs, then save+reload
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeann import AttrsVec, IndexPipeANN, Metric
from utils import bin_write, compute_recall, write_spmat


def label_match_rate(result_ids: np.ndarray, query_labels: np.ndarray, labels: np.ndarray) -> float:
    total = result_ids.size
    matched = 0
    for row, query_label in zip(result_ids, query_labels, strict=True):
        matched += sum(int(labels[int(vid)]) == int(query_label) for vid in row)
    return matched / total


def inserted_hits(result_ids: np.ndarray, split: int) -> int:
    return sum(int(vid) >= split for row in result_ids for vid in row)


def main() -> None:
    rng = np.random.default_rng(123)
    dim = 32
    n_total = int(os.environ.get("PIPEANN_INSERT_ATTRS_N_TOTAL", "12000"))
    n_insert = int(os.environ.get("PIPEANN_INSERT_ATTRS_N_INSERT", "2000"))
    split = n_total - n_insert
    n_queries = int(os.environ.get("PIPEANN_INSERT_ATTRS_N_QUERY", "10"))
    topk = int(os.environ.get("PIPEANN_INSERT_ATTRS_TOPK", "1"))
    search_L = int(os.environ.get("PIPEANN_INSERT_ATTRS_SEARCH_L", "250"))
    recall_threshold = float(os.environ.get("PIPEANN_INSERT_ATTRS_RECALL", "0.9"))
    n_centers = int(os.environ.get("PIPEANN_INSERT_ATTRS_N_CENTERS", "1"))

    # Use cluster id as label so the full-build path gives a stable filtered-search baseline.
    centers = rng.normal(0.0, 4.0, size=(n_centers, dim)).astype(np.float32)
    assignments = rng.integers(0, n_centers, size=n_total)
    vectors = (centers[assignments] + rng.normal(0.0, 0.30, size=(n_total, dim))).astype(np.float32)
    labels = assignments.astype(np.uint32)

    build_vecs = vectors[:split]
    insert_vecs = vectors[split:]
    build_labels = labels[:split]
    insert_labels = labels[split:]

    # Query only the inserted half so the incremental path is forced to retrieve inserted vectors.
    query_ids = rng.choice(n_total - split, size=n_queries, replace=False) + split
    queries = vectors[query_ids].copy() + rng.normal(0.0, 0.02, size=(n_queries, dim)).astype(np.float32)
    query_labels = labels[query_ids]

    gt = np.empty((n_queries, topk), dtype=np.int32)
    for i in range(n_queries):
        ids = np.flatnonzero(labels == query_labels[i])
        dists = np.sum((vectors[ids] - queries[i]) ** 2, axis=1)
        gt[i] = ids[np.argsort(dists)[:topk]]

    all_attrs = AttrsVec(attr_types={0: "label"})
    build_attrs = AttrsVec(attr_types={0: "label"})
    insert_attrs = AttrsVec(attr_types={0: "label"})
    for lbl in labels:
        all_attrs.append({0: [int(lbl)]})
    for lbl in build_labels:
        build_attrs.append({0: [int(lbl)]})
    for lbl in insert_labels:
        insert_attrs.append({0: [int(lbl)]})

    with tempfile.TemporaryDirectory(prefix="pipeann_insert_attrs_") as tmp_dir:
        tmp = Path(tmp_dir)
        all_data_path = tmp / "all.bin"
        base_data_path = tmp / "base.bin"
        full_label_index_path = tmp / "label.full.0"
        split_label_index_path = tmp / "label.split.0"
        label_query_path = tmp / "label_query.spmat"
        full_config_path = tmp / "filter.full.json"
        split_config_path = tmp / "filter.split.json"
        full_prefix = tmp / "full"
        split_prefix = tmp / "split"

        bin_write(vectors, all_data_path)
        bin_write(build_vecs, base_data_path)

        # Full index uses the full attr file. Split index starts from the base-only attr file
        # and updates it through add()+save().
        all_attrs.save(0, full_label_index_path)
        build_attrs.save(0, split_label_index_path)
        write_spmat(
            [[int(lbl)] for lbl in query_labels],
            [[1.0] for _ in query_labels],
            n_centers,
            label_query_path,
        )
        full_config_path.write_text(
            json.dumps(
                {
                    "base": [{"key": 0, "type": "label", "file": str(full_label_index_path)}],
                    "query": {
                        "key": 0,
                        "base_key": 0,
                        "type": "label",
                        "file": str(label_query_path),
                    },
                }
            )
        )
        split_config_path.write_text(
            json.dumps(
                {
                    "base": [{"key": 0, "type": "label", "file": str(split_label_index_path)}],
                    "query": {
                        "key": 0,
                        "base_key": 0,
                        "type": "label",
                        "file": str(label_query_path),
                    },
                }
            )
        )

        full_idx = IndexPipeANN(
            data_dim=dim,
            data_type=np.dtype(np.float32),
            metric=Metric.L2,
        )
        full_idx.omp_set_num_threads(4)
        full_idx.build(
            str(all_data_path),
            str(full_prefix),
            max_nbrs=32,
            build_L=64,
            PQ_bytes=16,
            memory_use_GB=1,
            attrs=all_attrs,
            range_dense=32,
        )
        full_idx.load(str(full_prefix))
        full_selector, full_query_attrs = full_idx.load_filter_from_json(str(full_config_path))
        full_ids, _ = full_idx.search(
            queries,
            topk=topk,
            L=search_L,
            selector=full_selector,
            query_attrs=full_query_attrs,
        )

        split_idx = IndexPipeANN(
            data_dim=dim,
            data_type=np.dtype(np.float32),
            metric=Metric.L2,
        )
        split_idx.omp_set_num_threads(4)
        split_idx.build(
            str(base_data_path),
            str(split_prefix),
            max_nbrs=32,
            build_L=64,
            PQ_bytes=16,
            memory_use_GB=1,
            attrs=build_attrs,
            range_dense=32,
        )
        # Build already writes the on-disk index. Re-open it like normal usage,
        # then load the native attr index before incremental inserts.
        split_idx = IndexPipeANN(
            data_dim=dim,
            data_type=np.dtype(np.float32),
            metric=Metric.L2,
        )
        split_idx.omp_set_num_threads(4)
        split_idx.load(str(split_prefix))
        split_selector, split_query_attrs = split_idx.load_filter_from_json(str(split_config_path))
        split_idx.add(
            insert_vecs,
            np.arange(split, n_total, dtype=np.uint32),
            attrs=insert_attrs,
        )
        split_idx.save(str(split_prefix))
        split_idx.load(str(split_prefix))        
        split_selector, split_query_attrs = split_idx.load_filter_from_json(str(split_config_path))

        split_ids, _ = split_idx.search(
            queries,
            topk=topk,
            L=search_L,
            selector=split_selector,
            query_attrs=split_query_attrs,
        )

        full_recall = compute_recall(full_ids, gt)
        split_recall = compute_recall(split_ids, gt)
        full_label_rate = label_match_rate(full_ids, query_labels, labels)
        split_label_rate = label_match_rate(split_ids, query_labels, labels)
        full_inserted_hits = inserted_hits(full_ids, split)
        split_inserted_hits = inserted_hits(split_ids, split)

        print(f"Dataset: total={n_total}, build={split}, insert={n_total - split}, queries={n_queries}")
        print(f"Full build recall@{topk}: {full_recall:.4f}")
        print(f"Full build label match rate: {full_label_rate:.4f}")
        print(f"Full build inserted hits: {full_inserted_hits}/{full_ids.size}")
        print(f"Split build+insert recall@{topk}: {split_recall:.4f}")
        print(f"Split build+insert label match rate: {split_label_rate:.4f}")
        print(f"Split build+insert inserted hits: {split_inserted_hits}/{split_ids.size}")

        assert full_inserted_hits > 0, (
            f"full build search never returned inserted-half ids for inserted queries: {full_ids.tolist()}"
        )
        assert split_inserted_hits > 0, (
            f"split build+insert search returned only original/build ids: {split_ids.tolist()}"
        )

        assert full_label_rate == 1.0, f"full build returned wrong labels: {full_label_rate:.4f}"
        assert split_label_rate == 1.0, f"split build+insert returned wrong labels: {split_label_rate:.4f}"
        assert full_recall >= recall_threshold, f"full build recall is unexpectedly low: {full_recall:.4f}"
        assert split_recall >= recall_threshold, f"split build+insert recall is unexpectedly low: {split_recall:.4f}"
        assert split_recall >= full_recall - 0.15, (
            f"split build+insert recall regressed too much: full={full_recall:.4f}, split={split_recall:.4f}"
        )
        print("PASS: full build and split build+insert behave consistently.")


def test_insert_attrs_recall() -> None:
    main()


if __name__ == "__main__":
    main()
