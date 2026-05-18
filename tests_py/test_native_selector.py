from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from pipeann import (
    AndSelector,
    AttrsVec,
    Attributes,
    IndexPipeANN,
    LabelAndSelector,
    LabelOrSelector,
    Metric,
    NotSelector,
    OrSelector,
    RangeSelector,
)
from utils import bin_write, compute_recall, write_spmat


def build_dataset(
    rng: np.random.Generator,
    n_base: int,
    n_queries: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    centers = rng.normal(0.0, 4.0, size=(6, dim)).astype(np.float32)
    assignments = rng.integers(0, len(centers), size=n_base)
    vectors = (centers[assignments] + rng.normal(0.0, 0.35, size=(n_base, dim))).astype(
        np.float32
    )
    tags = rng.integers(0, 8, size=n_base, dtype=np.uint32)
    prices = rng.integers(0, 1000, size=n_base, dtype=np.uint32)

    query_ids = rng.choice(n_base, size=n_queries, replace=False)
    queries = vectors[query_ids].copy()
    query_ranges: list[tuple[int, int, int]] = []
    for i, base_id in enumerate(query_ids):
        tag = int(tags[base_id])
        lower = max(0, int(prices[base_id]) - 80)
        upper = min(1000, lower + 160)
        lower = upper - 160
        query_ranges.append((tag, lower, upper))
        queries[i] += rng.normal(0.0, 0.02, size=dim).astype(np.float32)

    return vectors, tags, prices, queries, query_ranges


def compute_ground_truth(
    vectors: np.ndarray,
    tags: np.ndarray,
    prices: np.ndarray,
    queries: np.ndarray,
    query_ranges: list[tuple[int, int, int]],
    topk: int,
) -> np.ndarray:
    gt = np.empty((len(queries), topk), dtype=np.int32)
    for i, (tag, lower, upper) in enumerate(query_ranges):
        mask = (tags == tag) & (prices >= lower) & (prices < upper)
        ids = np.flatnonzero(mask)
        dists = np.sum((vectors[ids] - queries[i]) ** 2, axis=1)
        gt[i] = ids[np.argsort(dists)[:topk]]
    return gt


def main() -> None:
    rng = np.random.default_rng(42)
    n_base = int(os.environ.get("PIPEANN_NATIVE_SELECTOR_N_BASE", "10000"))
    n_queries = int(os.environ.get("PIPEANN_NATIVE_SELECTOR_N_QUERY", "16"))
    dim = int(os.environ.get("PIPEANN_NATIVE_SELECTOR_DIM", "32"))
    topk = int(os.environ.get("PIPEANN_NATIVE_SELECTOR_TOPK", "10"))
    search_L = int(os.environ.get("PIPEANN_NATIVE_SELECTOR_SEARCH_L", "2000"))
    recall_threshold = float(os.environ.get("PIPEANN_NATIVE_SELECTOR_RECALL", "0.9"))

    vectors, tags, prices, queries, query_ranges = build_dataset(
        rng, n_base, n_queries, dim
    )
    gt = compute_ground_truth(vectors, tags, prices, queries, query_ranges, topk)

    attrs = AttrsVec(attr_types={0: "label", 1: "range"})
    for tag, price in zip(tags, prices, strict=True):
        attrs.append({0: [int(tag)], 1: [int(price)]})

    with tempfile.TemporaryDirectory(prefix="pipeann_native_selector_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "base.bin"
        index_prefix = tmp / "native_selector"
        tag_index_path = tmp / "native.label.0"
        range_index_path = tmp / "native.label.1"
        tag_query_path = tmp / "tag_query.spmat"
        range_query_path = tmp / "range_query.spmat"

        bin_write(vectors, data_path)
        attrs.save(0, tag_index_path)
        attrs.save(1, range_index_path)

        tag_rows: list[list[int]] = []
        tag_values: list[list[float]] = []
        range_rows: list[list[int]] = []
        range_values: list[list[float]] = []
        for tag, lower, upper in query_ranges:
            tag_rows.append([tag])
            tag_values.append([1.0])
            range_rows.append([0, 0])
            range_values.append([float(lower), float(upper)])
        write_spmat(tag_rows, tag_values, int(tags.max()) + 1, tag_query_path)
        write_spmat(range_rows, range_values, 1, range_query_path)

        idx = IndexPipeANN(
            data_dim=dim,
            data_type=np.dtype(np.float32),
            metric=Metric.L2,
        )
        idx.omp_set_num_threads(4)
        idx.build(
            str(data_path),
            str(index_prefix),
            max_nbrs=32,
            build_L=100,
            PQ_bytes=16,
            memory_use_GB=1,
            attrs=attrs,
        )
        idx.load(str(index_prefix))

        tag_index = idx.load_attr_index_from_file(0, tag_index_path, "label")
        range_index = idx.load_attr_index_from_file(1, range_index_path, "range")

        native_attrs = AttrsVec()
        native_attrs.load_from_file(0, "label", tag_query_path)
        native_attrs.load_from_file(1, "range", range_query_path)
        npoints = idx.npoints()

        # This tree reduces to (tag == query_tag) AND (price in [l, r)),
        # but it also exercises the native Python constructors for all selector types.
        selector = AndSelector(
            OrSelector(
                LabelOrSelector(key=0, base_key=0, attr_index=tag_index),
                LabelAndSelector(key=0, base_key=0, attr_index=tag_index),
            ),
            RangeSelector(key=1, base_key=1, attr_index=range_index),
            NotSelector(
                NotSelector(
                    LabelOrSelector(key=0, base_key=0, attr_index=tag_index), npoints
                ),
                npoints,
            ),
        )

        ids, dists = idx.search(
            queries, topk=topk, L=search_L, selector=selector, query_attrs=native_attrs
        )
        recall = compute_recall(ids, gt)

        print("OK: native selector composition example finished.")
        print("Index dir:", tmp)
        print(f"Recall@{topk}:", f"{recall:.4f}")
        print("Query 0 attrs:", native_attrs[0].to_dict())
        print("Query 0 ids:", ids[0].tolist())
        print("Query 0 dists:", dists[0].tolist())
        assert recall >= recall_threshold, f"native selector recall is too low: {recall:.4f}"


def test_native_selector_recall() -> None:
    main()


if __name__ == "__main__":
    main()
