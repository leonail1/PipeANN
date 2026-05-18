from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from pipeann import AttrsVec, Attributes, IndexPipeANN, Metric, Selector
from utils import bin_write, compute_recall


class HybridSelector(Selector):
    def __init__(self, tags: np.ndarray, prices: np.ndarray):
        super().__init__()
        self.tags = np.asarray(tags, dtype=np.uint32)
        self.prices = np.asarray(prices, dtype=np.uint32)
        self.tag_to_ids: dict[int, list[int]] = {}
        for idx, tag in enumerate(self.tags.tolist()):
            self.tag_to_ids.setdefault(int(tag), []).append(idx)

    def estimate_selectivity(self, query_attrs: Attributes) -> float:
        tag, lower, upper = self._parse(query_attrs)
        ids = self.tag_to_ids.get(tag, [])
        if not ids:
            return 0.0
        matched = sum(lower <= int(self.prices[idx]) < upper for idx in ids)
        return matched / len(self.tags)

    def estimate_precision(self, query_attrs: Attributes) -> float:
        tag, lower, upper = self._parse(query_attrs)
        ids = self.tag_to_ids.get(tag, [])
        if not ids:
            return 1.0
        bucket_hits = sum(lower <= int(self.prices[idx]) < upper for idx in ids)
        return max(bucket_hits / len(ids), 0.2)

    def estimate_prefilter_reads(self, query_attrs: Attributes) -> int:
        return 1

    def pre_filter(self, query_attrs: Attributes) -> list[int]:
        tag, _, _ = self._parse(query_attrs)
        return self.tag_to_ids.get(tag, [])

    def is_member(
        self, target_id: int, query_attrs: Attributes, target_attrs: Attributes
    ) -> bool:
        tag, lower, upper = self._parse(query_attrs)
        row = target_attrs.to_dict()
        target_tag = row[0][0]
        target_price = row[1][0]
        return target_tag == tag and lower <= target_price < upper

    def estimate_infilter_reads(self, query_attrs: Attributes) -> int:
        return 0

    def prepare_in_filter(self, query_attrs: Attributes) -> None:
        _, lower, upper = self._parse(query_attrs)
        self.prepared_ids = set(
            np.flatnonzero((self.prices >= lower) & (self.prices < upper))
            .astype(np.uint32)
            .tolist()
        )

    def is_member_approx(self, target_id: int, query_attrs: Attributes) -> bool:
        tag, _, _ = self._parse(query_attrs)
        return int(self.tags[target_id]) == tag or target_id in self.prepared_ids

    @staticmethod
    def _parse(query_attrs: Attributes) -> tuple[int, int, int]:
        row = query_attrs.to_dict()
        return int(row[0][0]), int(row[1][0]), int(row[1][1])


def build_dataset(
    rng: np.random.Generator,
    n_base: int,
    n_queries: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[int, list[int]]]]:
    centers = rng.normal(0.0, 5.0, size=(10, dim)).astype(np.float32)
    assignments = rng.integers(0, len(centers), size=n_base)
    vectors = (centers[assignments] + rng.normal(0.0, 0.4, size=(n_base, dim))).astype(
        np.float32
    )
    tags = rng.integers(0, 6, size=n_base, dtype=np.uint32)
    prices = rng.integers(0, 1000, size=n_base, dtype=np.uint32)

    query_ids = rng.choice(n_base, size=n_queries, replace=False)
    queries = vectors[query_ids].copy()
    query_attrs: list[dict[int, list[int]]] = []
    for i, base_id in enumerate(query_ids):
        tag = int(tags[base_id])
        lower = max(0, int(prices[base_id]) - 70)
        upper = min(1000, lower + 140)
        lower = upper - 140
        query_attrs.append({0: [tag], 1: [lower, upper]})
        queries[i] += rng.normal(0.0, 0.02, size=dim).astype(np.float32)

    return vectors, tags, prices, queries, query_attrs


def compute_ground_truth(
    vectors: np.ndarray,
    tags: np.ndarray,
    prices: np.ndarray,
    queries: np.ndarray,
    query_attrs: list[dict[int, list[int]]],
    topk: int,
) -> np.ndarray:
    gt = np.empty((len(queries), topk), dtype=np.int32)
    for i, attrs in enumerate(query_attrs):
        tag = attrs[0][0]
        lower, upper = attrs[1]
        mask = (tags == tag) & (prices >= lower) & (prices < upper)
        ids = np.flatnonzero(mask)
        dists = np.sum((vectors[ids] - queries[i]) ** 2, axis=1)
        gt[i] = ids[np.argsort(dists)[:topk]]
    return gt


def main() -> None:
    rng = np.random.default_rng(7)
    n_base = int(os.environ.get("PIPEANN_FILTER_N_BASE", "10000"))
    n_queries = int(os.environ.get("PIPEANN_FILTER_N_QUERY", "16"))
    dim = int(os.environ.get("PIPEANN_FILTER_DIM", "32"))
    topk = int(os.environ.get("PIPEANN_FILTER_TOPK", "10"))
    search_L = int(os.environ.get("PIPEANN_FILTER_SEARCH_L", "2000"))
    recall_threshold = float(os.environ.get("PIPEANN_FILTER_RECALL", "0.8"))

    vectors, tags, prices, queries, query_attrs = build_dataset(
        rng, n_base, n_queries, dim
    )
    gt = compute_ground_truth(vectors, tags, prices, queries, query_attrs, topk)

    attrs = AttrsVec(attr_types={0: "label", 1: "range"})
    for tag, price in zip(tags, prices, strict=True):
        attrs.append({0: [int(tag)], 1: [int(price)]})

    with tempfile.TemporaryDirectory(prefix="pipeann_filter_py_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "base.bin"
        index_prefix = tmp / "custom"
        bin_write(vectors, data_path)

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

        selector = HybridSelector(tags, prices)
        ids, dists = idx.search(
            queries, topk=topk, L=search_L, selector=selector, query_attrs=query_attrs
        )
        recall = compute_recall(ids, gt)

        print("OK: python selector example finished.")
        print("Index dir:", tmp)
        print(f"Recall@{topk}:", f"{recall:.4f}")
        print("Query 0 attrs:", query_attrs[0])
        print(
            "Query 0 pre-filter size:",
            len(selector.pre_filter(Attributes(query_attrs[0]))),
        )
        selector.prepare_in_filter(Attributes(query_attrs[0]))
        print("Query 0 in-filter size:", len(selector.prepared_ids))
        print("Query 0 ids:", ids[0].tolist())
        print("Query 0 dists:", dists[0].tolist())
        assert recall >= recall_threshold, f"python selector recall is too low: {recall:.4f}"


def test_python_selector_recall() -> None:
    main()


if __name__ == "__main__":
    main()
