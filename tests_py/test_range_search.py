from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeann import IndexPipeANN, Metric
from utils import bin_write


def expected_ids(vectors: np.ndarray, query: np.ndarray, search_range: float) -> list[int]:
    actual_dists = np.linalg.norm(vectors - query, axis=1)
    hits = np.flatnonzero(actual_dists <= search_range + 1e-6)
    sq_dists = np.sum((vectors[hits] - query) ** 2, axis=1)
    order = np.argsort(sq_dists)
    return hits[order].astype(int).tolist()


def main() -> None:
    dim = 8
    topk = 7
    search_L = 64
    search_range = 0.19

    vectors = np.full((256, dim), 5.0, dtype=np.float32)
    vectors[:6] = 0.0
    vectors[0, 0] = 0.00
    vectors[1, 0] = 0.05
    vectors[2, 0] = 0.08
    vectors[3, 0] = 0.12
    vectors[4, 0] = 0.18
    vectors[5, 0] = 0.25

    queries = np.zeros((2, dim), dtype=np.float32)
    queries[1] = 20.0

    gt0 = expected_ids(vectors, queries[0], search_range)
    gt1 = expected_ids(vectors, queries[1], search_range)

    with tempfile.TemporaryDirectory(prefix="pipeann_range_search_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "base.bin"
        index_prefix = tmp / "range"
        bin_write(vectors, data_path)

        idx = IndexPipeANN(data_dim=dim, data_type=np.dtype(np.float32), metric=Metric.L2)
        idx.omp_set_num_threads(2)
        idx.build(
            str(data_path),
            str(index_prefix),
            max_nbrs=32,
            build_L=64,
            PQ_bytes=16,
            memory_use_GB=1,
        )
        idx.load(str(index_prefix))

        ids, dists = idx.search(queries, topk=topk, L=search_L, range=search_range)

    max_tag = np.iinfo(np.uint32).max
    max_dist = np.finfo(np.float32).max

    valid0 = ids[0][ids[0] != max_tag].astype(int).tolist()
    valid1 = ids[1][ids[1] != max_tag].astype(int).tolist()

    assert set(valid0).issubset(set(gt0[:topk])), (
        f"query0 returned ids outside range: got={valid0}, expected_subset={gt0[:topk]}"
    )
    assert valid0 and valid0[0] == gt0[0], f"query0 missed nearest id: got={valid0}, expected={gt0[0]}"
    assert valid1 == gt1[:topk], f"query1 ids mismatch: got={valid1}, expected={gt1[:topk]}"

    assert np.all(ids[0][len(valid0):] == max_tag), f"query0 ids not padded: {ids[0].tolist()}"
    assert np.all(ids[1] == max_tag), f"query1 ids not padded: {ids[1].tolist()}"

    assert np.all(dists[0][:len(valid0)] <= search_range * search_range + 1e-6), (
        f"query0 distances exceed partial-order range: {dists[0].tolist()}"
    )
    assert np.all(dists[0][len(valid0):] == max_dist), f"query0 dists not padded: {dists[0].tolist()}"
    assert np.all(dists[1] == max_dist), f"query1 dists not padded: {dists[1].tolist()}"

    print("PASS: range_search returns in-range ids and pads unused slots.")
    print("query0 ids:", ids[0].tolist())
    print("query0 dists:", dists[0].tolist())
    print("query1 ids:", ids[1].tolist())
    print("query1 dists:", dists[1].tolist())


def test_range_search_padding_and_bounds() -> None:
    main()


if __name__ == "__main__":
    main()
