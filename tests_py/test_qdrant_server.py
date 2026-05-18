"""Recall smoke test for PipeANN through the Qdrant-compatible API."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from utils import compute_gt, compute_recall, random_clustered_vectors

COLLECTION = "qdrant_recall"
PORT = int(os.environ.get("PIPEANN_QDRANT_PORT", "6335"))
N_BASE = int(os.environ.get("PIPEANN_QDRANT_N_BASE", "10000"))
N_QUERY = int(os.environ.get("PIPEANN_QDRANT_N_QUERY", "32"))
DIM = int(os.environ.get("PIPEANN_QDRANT_DIM", "32"))
TOPK = int(os.environ.get("PIPEANN_QDRANT_TOPK", "10"))
SEARCH_L = int(os.environ.get("PIPEANN_SEARCH_L", "250"))
BATCH_SIZE = int(os.environ.get("PIPEANN_QDRANT_BATCH", "10000"))
RECALL_THRESHOLD = float(os.environ.get("PIPEANN_QDRANT_RECALL", "0.9"))


def main() -> None:
    rng = np.random.default_rng(42)
    vectors = random_clustered_vectors(rng, N_BASE, DIM, n_centers=32)
    query_ids = rng.choice(N_BASE, size=N_QUERY, replace=False)
    queries = vectors[query_ids].copy()
    queries += rng.normal(0.0, 0.02, size=queries.shape).astype(np.float32)
    gt = compute_gt(vectors, queries, TOPK)

    with tempfile.TemporaryDirectory(prefix="pipeann_qdrant_") as data_dir:
        env = os.environ.copy()
        env.update(
            {
                "PIPEANN_DATA_DIR": data_dir,
                "PIPEANN_SEARCH_L": str(SEARCH_L),
                "PIPEANN_QDRANT_PORT": str(PORT),
                "PIPEANN_QDRANT_OMP_THREADS": "4",
            }
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "pipeann.qdrant_server"],
            cwd=str(ROOT),
            env=env,
        )
        try:
            client = _client(f"http://127.0.0.1:{PORT}")
            _wait_until_ready(client)
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
            )

            upload_start = time.perf_counter()
            for start in range(0, len(vectors), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(vectors))
                client.upload_points(
                    COLLECTION,
                    [_point(i, vectors[i]) for i in range(start, end)],
                    batch_size=BATCH_SIZE,
                )
                print(f"uploaded {end}/{len(vectors)}")
            upload_seconds = time.perf_counter() - upload_start

            result_ids = np.empty((N_QUERY, TOPK), dtype=np.int32)
            latencies = []
            for i, query in enumerate(queries):
                start = time.perf_counter()
                result = client.query_points(COLLECTION, query=query.tolist(), limit=TOPK)
                latencies.append(time.perf_counter() - start)
                result_ids[i] = [int(point.id) for point in result.points]

            recall = compute_recall(result_ids, gt)
            latency_ms = np.asarray(latencies) * 1000
            print(f"Dataset: base={N_BASE}, dim={DIM}, queries={N_QUERY}")
            print(
                f"Upload time: {upload_seconds:.3f}s, "
                f"throughput: {len(vectors) / upload_seconds:.2f} vectors/s"
            )
            print(f"Recall@{TOPK}: {recall:.4f}")
            print(
                "Search latency: "
                f"mean={latency_ms.mean():.3f}ms, "
                f"p95={np.percentile(latency_ms, 95):.3f}ms"
            )
            assert recall >= RECALL_THRESHOLD, f"qdrant recall is too low: {recall:.4f}"
            print("PASS: Qdrant-compatible server searches the uploaded vectors.")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _wait_until_ready(client: QdrantClient) -> None:
    for _ in range(100):
        try:
            client.get_collections()
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("qdrant-compatible server did not become ready")


def _client(url: str) -> QdrantClient:
    try:
        return QdrantClient(url=url, timeout=120, check_compatibility=False)
    except TypeError:
        return QdrantClient(url=url, timeout=120)


def _point(idx: int, vector: np.ndarray) -> PointStruct:
    return PointStruct(
        id=int(idx),
        vector=vector.tolist(),
        payload={"text": f"document {idx}", "metadata": {"idx": idx}},
    )


def test_qdrant_server_recall() -> None:
    main()


if __name__ == "__main__":
    main()
