"""Recall smoke test for the LangChain VectorStore integration."""

from __future__ import annotations

import os
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeann import Collection, IndexPipeANN, Metric
from pipeann.langchain import PipeANNVectorStore
from langchain_core.embeddings import Embeddings
from utils import bin_write, compute_gt, compute_recall, random_clustered_vectors

COLLECTION = "langchain_recall"
DIM = int(os.environ.get("PIPEANN_LANGCHAIN_DIM", "32"))
N_BASE = int(os.environ.get("PIPEANN_LANGCHAIN_N_BASE", "10000"))
N_QUERY = int(os.environ.get("PIPEANN_LANGCHAIN_N_QUERY", "32"))
TOPK = int(os.environ.get("PIPEANN_LANGCHAIN_TOPK", "10"))
SEARCH_L = int(os.environ.get("PIPEANN_LANGCHAIN_SEARCH_L", "250"))
RECALL_THRESHOLD = float(os.environ.get("PIPEANN_LANGCHAIN_RECALL", "0.9"))


class FixedEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def main() -> None:
    rng = np.random.default_rng(43)
    vectors = random_clustered_vectors(rng, N_BASE, DIM, n_centers=32)
    query_ids = rng.choice(N_BASE, size=N_QUERY, replace=False)
    queries = vectors[query_ids].copy()
    queries += rng.normal(0.0, 0.02, size=queries.shape).astype(np.float32)
    gt = compute_gt(vectors, queries, TOPK)

    ids = [f"doc-{i}" for i in range(N_BASE)]
    texts = [f"document {i}" for i in range(N_BASE)]
    metadatas = [{"idx": i} for i in range(N_BASE)]
    query_texts = [f"query {i}" for i in range(N_QUERY)]
    embeddings = FixedEmbeddings(
        {
            **{text: vector.tolist() for text, vector in zip(texts, vectors, strict=True)},
            **{text: query.tolist() for text, query in zip(query_texts, queries, strict=True)},
        }
    )

    with tempfile.TemporaryDirectory(prefix="pipeann_langchain_") as data_dir:
        build_disk_collection(
            data_dir,
            COLLECTION,
            vectors,
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )
        collection = Collection.load(data_dir, COLLECTION)
        store = PipeANNVectorStore.from_collection(
            collection,
            embeddings,
            metric="l2",
            search_L=SEARCH_L,
        )

        result_ids = np.empty((N_QUERY, TOPK), dtype=np.int32)
        for i, query_text in enumerate(query_texts):
            docs = store.similarity_search(query_text, k=TOPK)
            result_ids[i] = [int(doc.id.removeprefix("doc-")) for doc in docs]

        recall = compute_recall(result_ids, gt)
        fetched = store.get_by_ids(["doc-0", "missing"])
        assert len(fetched) == 1 and fetched[0].id == "doc-0"

        print(f"Dataset: base={N_BASE}, dim={DIM}, queries={N_QUERY}")
        print(f"Recall@{TOPK}: {recall:.4f}")
        assert recall >= RECALL_THRESHOLD, f"langchain recall is too low: {recall:.4f}"
        print("PASS: LangChain integration searches the disk-built index.")


def build_disk_collection(
    root: str | Path,
    name: str,
    vectors: np.ndarray,
    *,
    metric: str = "l2",
    documents: list[str] | None = None,
    ids: list[str] | None = None,
    metadatas: list[dict] | None = None,
    build_kwargs: dict | None = None,
) -> Path:
    root = Path(root)
    collection_dir = root / name
    collection_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    data_path = collection_dir / "base.bin"
    index_prefix = collection_dir / "index"
    bin_write(vectors, data_path)

    index = IndexPipeANN(vectors.shape[1], "float32", Metric.from_str(metric))
    index.omp_set_num_threads(4)
    kwargs = {
        "max_nbrs": 48,
        "build_L": 120,
        "PQ_bytes": 16,
        "memory_use_GB": 1,
    }
    if build_kwargs:
        kwargs.update(build_kwargs)
    index.build(str(data_path), str(index_prefix), build_mem_index=False, **kwargs)

    documents = documents or [f"document {i}" for i in range(len(vectors))]
    ids = ids or [str(i) for i in range(len(vectors))]
    metadatas = metadatas or [{"idx": i} for i in range(len(vectors))]

    db = sqlite3.connect(collection_dir / "documents.db")
    db.execute(
        """
        CREATE TABLE documents (
            id       TEXT PRIMARY KEY,
            tag      INTEGER UNIQUE NOT NULL,
            document TEXT,
            metadata TEXT
        )
        """
    )
    db.execute("CREATE INDEX idx_tag ON documents (tag)")
    db.executemany(
        "INSERT INTO documents (id, tag, document, metadata) VALUES (?, ?, ?, ?)",
        [
            (uid, tag, document, json.dumps(metadata, ensure_ascii=False))
            for tag, (uid, document, metadata) in enumerate(
                zip(ids, documents, metadatas, strict=True)
            )
        ],
    )
    db.commit()
    db.close()

    schema = {
        "type": "collection",
        "config": {
            "data_dim": int(vectors.shape[1]),
            "data_type": "float32",
            "metric": metric,
        },
        "attr_indexes": {},
    }
    (collection_dir / "schema.json").write_text(
        json.dumps(schema, indent=4), encoding="utf-8"
    )
    return collection_dir


def test_langchain_recall() -> None:
    main()


if __name__ == "__main__":
    main()
