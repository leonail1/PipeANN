"""LangChain integration for PipeANN."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from .collection import Collection

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "PipeANN LangChain support requires LangChain. "
        "Install it with `pip install langchain-core`."
    ) from exc

__all__ = ["PipeANNVectorStore"]


class PipeANNVectorStore(VectorStore):
    """LangChain VectorStore backed by a PipeANN ``Collection``."""

    def __init__(
        self,
        embedding: Embeddings,
        collection: Collection | None = None,
        *,
        collection_name: str = "langchain",
        data_dim: int = 0,
        data_type: str = "float32",
        metric: str = "l2",
        search_L: int = 64,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.embedding = embedding
        self.collection = collection or Collection(
            collection_name,
            data_dim=data_dim,
            data_type=data_type,
            metric=metric,
        )
        self.search_L = search_L
        self.metric = metric
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid4()) for _ in texts]
        vectors = self.embedding.embed_documents(texts)
        items = [
            (uid, text, np.asarray(vector, dtype=np.float32), metadata)
            for uid, text, vector, metadata in zip(ids, texts, vectors, metadatas)
        ]
        self.collection.insert(items, attrs=kwargs.get("attrs"))
        return ids

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            return None
        self.collection.delete_by_id(ids)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        result = self.collection.get_by_id(list(ids))
        return self._documents_from_result(result)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score(query, k, **kwargs)
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score_by_vector(
                embedding, k, **kwargs
            )
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        L = kwargs.pop("L", self.search_L)
        selector = kwargs.pop("selector", None)
        query_attrs = kwargs.pop("query_attrs", None)
        result = self.collection.search(
            np.asarray([embedding], dtype=np.float32),
            topk=k,
            L=L,
            selector=selector,
            query_attrs=query_attrs,
        )
        documents = self._documents_from_result(
            {
                "id": result["id"][0],
                "document": result["document"][0],
                "metadata": result["metadata"][0],
            }
        )
        return list(zip(documents, result["distance"][0]))

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PipeANNVectorStore:
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_collection(
        cls,
        collection: Collection,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> PipeANNVectorStore:
        return cls(embedding=embedding, collection=collection, **kwargs)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn
        if self.metric == "cosine":
            return self._cosine_relevance_score_fn
        if self.metric == "inner_product":
            return self._max_inner_product_relevance_score_fn
        return self._euclidean_relevance_score_fn

    @staticmethod
    def _documents_from_result(result: dict) -> list[Document]:
        documents = []
        for uid, page_content, metadata in zip(
            result["id"], result["document"], result["metadata"]
        ):
            if uid is not None and page_content is not None:
                documents.append(
                    Document(page_content=page_content, metadata=metadata, id=uid)
                )
        return documents
