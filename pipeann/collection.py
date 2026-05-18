"""Collection: document + metadata layer on top of IndexPipeANN.

Wraps a single ``IndexPipeANN`` and pairs it with an SQLite database that
stores user-facing IDs, document text, and JSON metadata.  The underlying
index is *not* modified — this module only adds a Python-level abstraction.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filter import AttrsVec, NativeAttrIndex, Selector
from .index import IndexPipeANN, Metric

__all__ = ["Collection"]

# Sentinel for "no database path yet".
_NO_DB = ":memory:"


class Collection:
    """A named collection of documents backed by a PipeANN disk index.

    Parameters
    ----------
    name : str
        Human-readable collection name (also used as directory name on disk).
    data_dim, data_type, metric : index identity parameters.
    """

    def __init__(self, name: str, data_dim: int = 0,
                 data_type: str = "float32", metric: str = "l2") -> None:
        self._name = name
        self._data_dim = data_dim
        self._data_type = data_type
        self._metric = metric
        self._index: Optional[IndexPipeANN] = None
        self._db_lock = threading.RLock()
        self._db: sqlite3.Connection = sqlite3.connect(_NO_DB, check_same_thread=False)
        self._next_tag: int = 0
        self._attr_indexes: Dict[int, dict] = {}  # key -> {filename, attr_type}
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._db_lock:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id       TEXT PRIMARY KEY,
                    tag      INTEGER UNIQUE NOT NULL,
                    document TEXT,
                    metadata TEXT
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tag ON documents (tag)"
            )
            self._db.commit()

    def _open_db(self, path: str) -> None:
        """Open (or create) a file-backed SQLite database at *path*."""
        with self._db_lock:
            self._db.close()
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._init_db()

    def _restore_next_tag(self) -> None:
        with self._db_lock:
            row = self._db.execute("SELECT MAX(tag) FROM documents").fetchone()
        self._next_tag = (row[0] + 1) if row[0] is not None else 0

    def _tags_to_docs(self, tags: list[int]) -> dict:
        """Fetch documents for a list of tags, preserving order."""
        if not tags:
            return {"id": [], "document": [], "metadata": []}
        placeholders = ",".join("?" * len(tags))
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT tag, id, document, metadata FROM documents WHERE tag IN ({placeholders})",
                tags,
            ).fetchall()
        lookup = {r[0]: r for r in rows}
        ids, docs, metas = [], [], []
        for t in tags:
            r = lookup.get(t)
            if r is not None:
                ids.append(r[1])
                docs.append(r[2])
                metas.append(json.loads(r[3]) if r[3] else {})
            else:
                ids.append(None)
                docs.append(None)
                metas.append({})
        return {"id": ids, "document": docs, "metadata": metas}

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _ensure_index(self) -> IndexPipeANN:
        if self._index is None:
            self._index = IndexPipeANN(self._data_dim, self._data_type,
                                       Metric.from_str(self._metric))
        return self._index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # -- Insert / Upsert -----------------------------------------------

    def insert(
        self,
        items: List[Tuple[str, str, np.ndarray, dict]],
        attrs: AttrsVec | Sequence | None = None,
    ) -> None:
        """Batch-insert documents.

        Parameters
        ----------
        items : list of (id, document, embedding, metadata)
            Each *id* must be unique within the collection.  *embedding* is a
            1-D numpy array.  *metadata* is a JSON-serialisable dict.
        attrs : AttrsVec or sequence, optional
            Per-vector attributes for filtered search (same length as *items*).
        """
        if not items:
            return

        # Infer data_dim from the first embedding if not yet set.
        if self._data_dim == 0:
            self._data_dim = len(items[0][2])

        vectors = np.array([it[2] for it in items], dtype=np.dtype(self._data_type))
        tags = np.empty(len(items), dtype=np.uint32)

        with self._db_lock:
            rows_to_insert = []
            for i, (uid, doc, _emb, meta) in enumerate(items):
                tag = self._next_tag
                self._next_tag += 1
                tags[i] = tag
                rows_to_insert.append((uid, tag, doc, json.dumps(meta, ensure_ascii=False)))

            self._db.executemany(
                "INSERT INTO documents (id, tag, document, metadata) VALUES (?, ?, ?, ?)",
                rows_to_insert,
            )
            self._db.commit()

        index = self._ensure_index()
        index.add(vectors, tags, attrs)

    def upsert(
        self,
        items: List[Tuple[str, str, np.ndarray, dict]],
        attrs: AttrsVec | Sequence | None = None,
    ) -> None:
        """Insert new items or update existing ones.

        For existing IDs the old vector is removed and a new one inserted.
        """
        if not items:
            return

        with self._db_lock:
            existing_ids: set = set()

            # Identify which IDs already exist.
            for idx, (uid, *_rest) in enumerate(items):
                row = self._db.execute(
                    "SELECT tag FROM documents WHERE id = ?", (uid,)
                ).fetchone()
                if row is not None:
                    existing_ids.add(uid)

            # Delete existing entries first.
            if existing_ids:
                self.delete_by_id(list(existing_ids))

            # Now insert everything (all are "new" after deletion).
            self.insert(items, attrs)

    # -- Search --------------------------------------------------------

    def search(
        self,
        queries: np.ndarray,
        topk: int,
        L: int,
        selector: Selector | None = None,
        query_attrs: AttrsVec | Sequence | None = None,
    ) -> dict:
        """Vector similarity search.

        Returns
        -------
        dict
            Keys: ``id``, ``document``, ``metadata``, ``distance``.
            Each value is a list-of-lists (one inner list per query).
        """
        if self._index is None:
            empty: dict = {"id": [], "document": [], "metadata": [], "distance": []}
            return empty

        tags_arr, dists_arr = self._index.search(
            np.ascontiguousarray(queries, dtype=np.dtype(self._data_type)),
            topk,
            L,
            selector=selector,
            query_attrs=query_attrs,
        )

        ret: dict = {"id": [], "document": [], "metadata": [], "distance": []}

        for row_tags, row_dists in zip(tags_arr, dists_arr):
            doc_info = self._tags_to_docs(row_tags.tolist())
            ret["id"].append(doc_info["id"])
            ret["document"].append(doc_info["document"])
            ret["metadata"].append(doc_info["metadata"])
            ret["distance"].append(row_dists.tolist())

        return ret

    # -- Delete --------------------------------------------------------

    def delete_by_id(self, ids: List[str]) -> None:
        """Remove documents by user-facing IDs."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT tag FROM documents WHERE id IN ({placeholders})", ids
            ).fetchall()
            if not rows:
                return
            tags = np.array([r[0] for r in rows], dtype=np.uint32)

            self._db.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})", ids
            )
            self._db.commit()

        if self._index is not None:
            self._index.remove(tags)

    # -- Get -----------------------------------------------------------

    def get_by_id(self, ids: List[str]) -> dict:
        """Fetch documents by user-facing IDs."""
        if not ids:
            return {"id": [], "document": [], "metadata": []}
        placeholders = ",".join("?" * len(ids))
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT id, document, metadata FROM documents WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        result: dict = {"id": [], "document": [], "metadata": []}
        lookup = {r[0]: r for r in rows}
        for uid in ids:
            r = lookup.get(uid)
            if r is not None:
                result["id"].append(r[0])
                result["document"].append(r[1])
                result["metadata"].append(json.loads(r[2]) if r[2] else {})
        return result

    def filter_query(
        self, metadata_filter: dict, limit: Optional[int] = None
    ) -> dict:
        """Filter documents by metadata key-value pairs (exact match).

        This is a Python-side scan on the SQLite table (suitable for
        moderate-size collections).
        """
        with self._db_lock:
            rows = self._db.execute(
                "SELECT id, document, metadata FROM documents"
            ).fetchall()
        result: dict = {"id": [], "document": [], "metadata": []}
        for uid, doc, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            if all(meta.get(k) == v for k, v in metadata_filter.items()):
                result["id"].append(uid)
                result["document"].append(doc)
                result["metadata"].append(meta)
                if limit is not None and len(result["id"]) >= limit:
                    break
        return result

    # -- Count ---------------------------------------------------------

    def npoints(self) -> int:
        """Return the number of documents in the collection."""
        with self._db_lock:
            row = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0]

    # -- Build (large-scale) -------------------------------------------

    def build(self, data_path: str, index_prefix: str, **build_kwargs) -> None:
        """Build a disk index from a binary vector file.

        Delegates directly to ``IndexPipeANN.build``.  After building you
        still need to call ``load`` on the index (handled automatically by
        ``Collection.save`` / ``Collection.load``).
        """
        index = self._ensure_index()
        index.build(data_path, index_prefix, **build_kwargs)

    # -- Attribute index management ------------------------------------

    def load_attr_index(
        self, key: int, filename: str, attr_type: str
    ) -> NativeAttrIndex:
        """Load a native attribute index and record it in the schema."""
        if self._index is None:
            raise RuntimeError("Index not loaded — call load() or insert() first")
        self._attr_indexes[key] = {"filename": filename, "attr_type": attr_type}
        return self._index.load_attr_index_from_file(key, filename, attr_type)

    # -- Persistence ---------------------------------------------------

    def save(self, url: str) -> dict:
        """Persist the collection to *url*/{name}/.

        Returns the schema dict (also written to ``schema.json``).
        """
        col_dir = os.path.join(url, self._name)
        os.makedirs(col_dir, exist_ok=True)

        # Persist index files.
        index_prefix = os.path.join(col_dir, "index")
        if self._index is not None:
            self._index.save(index_prefix)

        # Ensure SQLite is file-backed inside the collection dir.
        db_path = os.path.join(col_dir, "documents.db")
        if self._db_is_memory():
            self._persist_memory_db(db_path)

        # Write schema.
        schema = {
            "type": "collection",
            "config": {"data_dim": self._data_dim, "data_type": self._data_type, "metric": self._metric},
            "attr_indexes": {
                str(k): v for k, v in self._attr_indexes.items()
            },
        }
        with open(os.path.join(col_dir, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4)
        return schema

    @classmethod
    def load(cls, url: str, name: str) -> Collection:
        """Load a collection previously saved at *url*/{name}/."""
        col_dir = os.path.join(url, name)
        if not os.path.isdir(col_dir):
            raise FileNotFoundError(f"Collection directory not found: {col_dir}")

        with open(os.path.join(col_dir, "schema.json"), encoding="utf-8") as f:
            schema = json.load(f)
        if schema.get("type") != "collection":
            raise ValueError(f"{name} is not a collection (type={schema.get('type')!r})")

        config = schema["config"]
        instance = cls(name, data_dim=config["data_dim"], 
                       data_type=config["data_type"],
                       metric=config["metric"])

        # Open the file-backed SQLite database.
        db_path = os.path.join(col_dir, "documents.db")
        if os.path.isfile(db_path):
            instance._open_db(db_path)
            instance._restore_next_tag()

        # Load the PipeANN index.
        index_prefix = os.path.join(col_dir, "index")
        index = instance._ensure_index()
        index.load(index_prefix)

        # Restore attr_indexes metadata.
        instance._attr_indexes = {
            int(k): v for k, v in schema.get("attr_indexes", {}).items()
        }

        return instance

    # -- Private helpers ------------------------------------------------

    def _db_is_memory(self) -> bool:
        """Return True if the current database is in-memory."""
        # pragma_database_list returns (seq, name, file).  file=="" for :memory:.
        with self._db_lock:
            row = self._db.execute("PRAGMA database_list").fetchone()
        return row is None or row[2] == "" or row[2] == _NO_DB

    def _persist_memory_db(self, path: str) -> None:
        """Dump the in-memory database to a file, then re-open from file."""
        with self._db_lock:
            file_db = sqlite3.connect(path)
            self._db.backup(file_db)
            file_db.close()
            self._db.close()
            self._db = sqlite3.connect(path, check_same_thread=False)
