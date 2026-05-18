"""Client: multi-collection manager for PipeANN.

Manages named ``Collection`` instances and their persistence to a shared base directory. 
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

import json

from .collection import Collection

__all__ = ["Client"]


class Client:
    """Top-level entry point for managing PipeANN collections.

    Parameters
    ----------
    url : str, optional
        Base directory for on-disk persistence.  If provided, existing
        collections are discovered and loaded automatically.  If *None*,
        collections live only in memory.
    """

    def __init__(self, url: Optional[str] = None) -> None:
        self._collections: Dict[str, Collection] = {}
        self._url: Optional[str] = None

        if url is not None:
            self._url = os.path.abspath(url)
            os.makedirs(self._url, exist_ok=True)
            # Auto-discover saved collections.
            for name in sorted(os.listdir(self._url)):
                full = os.path.join(self._url, name)
                schema_path = os.path.join(full, "schema.json")
                if os.path.isdir(full) and os.path.isfile(schema_path) and \
                        json.load(open(schema_path, encoding="utf-8")).get("type") == "collection":
                    self._collections[name] = Collection.load(self._url, name)

    # ------------------------------------------------------------------
    # Collection CRUD
    # ------------------------------------------------------------------

    def create_collection(self, name: str, **kwargs) -> Collection:
        """Create a new, empty collection.

        Extra *kwargs* (``data_dim``, ``data_type``, ``metric``) are forwarded
        to ``Collection``.

        Raises
        ------
        RuntimeError
            If a collection with the same *name* already exists.
        """
        if name in self._collections:
            raise RuntimeError(f"Collection {name!r} already exists")
        col = Collection(name, **kwargs)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Optional[Collection]:
        """Return the collection with *name*, or *None* if it doesn't exist."""
        return self._collections.get(name)

    def get_or_create_collection(self, name: str, **kwargs) -> Collection:
        """Return existing collection or create a new one."""
        col = self.get_collection(name)
        if col is None:
            col = self.create_collection(name, **kwargs)
        return col

    def delete_collection(
        self, name: str, delete_on_disk: bool = False
    ) -> None:
        """Remove a collection from the client.

        Parameters
        ----------
        delete_on_disk : bool
            If *True*, also delete the on-disk directory.
        """
        if name not in self._collections:
            raise RuntimeError(f"Collection {name!r} does not exist")
        del self._collections[name]
        if delete_on_disk:
            if self._url is None:
                raise RuntimeError("Client has no url — cannot delete from disk")
            col_dir = os.path.join(self._url, name)
            if os.path.isdir(col_dir):
                shutil.rmtree(col_dir)

    def list_collections(self) -> List[str]:
        """Return the names of all managed collections."""
        return list(self._collections.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_collection(self, name: str) -> None:
        """Persist a single collection to disk."""
        if self._url is None:
            raise RuntimeError("Client has no url — cannot save")
        if name not in self._collections:
            raise RuntimeError(f"Collection {name!r} does not exist")
        self._collections[name].save(self._url)

    def save_all(self) -> None:
        """Persist every collection to disk."""
        for name in self._collections:
            self.save_collection(name)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def reset(self, delete_on_disk: bool = False) -> None:
        """Drop all collections from memory (and optionally from disk)."""
        if delete_on_disk:
            if self._url is None:
                raise RuntimeError("Client has no url — cannot delete from disk")
            for name in list(self._collections):
                col_dir = os.path.join(self._url, name)
                if os.path.isdir(col_dir):
                    shutil.rmtree(col_dir)
        self._collections.clear()
