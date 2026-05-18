"""Qdrant-compatible HTTP server backed by PipeANN."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

from .client import Client

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:
    raise ImportError(
        "PipeANN Qdrant-compatible server requires FastAPI. "
        "Install it with `pip install fastapi uvicorn`."
    ) from exc

NO_LIMIT = 999999999
OP_DONE = {"operation_id": 0, "status": "completed"}


def create_app(data_dir: Optional[str] = None, search_L: Optional[int] = None) -> FastAPI:
    data_dir = data_dir or os.environ.get("PIPEANN_DATA_DIR", os.path.abspath("./data"))
    search_L = search_L or int(os.environ.get("PIPEANN_SEARCH_L", "50"))
    omp_threads = int(os.environ.get("PIPEANN_QDRANT_OMP_THREADS", "1"))
    upsert_on_write = os.environ.get("PIPEANN_QDRANT_UPSERT_ON_WRITE", "false").lower() == "true"
    query_threads = int(os.environ.get("PIPEANN_QDRANT_QUERY_THREADS", "16"))
    os.makedirs(data_dir, exist_ok=True)

    client = Client(url=data_dir)
    query_pool = ThreadPoolExecutor(max_workers=query_threads)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            query_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="PipeANN Qdrant-compatible server",
        version="0.1.0",
        lifespan=lifespan,
    )

    def collection_or_404(name: str):
        collection = client.get_collection(name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection {name!r} not found")
        _set_threads(collection, omp_threads)
        return collection

    @app.get("/")
    def root():
        return {"title": "qdrant - vector search engine", "version": "pipeann"}

    @app.get("/collections")
    def get_collections():
        return _ok({"collections": [{"name": name} for name in client.list_collections()]})

    @app.get("/collections/{collection_name}")
    def get_collection(collection_name: str):
        collection = collection_or_404(collection_name)
        npoints = collection.npoints()
        return _ok({"status": "green", "vectors_count": npoints, "points_count": npoints})

    @app.get("/collections/{collection_name}/exists")
    def collection_exists(collection_name: str):
        return _ok({"exists": client.get_collection(collection_name) is not None})

    @app.put("/collections/{collection_name}")
    def create_collection(collection_name: str, request: dict[str, Any]):
        body = request
        if client.get_collection(collection_name) is None:
            data_dim, metric = _collection_config(body)
            client.create_collection(collection_name, data_dim=data_dim, metric=metric)
            _set_threads(client.get_collection(collection_name), omp_threads)
        return _ok(True)

    @app.delete("/collections/{collection_name}")
    def delete_collection(collection_name: str):
        if client.get_collection(collection_name) is not None:
            client.delete_collection(collection_name, delete_on_disk=True)
        return _ok(True)

    @app.put("/collections/{collection_name}/index")
    def create_payload_index(collection_name: str):
        collection_or_404(collection_name)
        return _ok(OP_DONE)

    @app.put("/collections/{collection_name}/points")
    @app.post("/collections/{collection_name}/points")
    def upsert_points(collection_name: str, request: dict[str, Any]):
        collection = collection_or_404(collection_name)
        items = _items_from_body(request)
        if upsert_on_write:
            collection.upsert(items)
        else:
            collection.insert(items)
        return _ok(OP_DONE)

    def query_points_result(collection_name: str, body: dict[str, Any]) -> dict[str, Any]:
        limit = _limit(body)
        collection = collection_or_404(collection_name)
        metadata_filter = _metadata_filter(body.get("filter") or body.get("query_filter"))
        topk = collection.npoints() if metadata_filter else limit
        query = np.asarray([_query_vector(body)], dtype=np.float32)
        result = collection.search(query, topk=topk, L=max(topk, search_L))
        points = _search_points(result, metadata_filter, limit)
        return _ok({"points": points})

    @app.post("/collections/{collection_name}/points/query")
    async def query_points(collection_name: str, request: dict[str, Any]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(query_pool, query_points_result, collection_name, request)

    @app.post("/collections/{collection_name}/points/search")
    async def search_points(collection_name: str, request: dict[str, Any]):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(query_pool, query_points_result, collection_name, request)
        return _ok(response["result"]["points"])

    @app.post("/collections/{collection_name}/points/scroll")
    def scroll_points(collection_name: str, request: dict[str, Any]):
        collection = collection_or_404(collection_name)
        body = request
        limit = _limit(body)
        metadata_filter = _metadata_filter(body.get("filter") or body.get("scroll_filter"))
        result = collection.filter_query(metadata_filter, limit=None if limit >= NO_LIMIT else limit)
        return _ok({"points": _records(result), "next_page_offset": None})

    @app.post("/collections/{collection_name}/points/delete")
    def delete_points(collection_name: str, request: dict[str, Any]):
        collection = collection_or_404(collection_name)
        body = request
        selector = body.get("points") or body.get("points_selector") or body
        ids = _ids_from_selector(selector)
        if ids:
            collection.delete_by_id(ids)
        else:
            matched = collection.filter_query(_metadata_filter(_selector_filter(selector)))
            if matched["id"]:
                collection.delete_by_id(matched["id"])
        return _ok(OP_DONE)

    @app.post("/collections/{collection_name}/save")
    def save_collection(collection_name: str):
        collection_or_404(collection_name)
        client.save_collection(collection_name)
        return _ok(OP_DONE)

    @app.post("/collections/{collection_name}/points/count")
    def count_points(collection_name: str, request: dict[str, Any]):
        collection = collection_or_404(collection_name)
        body = request
        metadata_filter = _metadata_filter(body.get("filter") or body.get("count_filter"))
        return _ok({"count": len(collection.filter_query(metadata_filter)["id"])})

    @app.post("/reset")
    def reset():
        client.reset(delete_on_disk=True)
        return _ok(True)

    return app


def _ok(result: Any) -> dict[str, Any]:
    return {"result": result, "status": "ok", "time": 0.0}


def _set_threads(collection: Any, omp_threads: int) -> None:
    if collection is not None and getattr(collection, "_index", None) is not None:
        collection._index.omp_set_num_threads(omp_threads)


def _limit(body: dict[str, Any]) -> int:
    return int(body.get("limit", 10) or NO_LIMIT)


def _collection_config(body: dict[str, Any]) -> tuple[int, str]:
    vectors_config = body.get("vectors") or body.get("vectors_config") or {}
    if not isinstance(vectors_config, dict):
        return 0, "cosine"
    data_dim = int(vectors_config.get("size", 0))
    distance = str(vectors_config.get("distance", "Cosine")).lower()
    return data_dim, "cosine" if distance == "cosine" else "l2"


def _items_from_body(body: dict[str, Any]) -> list[tuple[str, str, np.ndarray, dict]]:
    points = body.get("batch") or body.get("points") or body
    if isinstance(points, dict) and "ids" in points:
        return [
            _item(uid, vector, payload or {})
            for uid, vector, payload in zip(
                points.get("ids", []),
                points.get("vectors", []),
                points.get("payloads", []),
            )
        ]
    return [_item_from_point(point) for point in (points or [])]


def _query_vector(body: dict[str, Any]) -> list[float]:
    query = body.get("query") or body.get("vector")
    if isinstance(query, dict):
        return query.get("nearest") or query.get("vector") or query.get("values") or []
    return query or []


def _item_from_point(point: dict[str, Any]) -> tuple[str, str, np.ndarray, dict]:
    return _item(point["id"], point["vector"], point.get("payload") or {})


def _item(uid: Any, vector: Any, payload: dict[str, Any]) -> tuple[str, str, np.ndarray, dict]:
    return (
        str(uid),
        payload.get("text") or payload.get("document") or "",
        np.asarray(vector, dtype=np.float32),
        payload.get("metadata") or {},
    )


def _search_points(
    result: dict,
    metadata_filter: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    points = []
    for uid, document, metadata, distance in zip(
        result["id"][0],
        result["document"][0],
        result["metadata"][0],
        result["distance"][0],
    ):
        if uid is None or document is None or not _matches(metadata, metadata_filter):
            continue
        points.append(_point(uid, document, metadata, score=1.0 - float(distance)))
        if len(points) >= limit:
            break
    return points


def _records(result: dict) -> list[dict[str, Any]]:
    return [
        _point(uid, document, metadata)
        for uid, document, metadata in zip(result["id"], result["document"], result["metadata"])
    ]


def _point(
    uid: str,
    document: str,
    metadata: dict,
    score: float | None = None,
) -> dict[str, Any]:
    point = {
        "id": uid,
        "payload": {"text": document, "metadata": metadata},
        "vector": None,
    }
    if score is not None:
        point.update({"version": 0, "score": score})
    return point


def _ids_from_selector(selector: Any) -> list[str]:
    if isinstance(selector, list):
        return [str(uid) for uid in selector]
    if isinstance(selector, dict) and isinstance(selector.get("points"), list):
        return [str(uid) for uid in selector["points"]]
    return []


def _selector_filter(selector: Any) -> Any:
    return selector.get("filter") if isinstance(selector, dict) else selector


def _metadata_filter(qdrant_filter: Any) -> dict[str, Any]:
    if not isinstance(qdrant_filter, dict):
        return {}
    conditions = (qdrant_filter.get("must") or []) + (qdrant_filter.get("should") or [])
    metadata_filter = {}
    for condition in conditions:
        key = condition.get("key") if isinstance(condition, dict) else None
        match = condition.get("match") if isinstance(condition, dict) else None
        if key and key.startswith("metadata.") and isinstance(match, dict):
            metadata_filter[key[len("metadata."):]] = match.get("value")
    return metadata_filter


def _matches(metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Running PipeANN's Qdrant-compatible server requires uvicorn. "
            "Install it with `pip install fastapi uvicorn`."
        ) from exc

    host = os.environ.get("PIPEANN_QDRANT_HOST", "0.0.0.0")
    port = int(os.environ.get("PIPEANN_QDRANT_PORT", "6333"))
    uvicorn.run(
        "pipeann.qdrant_server:app",
        host=host,
        port=port,
        reload=False,
        access_log=os.environ.get("PIPEANN_QDRANT_ACCESS_LOG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
