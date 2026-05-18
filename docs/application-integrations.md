---
hide:
  - navigation
---

# Application Integrations

PipeANN includes high-level interfaces for application developers. The `Client` and `Collection` APIs manage vectors together with document text and metadata, while the integration layer also provides LangChain vector-store support and a Qdrant-compatible API surface for easier adoption in existing applications such as Open WebUI.

## Collection API

Collection config and loaded attribute-index metadata are persisted to `schema.json` automatically when you call `save()`.

```python
from pipeann import Client
import numpy as np

# Create a client (auto-discovers existing collections on disk)
client = Client(url="/path/to/data")

# Create a collection
col = client.create_collection("my_collection", data_dim=128, metric="l2")

# Insert items: (id, document, embedding, metadata)
items = [
    ("doc1", "PipeANN is an SSD-backed vector store", np.random.rand(128).astype(np.float32), {"source": "PipeANN"}),
    ("doc2", "LangChain wraps retrievers", np.random.rand(128).astype(np.float32), {"source": "LangChain"}),
]
col.insert(items)

# Search
queries = np.random.rand(1, 128).astype(np.float32)
results = col.search(queries, topk=10, L=50)
# Returns: {"id": [[...]], "document": [[...]], "metadata": [[...]], "distance": [[...]]}

# Get by ID
docs = col.get_by_id(["doc1"])

# Filter by metadata (exact-match scan in SQLite)
matched = col.filter_query({"source": "PipeANN"})

# Delete
col.delete_by_id(["doc2"])

# Upsert (insert or update)
col.upsert(items)

# Save to disk and reload
client.save_collection("my_collection")
# On next Client(url=...) init, the collection is auto-loaded.

# Cleanup
client.delete_collection("my_collection", delete_on_disk=True)
client.reset(delete_on_disk=True)
```

Run the example tests:
```bash
pip install -e .
python tests_py/collection_example.py
```

## LangChain API

Install LangChain yourself if you want to use `pipeann.langchain`:

```bash
pip install langchain-core
```

```python
import pipeann  # Load PipeANN before LangChain modules in current builds.
from langchain_openai import OpenAIEmbeddings
from pipeann.langchain import PipeANNVectorStore

store = PipeANNVectorStore.from_texts(
    ["PipeANN is an SSD-backed vector store", "LangChain wraps retrievers"],
    OpenAIEmbeddings(),
    metadatas=[{"source": "pipeann"}, {"source": "langchain"}],
    data_dim=1536,
    metric="l2",
    search_L=64,
)

docs = store.similarity_search("SSD vector search", k=2)
retriever = store.as_retriever(search_kwargs={"k": 4, "L": 128})
```

You can also wrap an existing `Collection` with
`PipeANNVectorStore.from_collection(collection, embeddings)`.

## Qdrant-compatible API

PipeANN provides built-in Qdrant backend. Running it also requires:

```bash
pip install fastapi uvicorn
```

Start PipeANN's Qdrant-compatible HTTP server:

```bash
PIPEANN_DATA_DIR=/path/to/pipeann-data \
PIPEANN_SEARCH_L=64 \
PIPEANN_QDRANT_PORT=6333 \
python -m pipeann.qdrant_server
```

You can smoke-test the endpoint with `qdrant-client`:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://127.0.0.1:6333")
client.create_collection("open-webui_docs", VectorParams(size=1536, distance=Distance.COSINE))
client.upsert("open-webui_docs", [
    PointStruct(
        id="doc-1",
        vector=embedding,
        payload={"text": "PipeANN stores Open WebUI chunks.", "metadata": {"file_id": "demo"}},
    )
])

result = client.query_points("open-webui_docs", query=query_embedding, limit=5)
```

Write operations do not automatically save the PipeANN index to disk. Persist explicitly when needed:

```bash
curl -X POST http://127.0.0.1:6333/collections/open-webui_docs/save
```

The server implements the Qdrant REST subset for collection existence /
creation / deletion, payload index creation, point upsert, vector query,
scroll, filter delete, and count.

## Open WebUI

Open WebUI's built-in Qdrant backend can target PipeANN directly:

```bash
VECTOR_DB=qdrant
QDRANT_URI=http://127.0.0.1:6333
ENABLE_QDRANT_MULTITENANCY_MODE=false
```
