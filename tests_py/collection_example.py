"""Example / smoke test for the Collection and Client API layers.

This script uses synthetic random vectors so it can run anywhere without
external data files.  It exercises:

  1. Client + Collection creation
  2. insert / search / get_by_id / delete_by_id / upsert
  3. filter_query (metadata filtering)
  4. save + reload from disk
  5. search after reload (persistence round-trip)
"""

import os
import shutil
import tempfile
import numpy as np

from pipeann import Client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 96
N_BASE = 2000
N_QUERY = 10
TOPK = 5
L_SEARCH = 30
DATA_TYPE = "float32"


def random_vectors(n: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(42).random((n, dim), dtype=np.float32)


def make_items(n: int, offset: int = 0):
    """Return a list of (id, document, embedding, metadata) tuples."""
    vecs = random_vectors(n)
    items = []
    for i in range(n):
        uid = f"doc-{offset + i}"
        doc = f"This is document number {offset + i}."
        meta = {"category": "A" if (offset + i) % 2 == 0 else "B",
                "value": offset + i}
        items.append((uid, doc, vecs[i], meta))
    return items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_crud(tmp_dir: str) -> None:
    """Create → insert → search → get → delete → upsert."""
    print("=== test_basic_crud ===")

    client = Client(url=tmp_dir)
    col = client.create_collection("test1", data_dim=DIM, metric="l2",
                                   data_type=DATA_TYPE)
    assert col is not None
    assert client.list_collections() == ["test1"]
    print(f"  Created collection, npoints={col.npoints()}")

    # Insert
    items = make_items(N_BASE)
    col.insert(items)
    assert col.npoints() == N_BASE
    print(f"  Inserted {N_BASE} items, npoints={col.npoints()}")

    # Search
    queries = random_vectors(N_QUERY)
    results = col.search(queries, topk=TOPK, L=L_SEARCH)
    assert len(results["id"]) == N_QUERY
    assert len(results["id"][0]) == TOPK
    assert results["document"][0][0] is not None
    print(f"  Search OK — first result: id={results['id'][0][0]}, "
          f"dist={results['distance'][0][0]:.4f}")

    # get_by_id
    fetched = col.get_by_id(["doc-0", "doc-1", "nonexistent"])
    assert "doc-0" in fetched["id"]
    assert "doc-1" in fetched["id"]
    assert "nonexistent" not in fetched["id"]
    print(f"  get_by_id OK — got {len(fetched['id'])} docs")

    # delete_by_id
    col.delete_by_id(["doc-0", "doc-1"])
    assert col.npoints() == N_BASE - 2
    fetched2 = col.get_by_id(["doc-0"])
    assert len(fetched2["id"]) == 0
    print(f"  delete_by_id OK — npoints={col.npoints()}")

    # upsert (update existing + insert new)
    upsert_items = [
        ("doc-2", "Updated doc 2", random_vectors(1)[0], {"category": "C", "value": 2}),
        ("doc-new-1", "Brand new doc", random_vectors(1)[0], {"category": "D", "value": 9999}),
    ]
    col.upsert(upsert_items)
    fetched3 = col.get_by_id(["doc-2", "doc-new-1"])
    assert "doc-2" in fetched3["id"]
    assert "doc-new-1" in fetched3["id"]
    assert fetched3["document"][fetched3["id"].index("doc-2")] == "Updated doc 2"
    print(f"  upsert OK — npoints={col.npoints()}")

    print("  PASSED\n")


def test_filter_query(tmp_dir: str) -> None:
    """Metadata filtering via filter_query."""
    print("=== test_filter_query ===")

    client = Client(url=tmp_dir)
    col = client.get_or_create_collection("test_filter", data_dim=DIM,
                                          metric="l2", data_type=DATA_TYPE)
    items = make_items(100)
    col.insert(items)

    # Filter: category == "A"  (even-numbered docs)
    filtered = col.filter_query({"category": "A"})
    assert all(m["category"] == "A" for m in filtered["metadata"])
    assert len(filtered["id"]) == 50  # half of 100
    print(f"  filter_query(category=A) returned {len(filtered['id'])} docs")

    # Filter with limit
    limited = col.filter_query({"category": "A"}, limit=5)
    assert len(limited["id"]) == 5
    print(f"  filter_query with limit=5 returned {len(limited['id'])} docs")

    print("  PASSED\n")


def test_persistence(tmp_dir: str) -> None:
    """Save → reload should preserve data and search results."""
    print("=== test_persistence ===")

    # Create and populate
    client = Client(url=tmp_dir)
    col = client.create_collection("persist_test", data_dim=DIM, metric="l2",
                                   data_type=DATA_TYPE)
    items = make_items(N_BASE)
    col.insert(items)
    print(f"  Inserted {N_BASE} items")

    queries = random_vectors(N_QUERY)
    persist_L = max(L_SEARCH, 512)
    results_before = col.search(queries, topk=TOPK, L=persist_L)

    # Save
    client.save_collection("persist_test")
    print("  Saved collection to disk")

    # Reload from scratch
    client2 = Client(url=tmp_dir)
    assert "persist_test" in client2.list_collections()
    col2 = client2.get_collection("persist_test")
    assert col2 is not None
    assert col2.npoints() == N_BASE
    print(f"  Reloaded collection, npoints={col2.npoints()}")

    # Search after reload with a large enough L to make the traversal stable.
    results_after = col2.search(queries, topk=TOPK, L=persist_L)
    assert results_before == results_after
    print("  Search results match after reload")

    # Verify document content survived
    fetched = col2.get_by_id(["doc-0", "doc-42"])
    assert "doc-0" in fetched["id"]
    assert "doc-42" in fetched["id"]
    print("  Document content survived persistence round-trip")

    print("  PASSED\n")


def test_client_lifecycle(tmp_dir: str) -> None:
    """Client: create / list / delete / reset."""
    print("=== test_client_lifecycle ===")

    client = Client(url=tmp_dir)
    client.create_collection("col_a", data_dim=DIM, metric="l2",
                             data_type=DATA_TYPE)
    client.create_collection("col_b", data_dim=DIM, metric="l2",
                             data_type=DATA_TYPE)
    assert set(client.list_collections()) == {"col_a", "col_b"}
    print(f"  Created 2 collections: {client.list_collections()}")

    # Duplicate name should fail
    try:
        client.create_collection("col_a")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  Duplicate name correctly rejected")

    # Delete one
    client.delete_collection("col_b")
    assert client.list_collections() == ["col_a"]
    print("  Deleted col_b")

    # Reset
    client.reset()
    assert client.list_collections() == []
    print("  Reset OK — no collections left")

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tmp_dir = tempfile.mkdtemp(prefix="pipeann_collection_test_")
    print(f"Using temp directory: {tmp_dir}\n")
    try:
        test_basic_crud(os.path.join(tmp_dir, "crud"))
        test_filter_query(os.path.join(tmp_dir, "filter"))
        test_persistence(os.path.join(tmp_dir, "persist"))
        test_client_lifecycle(os.path.join(tmp_dir, "lifecycle"))
        print("=" * 40)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nCleaned up {tmp_dir}")


if __name__ == "__main__":
    main()
