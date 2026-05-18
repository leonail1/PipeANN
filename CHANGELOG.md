# Changelog

## v0.3.0 (2026-05-18)

It adds new search capabilities first, then the Python-facing stack built on top of them, plus the refactors and tests needed to support updates and filtering consistently.

### Feature highlights

#### Speculative filtering

- Added speculative filtered ANNS for arbitrary attribute constraints.
- Uses lightweight in-memory probabilistic filters to explore a superset of valid vectors, then verifies final candidates exactly against SSD-resident attributes.
- A cost model chooses speculative pre-filtering, speculative in-filtering, or post-filtering per query.
- Supports label filters, range filters `[l, r)`, and Boolean combinations through native selectors.
- Added typed attributes (`Attributes`, `AttrsVec`), on-disk attribute indexes, dense-neighbor support (`range_dense`), and JSON filter config loading.

#### OOD refinement

- Added NGFix-style out-of-distribution graph refinement.
- Exposed `train_query_path`, `R_ood`, and `L_ood` in C++ build tools and `IndexPipeANN.build()`.
- Persisted OOD metadata in SSD index metadata.

#### Range search

- Added finite-threshold range search in C++ and Python.
- Results outside the threshold are filtered out and padded with `UINT32_MAX` / `inf`.
- Reuses the common pipelined traversal and result-copy path.

#### SPDK backend

- Added an optional SPDK I/O backend through `-DIO_ENGINE=spdk` for raw NVMe vector reads.
- Supports RAID-0-style striping across PCIe NVMe devices listed in `spdk_bdevs.json`, with one poller thread per device.
- Copies `{index_prefix}_disk.index` to the SPDK target on first open and reuses a marker to skip repeated copies.
- Keeps filtered-search attribute reads on `io_uring` while vector I/O uses SPDK.

#### Python API and integrations

- Added the current `IndexPipeANN(data_dim, data_type, metric)` API for build, load, search, insert, delete, save, filters, range search, and attribute-aware inserts.
- Added `Collection` and `Client` for SQLite-backed documents/payloads, vector CRUD, persistence, and collection auto-discovery.
- Added LangChain integration through `pipeann.langchain.PipeANNVectorStore`.
- Added a Qdrant-compatible FastAPI server with collection management, point upsert/query/scroll, payload indexes, filter delete, count, and save.
- Added `schema.json` persistence for collection config and attribute-index metadata.

### Code refactoring and implementation changes

- Merged dynamic search, insert, delete, merge, and save behavior into header-only `DynamicIndex<T>`.
- Unified update save/merge with same-prefix double-version replacement.
- Merged PiPNN and Vamana-style build paths behind `build_disk_index` and the shared SSD file format.
- Refactored pipelined top-k, range, and filtered search around `pipe_search_common.h` and `spec_filter_search.cpp`.
- Simplified metric/distance handling and updated PiPNN file layout.
- Added `IO_ENGINE` selection for `uring`, `aio`, and `spdk`, separate dense-node I/O sizing, CMake/CI cleanup, and package version `0.3.0`.

### Tests and examples

- Switched the project license from MIT to Apache License 2.0 and updated NOTICE attribution, including NGFix graph-refinement logic under MIT.
- Added Python insert/delete search tests, including assertions that results include inserted data and exclude deleted data.
- Added filtered insert regression tests comparing full build vs. build-plus-insert with attributes.
- Added native selector, range-search, collection, LangChain, and Qdrant server tests/examples.
- Added C++ filtered build/search and update test coverage.

### Breaking changes

- `DynamicSSDIndex` was removed; use `DynamicIndex<T>` in C++ or `IndexPipeANN` in Python.
- `dynamic_index.cpp` was removed; `DynamicIndex<T>` is now header-only.
- `filter/label.h` was removed; use `filter/attribute.h` and `filter/selector.h`.
- `build_pipnn_index` was removed; use `build_disk_index` with PiPNN-compatible parameters.
- The old schema-centered Python API was removed; construct `IndexPipeANN(data_dim, data_type, metric)` directly and let `Collection` / `Client` own collection persistence.
- `PyIndexInterface` construction changed from a Python dict to explicit `(dim, dtype, metric)` arguments.
- `IndexPipeANN.search()` now takes `selector`, `query_attrs`, and finite `range` keyword arguments.
- `IndexPipeANN.build()` now takes `attrs`, `range_dense`, `train_query_path`, `R_ood`, and `L_ood`; major build knobs auto-configure when left at `0`.
- `pipnn.h` / `pipnn.cpp` moved under the utils layout.
