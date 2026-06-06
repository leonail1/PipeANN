# OpenHarmony ANNS Filtered Search Backend

This repository is a clean OpenHarmony ANNS acceptance backend. It keeps only
the implementation needed for the independent
`leonail1/OpenHarmony-ANNS-Test` harness:

- sidecar label ingestion and live-set tracking
- filtered search for `match_all`, `equality`, `intersect`, and `range`
- exact filtered search for small candidate sets
- FAISS IVFFlat plus exact rerank for large candidate sets
- insert/delete/selectivity command adapters

The old PipeANN SSD graph, PQ, ARIS experiments, packing experiments, threshold
prediction experiments, and Beamer deck have been removed from this codebase.
The historical Beamer PDF and source were moved to the test repository under
`docs/presentations/pipeann_beamer/`.

## Implementation

The search system is implemented in C++:

```text
src/openharmony_anns_adapter.cpp
```

It builds a single CLI binary:

```text
build/openharmony_anns_adapter
```

The backend uses FAISS C++ wherever FAISS provides the core primitive:

- `faiss::IndexIVFFlat` for high-selectivity ANN routing
- `faiss::write_index` / `faiss::read_index` for FAISS index persistence
- `faiss::fvec_L2sqr` plus FAISS heap utilities for filtered exact top-k and exact rerank
- OpenMP / FAISS threading for vector search kernels

The only custom C++ logic is the acceptance-contract layer around FAISS:
label CSV ingestion, sidecar label storage, live bitmap deletion, selector
matching, state manifests, and JSON command output.

## Public Commands

The five public commands are exposed through:

```text
build/openharmony_anns_adapter
```

The command templates used by the test harness live in:

```text
acceptance/adapter_manifest.openharmony.yaml
```

The backend deliberately does not expose PQ, graph routing, maintenance, or
threshold-prediction APIs. Those choices are internal implementation details or
not used by this clean backend.

## Dependencies

C++ backend dependencies on v100:

```bash
sudo apt-get install -y libfaiss-dev libopenblas-dev nlohmann-json3-dev
cmake -S /mnt/nvme1n1/PipeANN-github -B /mnt/nvme1n1/PipeANN-github/build
cmake --build /mnt/nvme1n1/PipeANN-github/build -j
```

The acceptance tests themselves use the Python environment from the test
repository:

```bash
cd /mnt/nvme1n1/OpenHarmony-ANNS-Test
source .venv/bin/activate
pip install -r requirements.txt
./tools/build_groundtruth.sh
```

## Acceptance Runs

Smoke:

```bash
cd /mnt/nvme1n1/OpenHarmony-ANNS-Test
source .venv/bin/activate
pytest --config /mnt/nvme1n1/PipeANN-github/acceptance/sift_smoke_config.yaml -q
```

Static, space, selectivity, and single-query resource:

```bash
pytest --config /mnt/nvme1n1/PipeANN-github/acceptance/sift1m_baseline_config.yaml -q
```

Dynamic 5-cycle insert/delete chain:

```bash
pytest -m dynamic --config /mnt/nvme1n1/PipeANN-github/acceptance/sift1m_dynamic5_config.yaml -q
```

Latest v100 C++ acceptance results:

- smoke: `5 passed in 13.27s`
- SIFT1M baseline: `5 passed in 801.38s`
- SIFT1M dynamic 5-cycle: `1 passed, 4 deselected in 1798.24s`
- dynamic5 checkpoints: 150/150 pass, min recall@10 `98.10%`, max avg latency `9.6515 ms`
- dynamic foreground search: 79/79 pass, max avg latency `2.0149 ms`
- delete API: 5 cycles pass, max `0.002509 ms/vector`
- space expansion: `1.2092425123555857x`

See `acceptance/FINAL_CPP_BACKEND_ACCEPTANCE_EVIDENCE.md` and
`acceptance/results/clean_cpp_backend_acceptance_summary.json`.

## Data Policy

Raw vectors, generated dynamic batches, runtime work directories, logs, and
indexes stay local on the experiment host and are ignored by git. Commit only
the adapter, configs, lightweight JSON/JSONL/CSV summaries, and concise
evidence documents.
