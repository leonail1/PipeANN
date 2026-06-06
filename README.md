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

## Public Commands

The five public commands are exposed through:

```text
scripts/openharmony_anns_adapter.py
```

The command templates used by the test harness live in:

```text
acceptance/adapter_manifest.openharmony.yaml
```

The backend deliberately does not expose PQ, graph routing, maintenance, or
threshold-prediction APIs. Those choices are internal implementation details or
not used by this clean backend.

## Dependencies

The v100 runs use the Python environment from the test repository:

```bash
cd /mnt/nvme1n1/OpenHarmony-ANNS-Test
source .venv/bin/activate
pip install -r requirements.txt
./tools/build_groundtruth.sh
```

The adapter requires `numpy`; `faiss` is used for IVFFlat acceleration when it
is available. If FAISS is unavailable, exact filtered search still preserves
correctness but may not meet the full latency target at high selectivity.

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

## Data Policy

Raw vectors, generated dynamic batches, runtime work directories, logs, and
indexes stay local on the experiment host and are ignored by git. Commit only
the adapter, configs, lightweight JSON/JSONL/CSV summaries, and concise
evidence documents.
