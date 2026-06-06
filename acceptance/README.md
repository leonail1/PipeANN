# OpenHarmony ANNS Acceptance Adapter

This directory contains the search-system side of the
`OpenHarmony-ANNS-Test` contract: command manifest, SIFT1M configs, lightweight
results, and final evidence.

The tests live in `/mnt/nvme1n1/OpenHarmony-ANNS-Test` and must not be modified
to make this backend pass. The test harness computes recall groundtruth with
its own C++ exact L2 tool, so this repository no longer depends on old PipeANN
C++ utilities.

## Backend

`scripts/openharmony_anns_adapter.py` implements:

- `ann_build_index`
- `ann_filter_search`
- `ann_apply_insert`
- `ann_apply_delete`
- `ann_label_selectivity`

Search uses exact filtered evaluation for small candidate sets and FAISS
IVFFlat plus exact rerank for large candidate sets. Labels and live/deleted
state are stored as sidecar arrays.

## Required Local Inputs

The SIFT1M configs expect these host-local files:

- `data/sift1m/sift_base.bin`
- `data/sift1m/sift_query.bin`
- `acceptance/work/dynamic_batches/cycle{1..5}_600k.fbin`
- `acceptance/work/dynamic_batches/cycle{1..5}_600k.ids`

The dynamic batch files are large generated inputs and are intentionally not
tracked by git.

## Run

```bash
cd /mnt/nvme1n1/OpenHarmony-ANNS-Test
source .venv/bin/activate
./tools/build_groundtruth.sh

pytest --config /mnt/nvme1n1/PipeANN-github/acceptance/sift_smoke_config.yaml -q
pytest --config /mnt/nvme1n1/PipeANN-github/acceptance/sift1m_baseline_config.yaml -q
pytest -m dynamic --config /mnt/nvme1n1/PipeANN-github/acceptance/sift1m_dynamic5_config.yaml -q
```
