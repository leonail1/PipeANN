# OpenHarmony ANNS C++ Backend Acceptance Evidence

Date: 2026-06-06
Host: `v100`
Search backend repo: `/mnt/nvme1n1/PipeANN-github`
Test repo: `/mnt/nvme1n1/OpenHarmony-ANNS-Test`

## Backend Scope

The backend is implemented as a C++ CLI binary:

```text
build/openharmony_anns_adapter
```

It implements the five `OpenHarmony-ANNS-Test` interfaces without using a
Python search-system adapter:

- `ann_build_index`
- `ann_filter_search`
- `ann_apply_insert`
- `ann_apply_delete`
- `ann_label_selectivity`

FAISS C++ is used for IVFFlat, index IO, exact filtered top-k, exact rerank,
and threaded vector kernels. Custom code is limited to acceptance-contract
plumbing: labels, selectors, live bitmap deletion, state files, and JSON output.

The old PipeANN SSD graph, PQ codebooks, PQ retraining, supersector packing,
ARIS experiments, and graph-routing threshold prediction are not part of this
backend.

## Groundtruth

Recall groundtruth is produced by the independent test repository:

```text
/mnt/nvme1n1/OpenHarmony-ANNS-Test/tools/bin/compute_groundtruth
```

The tested search system does not provide or influence groundtruth.

## Acceptance Runs

| Scope | Result dir | Pytest result |
| --- | --- | --- |
| Smoke | `acceptance/results/sift_smoke_cpp` | `5 passed in 13.27s` |
| Space, selectivity, static filtered search, single-query resource | `acceptance/results/sift1m_baseline_cpp` | `5 passed in 801.38s` |
| Dynamic 5-cycle insert/delete chain | `acceptance/results/sift1m_dynamic5_cpp` | `1 passed, 4 deselected in 1798.24s` |

## Key Metrics

Summarized in:

```text
acceptance/results/clean_cpp_backend_acceptance_summary.json
```

| Metric | Result |
| --- | --- |
| Space expansion | `1.2092425123555857x` on SIFT1M baseline, pass `<2.0x` |
| Baseline static filtered search | 25/25 rows pass; min recall@10 `98.91%`; max avg latency `9.2406 ms` |
| Baseline dynamic checkpoint | 25/25 rows pass; min recall@10 `98.91%`; max avg latency `8.9207 ms` |
| Baseline single-query resource | 6/6 rows pass; max avg latency `2.0216 ms`; max RSS recorded |
| Dynamic 5-cycle search checkpoints | 150/150 rows pass; min recall@10 `98.10%`; max avg latency `9.6515 ms` |
| Dynamic foreground search during insert/delete | 79/79 rows pass; max avg latency `2.0149 ms` |
| Dynamic delete latency | 5 cycles pass; 3,000,000 total deleted; max `0.002509 ms/vector` |
| Dynamic insert wall time | 6 insert calls pass; max wall time `22.1871 s` |
| Label selectivity | baseline 32/32 pass; dynamic5 96/96 pass |

There are no `*failures.csv` files in the three C++ result directories.

## Design Notes

- Static and dynamic filtered search use exact filtered top-k for candidate
  sets up to 10,000 vectors.
- Larger candidate sets use FAISS C++ IVFFlat followed by FAISS exact L2
  rerank over filtered ANN hits.
- Batch filtered search uses `nprobe=224`; full-selectivity batch queries use
  `nprobe=128`; single-query/no-groundtruth probes use a lighter FAISS warmup
  path with `nprobe=1`.
- Deletes are mark-delete/live-bitmap updates. Insert builds the next immutable
  FAISS snapshot in a staging directory and publishes it with a short commit
  step, so foreground reads continue to use a consistent snapshot.
