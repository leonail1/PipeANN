# OpenHarmony ANNS Acceptance Evidence

Date: 2026-06-06
Host: `v100`
Search backend repo: `/mnt/nvme1n1/PipeANN-github`
Test repo: `/mnt/nvme1n1/OpenHarmony-ANNS-Test`

## Backend Scope

The cleaned backend is exact filter + FAISS IVFFlat. It no longer contains or claims the old PipeANN SSD graph, PQ codebooks, PQ retraining, supersector packing, ARIS experiments, or graph-routing threshold prediction.

## Groundtruth

Recall groundtruth is produced by the independent test repository:

```text
/mnt/nvme1n1/OpenHarmony-ANNS-Test/tools/bin/compute_groundtruth
```

The tested search system does not provide or influence groundtruth.

## Acceptance Runs

| Scope | Result dir | Pytest result |
| --- | --- | --- |
| Smoke | `acceptance/results/sift_smoke` | `5 passed in 27.02s` |
| Space, selectivity, static filtered search, single-query resource | `acceptance/results/sift1m_baseline` | `5 passed in 762.33s` |
| Dynamic 5-cycle insert/delete chain | `acceptance/results/sift1m_dynamic5` | `1 passed, 4 deselected in 1809.95s` |

Both full result directories report `pass: true`, `exitstatus: 0`, and no failure files.

## Key Metrics

| Metric | Value |
| --- | ---: |
| Space expansion ratio | 0.189465x |
| Static filtered rows passing | 25 / 25 |
| Static min recall@10 | 0.9995 |
| Static max avg latency | 6.4785 ms |
| Single-query rows passing | 6 / 6 |
| Single-query max avg latency | 4.2666 ms |
| Dynamic checkpoint rows passing | 150 / 150 |
| Dynamic min recall@10 | 0.9992 |
| Dynamic max avg latency | 6.8071 ms |
| Foreground mutation-search rows passing | 10 / 10 |
| Foreground max avg latency | 2.6910 ms |
| Insert rows passing | 6 / 6 |
| Delete rows passing | 5 / 5 |
| Max delete cost | 0.008565 ms/vector |
| Label selectivity rows passing | 128 / 128 |

## Commit Scope Note

Commit only lightweight files: adapter, configs, README/evidence, and JSON/JSONL/CSV result summaries. Do not commit `acceptance/work/`, logs, raw vector files, dynamic batches, generated indexes, or local data.
