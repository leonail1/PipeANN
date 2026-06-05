# OpenHarmony ANNS Acceptance Evidence

Date: 2026-06-06
Host: `v100`
PipeANN repo: `/mnt/nvme1n1/PipeANN-github`
Test repo: `/mnt/nvme1n1/OpenHarmony-ANNS-Test`

## Test Runs

| Scope | Result dir | Pytest result |
| --- | --- | --- |
| Space, selectivity, static filtered search, single-query resource | `acceptance/results/sift1m_baseline` | `4 passed, 1 deselected in 483.63s` |
| Dynamic insert/delete chain | `acceptance/results/sift1m_dynamic5` | `1 passed, 4 deselected in 1896.47s` |

Both `acceptance_summary.json` files report `pass: true`, `exitstatus: 0`, and no failure files.

## Key Metrics

| Metric | Evidence | Value |
| --- | --- | --- |
| Static filtered rows | `static_filtered_search_results.jsonl` | 25 / 25 pass |
| Static min recall@10 | `static_filtered_search_results.jsonl` | 0.9991 |
| Static max avg latency | `static_filtered_search_results.jsonl` | 7.4935 ms |
| Dynamic checkpoint rows | `dynamic_update_chain_results.jsonl` | 150 / 150 pass |
| Dynamic min recall@10 | `dynamic_update_chain_results.jsonl` | 0.9990 |
| Dynamic max avg latency | `dynamic_update_chain_results.jsonl` | 6.3053 ms |
| Foreground mutation search rows | `dynamic_update_foreground_latency.jsonl` | 10 / 10 pass |
| Foreground max avg latency | `dynamic_update_foreground_latency.jsonl` | 7.6288 ms |
| Delete rows | `delete_api_timing.jsonl` | 5 / 5 pass |
| Max delete cost | `delete_api_timing.jsonl` | 0.00875 ms/vector |
| Space expansion | `space_audit.json` | 0.1895x |

## Groundtruth Path

The OpenHarmony-ANNS-Test harness computes recall groundtruth through PipeANN C++ `compute_groundtruth`.
The Python harness prepares candidate/query files and reads the C++ truthset output; Python exact distance/sort fallback was removed.

## Commit Scope Note

Commit only lightweight files:

- `acceptance/*.yaml`
- `acceptance/*.md`
- `acceptance/results/**/*.json`
- `acceptance/results/**/*.jsonl`
- `acceptance/results/**/*.csv`
- `scripts/openharmony_anns_adapter.py`
- OpenHarmony-ANNS-Test harness changes

Do not commit `acceptance/work/`, logs, generated indexes, or raw vector slices.
