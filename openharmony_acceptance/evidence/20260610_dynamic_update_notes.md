# 2026-06-10 Dynamic Update Notes

This note records small, GitHub-safe evidence for the OpenHarmony C++ acceptance
work on PipeANN 0.3.0. Large indexes, GT files, logs, and raw result directories
remain on the experiment machine and are intentionally not committed.

## Code fixes

- `DynamicIndex::search()` filtered path now passes `topk` as `k_search` and
  `L` as `l_search` to `SSDIndex::spec_filter_search()`.
  - Previous behavior passed `L, L`, so dynamic checkpoint search internally
    requested top-L results and then truncated to top-k.
  - Static filtered search already used `k=10, L=<candidate>`, so this fix
    aligns dynamic checkpoint latency with the static search contract.
- Dynamic update runner now writes `dynamic_progress.jsonl`.
  - Progress events include mark-delete start/done, merge running/done, and
    insert running/done.
  - Insert progress reports `done`, `total`, `percent`, elapsed time, and
    vectors/s.
- Attribute merge remap now uses fast remap modes before falling back to hash
  lookup.
  - Dense remap handles arbitrary delete patterns when the old-id domain fits
    the configured memory budget.
  - This keeps official merge semantics while avoiding per-posting cuckoo hash
    lookup in the hot path.

## Targeted observations

Run paths on `v100`:

- Repository: `/mnt/nvme1n1/pipeann_03_s2pq_work/PipeANN-0.3.0-s2pq-clean`
- Interrupted diagnostic full run:
  - `acceptance_results/full`
  - `acceptance_logs/full_fastremap_r96_rd0_st1_upd4_gt16_sift100m_3m6m_20260610T152758.log`
- New full run with 8 update threads:
  - `acceptance_results/full_fastremap_r96_rd0_st1_upd8_gt16_sift100m_3m6m_20260610T161318`
  - `acceptance_logs/full_fastremap_r96_rd0_st1_upd8_gt16_sift100m_3m6m_20260610T161318.log`

Partial thread probe:

| update threads | status | foreground avg max | notes |
| --- | --- | ---: | --- |
| 16 | stopped early | 10.998 ms | merge foreground exceeded 10 ms |
| 8 | stopped early | 7.200 ms | insert reached 78,430 / 600,000 with foreground avg below 10 ms |

The 16-thread probe used more background parallelism but triggered a foreground
latency violation during merge. The 8-thread setting was selected for the new
full run.

## Known current status

The 8-thread full run was launched with:

- `R=96`
- `R_DENSE=0`
- `SEARCH_THREADS=1`
- `UPDATE_THREADS=8`
- `GT_THREADS=16`
- SIFT1M base/query
- SIFT100M rows `[3M, 6M)` as update vectors

The final pass/fail result is pending completion of the full run.
