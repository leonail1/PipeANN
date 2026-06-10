# Single-query L and RSS optimization notes

Context: SIFT1M, R=96, R_dense=0, search threads=1, selector `range_s10`, final saved checkpoint from
`full_node4_r96_rd0_st1_fg4_batch32_gt16_save_static_sift100m_3m6m_20260610T100604`.

Key findings:

- The original single-query failure was caused by using the global `SEARCH_L=100`.
- Static filtered search selected `L=40` for `range_s10` with recall@10 around 99.25%, so the single-query test now reuses static `selected_L`.
- Hash-map preallocation and identity table optimizations reduce max RSS from about 126 MB to about 55 MB while keeping latency below 10 ms.
- PQ32 compressed codes remain the dominant RSS component. PQ mmap/stream and PQ16/PQ8 trials did not find a configuration that satisfies both `<10 ms` latency and `<30 MB` max RSS for this selector.
- PQ16/PQ8 were not adopted: both reduced RSS but increased latency above 10 ms in this checkpoint-level probe.

Artifacts are small summaries only; full raw experiment directories remain under `acceptance_results/` on node4.
