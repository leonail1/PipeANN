# Single-query L and RSS optimization notes

Context: SIFT1M, R=96, R_dense=0, search threads=1, selector `range_s10`, final saved checkpoint from
`full_node4_r96_rd0_st1_fg4_batch32_gt16_save_static_sift100m_3m6m_20260610T100604`.

Key findings:

- The original single-query failure was caused by using the global `SEARCH_L=100`.
- Static filtered search selected `L=40` for `range_s10` with recall@10 around 99.25%, so the single-query test now reuses static `selected_L`.
- Hash-map preallocation and identity table optimizations reduce max RSS from about 126 MB to about 55 MB while keeping latency below 10 ms.
- `/proc/*/smaps_rollup` shows the remaining PQ32 RSS is dominated by private anonymous memory when loaded as a vector, and by file-backed PQ pages when loaded via mmap.
- PQ32 compressed codes remain the dominant RSS component. PQ mmap/stream and PQ12/PQ14/PQ16/PQ8 trials did not find a configuration that satisfies both `<10 ms` latency and `<30 MB` max RSS for this selector.
- PQ12/PQ14/PQ16/PQ8 were not adopted: smaller PQ sidecars reduce RSS, but require larger L or trigger slower search paths and fail the single-query latency check.

Artifacts are small summaries only; full raw experiment directories remain under `acceptance_results/` on node4.
