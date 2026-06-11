# Zero-start full acceptance evidence: full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126

This is a GitHub-safe compact evidence note. Raw logs, GT files, indexes, and datasets remain on the experiment host and are not committed.

## Artifact paths

- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/acceptance_summary.json`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/space_audit.json`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/static_filtered.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/dynamic_foreground_chain.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/dynamic_foreground_latency.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/dynamic_batch_chain.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/dynamic_batch_checkpoint_search.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/single_query_resource.jsonl`
- `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126/single_query_time.txt`

## Summary

- Overall pass: `False`
- space_expansion_ratio: `2.07059`
- static_rows: `25`
- static_min_recall: `98.4`
- static_worst_avg_latency_ms: `4.54562`
- dynamic_batch_checkpoint_rows: `125`
- dynamic_batch_checkpoint_min_recall: `98.05`
- dynamic_batch_checkpoint_worst_avg_latency_ms: `9.13025`
- dynamic_foreground_rows: `737`
- dynamic_foreground_worst_avg_latency_ms: `12.1851`
- dynamic_foreground_mean_avg_latency_ms: `3.18454`
- dynamic_foreground_avg_latency_warning_rows: `1`
- dynamic_foreground_avg_latency_warning_ratio: `0.00135685`
- dynamic_foreground_max_delete_ms_per_vector: `8.9096e-05`
- dynamic_batch_delete_rows: `5`
- dynamic_batch_max_delete_ms_per_vector: `7.25778e-05`
- single_query_latency_ms: `21.406`
- single_query_max_rss_bytes: `50880512`

## Foreground Phase Means

- after_insert: rows `1`, mean avg `2.21585`, worst row avg `2.21585`, row warning ratio `0`
- after_mark_delete: rows `1`, mean avg `9.64622`, worst row avg `9.64622`, row warning ratio `0`
- insert: rows `688`, mean avg `2.94028`, worst row avg `6.43201`, row warning ratio `0`
- merge: rows `47`, mean avg `6.64331`, worst row avg `12.1851`, row warning ratio `0.0212766`

## Failures

- space: expansion_ratio=2.070587
- rss: max_rss_bytes=50880512

## Warnings

- dynamic_foreground: phase=merge,cycle=1,avg_latency_ms=12.185100

## Scope notes

- Zero-start path was exercised by both foreground 1-cycle and batch 5-cycle dynamic runs.
- PQ retrain was intentionally not implemented in this run; post-10k updates use the native PipeANN dynamic path.
- Measured steady-state search latency uses single-thread 1000-query static and checkpoint searches; single-query latency is a cold resource diagnostic and is not a pass/fail latency gate.
- Foreground latency is judged phase by phase using the mean of probe avg latencies; individual probe rows above 10 ms are warnings.
