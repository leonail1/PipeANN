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
- dynamic_foreground_max_delete_ms_per_vector: `8.9096e-05`
- dynamic_batch_delete_rows: `5`
- dynamic_batch_max_delete_ms_per_vector: `7.25778e-05`
- single_query_latency_ms: `21.406`
- single_query_max_rss_bytes: `50880512`

## Failures

- space: expansion_ratio=2.070587
- dynamic_foreground: phase=merge,cycle=1,avg_latency_ms=12.185100
- single_query: latency_ms=21.406000
- rss: max_rss_bytes=50880512

## Scope notes

- Zero-start path was exercised by both foreground 1-cycle and batch 5-cycle dynamic runs.
- PQ retrain was intentionally not implemented in this run; post-10k updates use the native PipeANN dynamic path.
- Measured search latency used single-thread search; foreground updates used 4 threads and batch updates used 32 threads.
