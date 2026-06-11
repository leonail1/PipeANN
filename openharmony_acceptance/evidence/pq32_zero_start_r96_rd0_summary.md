# PQ32 Zero-Start R96/Rdense0 Summary

This is a GitHub-safe pointer summary for the PQ32 zero-start full acceptance run.
Raw logs, indexes, GT files, and datasets remain on node4 and are not committed.

## Run

- Run tag: `full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126`
- Host path: `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony`
- Result path: `/data/lzg/pipeann_recovery_20260610T213643/PipeANN-0.3-s2pq-openharmony/acceptance_results/full_zero_start_r96_rd0_st1_fg4_batch32_gt16_data_sift100m_3m6m_20260610T221126`
- PQ setting: `PQ_BYTES=32`
- Index setting: `R=96`, `R_DENSE=0`
- Search threads: `1`
- Foreground update threads: `4`
- Batch update threads: `32`

## Key Results

- Overall pass: `false`
- Space expansion: `2.07059x`
- Static filtered rows: `25`
- Static min recall@10: `98.4`
- Static worst avg latency: `4.54562 ms`
- Dynamic batch checkpoint rows: `125`
- Dynamic batch min recall@10: `98.05`
- Dynamic batch worst avg latency: `9.13025 ms`
- Foreground rows: `737`
- Foreground phase means:
  - `after_mark_delete`: `9.64622 ms`
  - `merge`: `6.64331 ms`
  - `insert`: `2.94028 ms`
  - `after_insert`: `2.21585 ms`
- Foreground row warnings: `1/737`
- Delete max latency:
  - foreground: `8.9096e-05 ms/vector`
  - batch: `7.25778e-05 ms/vector`
- Single-query resource max RSS: `50,880,512 bytes`

## Failures Under Current Policy

- `space`: `expansion_ratio=2.070587`
- `rss`: `max_rss_bytes=50880512`

## Notes

- Single-query latency is retained as a cold resource diagnostic and is not a pass/fail latency gate.
- Foreground latency is judged phase by phase using the mean of probe average latencies; individual rows above 10 ms are warnings.
- PQ retrain was intentionally not implemented in this run.
