# Node4 Full Acceptance Result: R96/R_dense0 Saved Static Checkpoints

Run id: `full_node4_r96_rd0_st1_fg4_batch32_gt16_save_static_sift100m_3m6m_20260610T100604`

Code commit: `a7467d5` (`experiment/pipeann-0.3-s2pq-openharmony`)

Host and data:
- Host: `node4`
- Repository path during run: `/mnt/nvme2n1/pipeann_recovery_20260610T091347/PipeANN-0.3-s2pq-openharmony`
- Base dataset: SIFT1M
- Update vectors: real SIFT/BIGANN rows `[3M,6M)`
- Build/search: `R=96`, `R_dense=0`, measured search threads `1`
- Foreground update test: `1` cycle, update threads `4`
- Batch quality test: `5` cycles, update threads `32`
- Batch checkpoint mode: insert to 1M, call `save()`, then run static filtered search on the saved disk snapshot

Overall result: `pass=false`, but all planned test sections ran and produced artifacts.

Key metrics from `acceptance_summary.json`:
- Space expansion: `2.07059x` core index bytes / raw vector bytes, failed `<2.0x`.
- Static filtered search: `25/25` rows, min recall `98.56`, worst avg latency `4.5815 ms`, passed.
- Dynamic batch checkpoints: `125/125` rows, min recall `98.01`, worst avg latency `5.121 ms`, passed.
- Foreground update search: `811` rows, worst avg latency `8.62443 ms`, passed.
- Delete latency: foreground max `9.42632e-05 ms/vector`, batch max `8.39842e-05 ms/vector`, passed.
- Single query: latency `16.203 ms`, failed `<10 ms`.
- Single query max RSS: `129654784 bytes`, failed `<30000000 bytes`.

Failure list:
- `space`: `expansion_ratio=2.070587`
- `single_query`: `latency_ms=16.203000`
- `rss`: `max_rss_bytes=129654784`

Included artifacts are GitHub-safe small summaries/results only. Full logs, GT files, generated indices, work directories, and raw datasets are intentionally omitted.
