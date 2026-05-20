# PQ16 r116 PQ Residency ARIS Self Review

Generated: 2026-05-20T07:39:14.949362+00:00

## Verdict

WARN until independent subagent review completes.

## Scope

- Dataset/index: r116 SIFT1M, PQ16, PPT per-bucket calibrated route/L.
- Selectors: intersect and range.
- Buckets: all 10 PPT selectivity buckets.
- Modes: `pq_memory` (`PIPEANN_PQ_MMAP=0`) and `pq_disk_no_cache` (`PIPEANN_PQ_MMAP=1`, `PIPEANN_PQ_MMAP_DROP_CACHE=1`).

## Memory Accounting

- Formula: `adjusted_rss_kb = measured_peak_rss_kb - ceil(query_file_bytes/1024) - ceil(gt_file_bytes/1024)`.
- Query label, base label, densebit/filter sidecar, tags/id map, PQ resident pages, and runtime buffers are counted.
- Max adjusted RSS, PQ resident memory: 32.41 MiB.
- Max adjusted RSS, PQ disk/no-cache: 18.71 MiB.

## Calibration and Recall

- Selected route/L rows: 20.
- Measurement rows: 40.
- Recall-failed measurement rows: 0.
- Failed points must be marked in plots and excluded from positive recall claims.

## Artifacts

- `pq16_pq_residency_compare.csv/jsonl`
- `selected_route_l.csv/jsonl`
- `calibration_results.csv/jsonl`
- `pq16_pq_residency_metadata.json`
- `pq16_pq_residency_input_inventory.csv`
- Four PNG/PDF residency plots.
