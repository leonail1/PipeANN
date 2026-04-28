# Codex Satisfied Validation Results

Thresholds used for this summary: recall@10 >= 98%, average latency < 10 ms, single-query RSS <= 30 MB, and extra index bloat <= 1x raw vectors.

## Equality Queries

| Bucket | Recall@10 (%) | Avg latency (ms) | P99 latency (ms) | Route |
| --- | ---: | ---: | ---: | --- |
| 0.1% | 98.20 | 0.119 | 2.884 | prefilter |
| 0.3% | 98.45 | 0.210 | 3.222 | prefilter |
| 1% | 98.30 | 0.440 | 2.311 | prefilter |
| 5% | 98.10 | 1.998 | 6.598 | prefilter |
| 10% | 98.25 | 2.812 | 6.412 | prefilter |
| 25% | 98.05 | 9.321 | 12.502 | graph |
| 30% | 98.05 | 9.435 | 14.073 | graph |
| 50% | 99.55 | 9.305 | 12.912 | graph |
| 75% | 99.70 | 9.200 | 11.930 | graph |
| 100% | 99.95 | 9.276 | 12.141 | graph |

## Range Query

- range_0_2: recall@10=100.00%, avg=1.255 ms, p99=4.199 ms.

## Resource Footprint

- equality_u1e-03: max RSS=13.82 MiB.
- equality_u100: max RSS=16.44 MiB.
- range_0_2: max RSS=16.78 MiB.
- Index extra bloat: 0.626x raw vectors; total/raw=1.626x.
