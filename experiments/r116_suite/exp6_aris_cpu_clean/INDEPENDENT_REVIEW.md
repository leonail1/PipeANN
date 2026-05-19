# Independent Review: r116 ARIS CPU Clean Run

日期：2026-05-19

Reviewer subagent: `019e3f6d-ff66-7613-8f99-375f2c85e430`

## Verdict

`experiments/r116_suite/exp6_aris_cpu_clean` resolves the earlier ledger inconsistency. The reviewer verified that `results.jsonl`, `measure_driver.jsonl`, `jobs/*.json`, and `table.csv` all contain 320 same-source records, with no mismatches found.

The clean run supports the bounded conclusion that, on node6 with the current r116/PQ32 index and documented `u75 -> graph L=75` override, all `intersect/range x 10 selectivity buckets x 1..16 threads` combinations satisfy:

- `avg_latency_us <= 10000`
- `recall@10 >= 98`

## Confirmed Fixes

- The clean directory starts from the final override set; `u75` is present in manifest/jobs as `graph + L=75`.
- `summary.status` is stricter than the first runner version: it fails when rows are incomplete, recall falls below 98, or average latency exceeds 10ms.
- `summary.json` reports `recall_failed_rows=0`, `avg_budget_failed_rows=0`, `min_recall@10=98.839996`, and `max_avg_latency_ms=9.0638`.

## Remaining Risks

- Tail latency is not guaranteed under 10ms. The clean run still has `p90>10ms` in 19 rows, `p95>10ms` in 23 rows, and `p99>10ms` in 86 rows.
- `u75` is a documented post-review recalibration, so the result should not be described as "exp4 original route/L transfers unchanged."
- The run uses an existing r116 index prefix from the interrupted exp6 setup; it is explicitly referenced in `manifest.json`.
- The original ARIS GPU queue manager is not used because this is a CPU/SSD benchmark on node6.

## Reviewed Files

- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/summary.json`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/manifest.json`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/table.csv`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/results.jsonl`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/measure_driver.jsonl`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/l_overrides.json`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu_clean/ARIS_EXPERIMENT_REVIEW.md`
- `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_aris_cpu/aris_cpu_exp6_runner.py`
