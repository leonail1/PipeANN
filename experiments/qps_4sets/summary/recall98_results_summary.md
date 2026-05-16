# qps_4sets Recall@10>=98 Results

Generated: 2026-05-16T14:56:58Z

## Scope

This directory contains the overwrite-style qps_4sets formal sweep where every plotted QPS point uses the calibrated route and selected search budget satisfying recall@10 >= 98%.

## Validation

- formal rows: 136
- fashion_mnist784: 36 rows, minimum recall 98.000000
- gist960: 36 rows, minimum recall 98.000000
- glove100: 36 rows, minimum recall 98.000000
- yfcc10m: 28 rows, minimum recall 98.000000
- below-98 rows: 0

## Budget Adjustments

- glove100/u50: L 2155 -> 2371 (formal thread sweep recall was 97.989998, just below 98.0)
- glove100/u100: L 1520 -> 1672 (formal thread sweep recall was 97.989998, just below 98.0)
- yfcc10m/real_t1e-01_l17: L 2101 -> 2312 (formal thread sweep recall was 97.989998, just below 98.0)

## Key Artifacts

- `formal_results.csv`
- `formal_results.jsonl`
- `bucket_plan_recall98.json`
- `qps_4sets_reproduction_qps.png`
- `qps_4sets_recall_budget.png`
- `scaling_efficiency.csv`
- `bottleneck_summary.csv`

## Notes

Per-bucket recall workloads, exact GT, and calibration probe outputs are generated workdirs and are intentionally ignored; they can be regenerated from the scripts and manifests.
