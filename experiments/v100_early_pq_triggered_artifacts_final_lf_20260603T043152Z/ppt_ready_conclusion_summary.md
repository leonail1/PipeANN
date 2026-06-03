# Early-PQ Triggered Maintenance PPT Summary

- No-retrain early-PQ baseline fast-fails: `12` rows, worst avg `13.616 ms`.
- Triggered retrain targeted sentinels: `24` rows, min recall `98.10`, worst avg `9.236 ms`, worst p95 `9.497 ms`.
- Chain progress: `5/5` cycles, rows `30/30`, status `PASS`.
- Maintenance max: build `314.548s`, PQ train `2.976s`, PQ recode `3.471s`.
- Background interference: `PASS` for early-PQ foreground overlap/timing, rows `30`, worst avg `9.346 ms`, worst p95 `9.633 ms`; layout/4KB status is scoped separately by the reused v100 layout audit.
- Figures: `['experiments/v100_early_pq_triggered_artifacts_final_lf_20260603T043152Z/figures/early_pq_latency_recall_targeted.svg', 'experiments/v100_early_pq_triggered_artifacts_final_lf_20260603T043152Z/figures/early_pq_chain_latency_by_cycle.svg', 'experiments/v100_early_pq_triggered_artifacts_final_lf_20260603T043152Z/figures/early_pq_maintenance_costs.svg']`
