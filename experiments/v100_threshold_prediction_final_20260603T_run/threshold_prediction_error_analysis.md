# Threshold Prediction Error Analysis

## Worst Threshold Errors
- `pq_drift_1m_aris_main_20260522__phaseC_cycle03_retrain_each_cycle__range__5a4896abce16`: status=crossing, s_exp=0.1006022916812819, s_pred=0.11200000000000002, rel_err=0.11329471852219024
- `r116_suite_pq16_aris_20260520_072453__static__range__6383dcd166d5`: status=crossing, s_exp=0.5039632345942893, s_pred=0.56, rel_err=0.11119216950582247
- `v100_pq_drift_1m_baseline_20260601T043217Z__phaseC_cycle03_no_retrain_across_cycles__range__20f7e7125801`: status=crossing, s_exp=0.04586870503660062, s_pred=0.044000000000000004, rel_err=0.04074030507531216
- `v100_pq_drift_1m_baseline_20260601T043217Z__phaseC_cycle01_no_retrain_across_cycles__range__a39212d36d7b`: status=crossing, s_exp=0.042400064820849834, s_pred=0.044000000000000004, rel_err=0.03773426257507553
- `pq_drift_1m_aris_main_20260522__phaseB_cycle00_direct_retrain_1m__intersect__a092c6ce585b`: status=crossing, s_exp=0.08404074094290895, s_pred=0.08404074094290895, rel_err=0.0

## Notes
- Validation is an offline replay of calibration artifacts produced with original query files. It does not start new query binaries.
