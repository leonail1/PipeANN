# Cleanup Previous Threshold Predictor

Removed the previous calibration-assisted graph/prefilter threshold predictor from tracked code and evidence via a non-history-rewriting cleanup commit.

Deleted tracked artifacts introduced by commits 0d576d1 and 216aef1:

- scripts/threshold_prediction_common.py
- scripts/build_threshold_prediction_dataset.py
- scripts/train_threshold_predictor.py
- scripts/predict_graph_prefilter_threshold.py
- scripts/validate_threshold_predictor_aris.py
- experiments/v100_threshold_prediction_final_20260603T_run/

Also removed untracked local scratch experiment directories matching v100_threshold_prediction_* from v100. These were calibration-assisted runs and are not valid evidence for the new feature-only goal.

Reason: the old predictor used prior calibration curves / sparse route probes and therefore cannot support the feature-only claim. The new goal must only use prediction-time observable dataset, vector, equality-label, index/layout, and hardware microbenchmark statistics as model inputs.
