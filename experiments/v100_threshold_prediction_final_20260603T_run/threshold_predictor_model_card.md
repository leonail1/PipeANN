# Threshold Predictor Model Card

- Model type: `prior_guided_sparse_linear_threshold`.
- Training curves: `81`.
- Crossing cases used for threshold/orientation priors: `61`.
- Sparse-correction cases: `61`; correction remains neutral in this model.
- Global correction: `1.000000` (kept neutral; no residual-correction claim).
- Inputs: sparse calibration points from graph and prefilter latency curves, selector type, and route-latency differences.
- Output: crossing threshold `s_pred`, or a boundary route when no crossing is predicted.
- Rationale: with limited historical curves, a transparent sparse piecewise-linear cost model is safer than a high-capacity model. The learned part is a selector/range median crossing prior, orientation prior, and an exact selector/range boundary-route prior, re-estimated during leave-one-out validation. Broader boundary priors are used only as final route fallbacks for no-single-threshold cases. When sparse probes show graph already faster at the lowest sampled selectivity under a low-to-high crossing orientation, the model extrapolates the crossing just left of that probe rather than treating it as a no-crossing boundary.
- Limitations: this is not a per-query neural model; it predicts query-set thresholds from original-query calibration summaries. Low-confidence or boundary cases should fall back to extra calibration points.
