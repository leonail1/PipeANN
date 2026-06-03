# Threshold Feature Schema

Each curve is keyed by `case_id = experiment_dir x cycle x selector_type`.

- `selectivity`: mean candidate_count / live_point_count, falling back to bucket name parsing.
- `graph_*` and `prefilter_*`: fastest measured row for that route at the selectivity. If no row reaches recall>=98, the fastest row is retained and marked by `*_source_status`.
- `latency_diff_graph_minus_prefilter_ms`: positive means prefilter is faster; negative means graph is faster.
- `s_exp`: piecewise-linear intersection of graph and prefilter avg-latency curves.
- `threshold_status`: `crossing`, `boundary`, `insufficient`, or `non_monotonic_no_single_crossing`.
- `boundary_route`: route that is no slower across the observed selectivity range when no crossing exists.
- Validation uses sparse calibration points from each curve to predict `s_pred`, then compares against `s_exp`.
