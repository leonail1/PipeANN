# Equality Label Distribution Suite

All generated filters are equality filters: a query asks for exactly one target label.
Feature extraction is allowed to inspect vectors, target-label membership ids, label counts, index/layout metadata, and independent hardware constants.
Feature extraction is not allowed to inspect graph/prefilter latency curves, sparse probes, oracle thresholds, or query-sweep results.

Generated label families:
- `random_uniform`
- `contiguous_front`

Families are used for benchmark generation and held-out splitting. They must not be used as a shortcut deployment feature unless a leakage/ablation audit explicitly permits it.
