# Threshold Prediction Related Work

The target here is narrower than general ANN hyperparameter search: predict the
selectivity threshold where graph-search latency and prefilter latency cross.

- Faiss ParameterSpace autotuning builds operating points and keeps Pareto
  optimal recall/time configurations. This is useful as an oracle and validation
  pattern, but it still explores parameter combinations rather than predicting a
  PipeANN graph/prefilter route threshold from features.
  Source: https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning
- Google's constrained-optimization framing for ANN configuration treats recall,
  latency and resource limits as constrained objectives. The useful transfer is
  the explicit constrained selection layer: predict performance first, then pick
  a route only if it satisfies recall/latency gates.
  Source: https://arxiv.org/abs/2301.01702
- VDTuner studies automatic tuning for vector data management systems. It
  supports the idea that vector DB tuning should use workload/index/system
  features, but PipeANN still needs disk-graph, 4KB IO and dynamic-update
  features that are not generic DB knobs.
  Source: https://arxiv.org/abs/2404.10413
- FastPGT targets proximity-graph construction parameter tuning and reduces
  repeated graph-build cost by estimating multiple candidate parameter settings
  together. It is relevant to offline graph-build tuning, while this goal is an
  online route-threshold predictor over already-built graph/prefilter curves.
  Source: https://arxiv.org/abs/2602.11573
- RP-Tuning adjusts DiskANN-style graph reachability parameters by pruning
  rather than rebuilding the whole graph. It is relevant for future graph-quality
  maintenance, but it does not predict filter selectivity thresholds.
  Source: https://arxiv.org/abs/2602.08097
- Learned adaptive early termination predicts when ANN search can stop. The
  transferable idea is a learned cost/risk model over query/index state, though
  this goal predicts route thresholds across filter selectivity rather than
  stopping depth inside one route.
  Source: https://www.pdl.cmu.edu/PDL-FTP/BigLearning/mod0246-liA.pdf
- Learning-to-rank / learned routing work for ANN suggests query- or workload-
  aware route choice. The PipeANN variant here uses route latency curves and
  4KB IO statistics as the supervised signal.
  Source: https://arxiv.org/abs/2404.11731
