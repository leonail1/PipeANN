# ARIS Threshold Prediction Final Review

- Claim T2 threshold accuracy status: `PASS`.
- Main gate: 90% of crossing cases within 5% relative error; observed `0.9672131147540983`.
- Boundary accuracy: `1.0`.
- Single-threshold max latency regret: `0.0000`; all-case max latency regret: `0.4202`.
- Multi/non-single-threshold cases: `16`; all-case route-risk claim: `FAIL`.
- Caveat: the accepted claim is the single-threshold graph/prefilter crossing predictor. Multi-crossing curves violate the single-threshold assumption and need either extra calibration points, a multi-segment route policy, or a richer query/workload model before claiming all-case oracle-regret control.
