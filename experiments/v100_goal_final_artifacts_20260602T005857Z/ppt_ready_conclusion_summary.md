# V100 PPT-Ready Conclusion Summary

- Dynamic selected rows: `200`; recall pass `200`, avg<10ms pass `200`, p95<10ms pass `200`.
- Serving configuration口径: selected route/L/beamwidth; beamwidth distribution 4=165, 8=35.
- Worst avg latency: `9.179 ms`; worst p95 latency: `9.561 ms`.
- PQ matched-reference: `100/100` matched; unmatched cases: `[]`.
- 4KB read invariant violations: `0`; layout violations: `0`.
- Space: worst strict total/raw `1.980615x`, worst strict excess/raw `0.980615x`.
- Background maintenance: `PASS`; during max avg `9.254442801` ms.
- Figures: `['experiments/v100_goal_final_artifacts_20260602T005857Z/figures/v100_dynamic_recall_latency.svg', 'experiments/v100_goal_final_artifacts_20260602T005857Z/figures/v100_space_components.svg', 'experiments/v100_goal_final_artifacts_20260602T005857Z/figures/v100_background_interference.svg']`
