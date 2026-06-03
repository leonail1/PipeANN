# PPT-ready Threshold Prediction Summary

- Task: predict graph/prefilter latency crossing selectivity `s*`.
- Accuracy: `59/61` crossing cases within 5%; rate `0.9672131147540983`.
- Boundary accuracy: `1.0`.
- Single-threshold max latency regret: `0.000`; all-case max latency regret: `0.420`; mean calibration cost `0.286` of full sweep.
- Use: predictor can reduce sweep cost when curves are close to linear; failed cases indicate where extra calibration points or richer features are needed.
