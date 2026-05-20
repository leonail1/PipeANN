# Label Space Ratio Experiment

ARIS-style experiment record.

- Evaluation type: `file_size_accounting`.
- Scope: original `.spmat` label files versus processed mixed densebit sidecar files.
- Measurement: byte counts from filesystem `stat`; no synthetic ground truth and no normalized score.
- Reproduce: run `python3 experiments/label_space_ratio/measure_label_space.py` from the repo root on node6.

## Outputs

- `table.csv`: per-dataset byte counts and ratios.
- `summary.json`: machine-readable summary and exact input paths.
- `label_space_ratio.png`: high-resolution visualization used by the PPT.

## Result

- Minimum processed/original ratio: SIFT1M/r116 at 2.94%.
- Maximum processed/original ratio: YFCC10M at 38.08%.
- The processed label sidecar is smaller than the original `.spmat` label file for every measured dataset.
