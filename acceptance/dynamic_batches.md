# Dynamic Batch Inputs

`acceptance/sift1m_dynamic5_config.yaml` references five generated replacement
batches under `acceptance/work/dynamic_batches/`.

These files are kept local on v100 because each vector batch is large:

- `cycle1_600k.fbin` and `cycle1_600k.ids`
- `cycle2_600k.fbin` and `cycle2_600k.ids`
- `cycle3_600k.fbin` and `cycle3_600k.ids`
- `cycle4_600k.fbin` and `cycle4_600k.ids`
- `cycle5_600k.fbin` and `cycle5_600k.ids`

They are not source artifacts and must not be committed to GitHub.
