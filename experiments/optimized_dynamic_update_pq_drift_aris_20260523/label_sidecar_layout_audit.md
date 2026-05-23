# Label Sidecar Layout Audit (ARIS Phase4)

## Claim Being Audited

Strong claim under audit: when builds use `--label-storage sidecar`, the main `_disk.index` does not embed per-node label payloads; labels are query-serving sidecar files.

## Code Path Evidence

Build-time selection:

- `tests/build_disk_index.cpp:24` defaults `label_storage` to `sidecar`.
- `tests/build_disk_index.cpp:113-120` accepts only `sidecar` or `embedded`.
- `tests/build_disk_index.cpp:220-221` sets `main_index_label = label` only for `embedded`; for `sidecar`, it passes `nullptr` as the main index label writer.
- `tests/build_disk_index.cpp:227-239` passes `main_index_label` and `label_source_file` into `build_disk_index`.

Main index layout:

- `include/utils/index_build_utils.h:16-24` explicitly allows a `label_source_file` without a label writer and logs that labels will be written to sidecar only.
- `src/utils/index_build_utils.cpp:399-404` computes `label_size = 0` when the label writer is `nullptr`; this feeds `max_node_len` and disk metadata.
- `src/utils/index_build_utils.cpp:449-451` writes labels into node records only if `label_size > 0`.
- `src/utils/index_build_utils.cpp:481-482` builds the densebit sidecar from `label_source_file` after the main disk layout is written.
- `include/ssd_index_defs.h:120-132` persists `label_size`; audited indexes report `label_size=0`.

Runtime loading:

- `src/ssd_index.cpp:223-236` looks for `<index_prefix>_labels.densebit` and loads it with `DenseBitsetIndex::load`.
- `src/ssd_index.cpp:244-254` loads `<index_prefix>_hybrid.meta` and validates it against the densebit header.
- `src/ssd_index.cpp:340-352` loads `_disk.index.tags` separately only when tags are enabled; tags are not labels.

Sidecar format:

- `src/filter/densebit_index.cpp:203-205` defines the sidecar path as `<index_prefix>_labels.densebit`.
- `src/filter/densebit_index.cpp:344-493` writes v2 densebit/posting sidecars atomically.
- `src/filter/densebit_index.cpp:255-337` validates v1/v2 sidecar layout and rejects bad magic, size, or payload ranges.

## Measured Evidence

| Artifact | Value |
|---|---:|
| PhaseC cycle05 `_disk.index` metadata `label_size` | 0 |
| PhaseC cycle05 `_labels.densebit` size | 931416 bytes |
| PhaseC cycle05 sidecar version | 2 |
| PhaseC cycle05 sidecar npoints | 1,000,000 |
| PhaseC cycle05 sidecar nlabels | 10 |
| PhaseC cycle05 sidecar nnz | 2,964,000 |
| PhaseC cycle05 sidecar dense/sparse labels | 7 / 3 |

Driver evidence also reports sidecar mode:

- `experiments/dynamic_delete_pq_drift_aris_20260522_phase12_rerun/raw/phase2_delete_then_merge.jsonl` has `main_index_label_size=0`, `label_sidecar_loadable=true`, `label_storage_mode=sidecar`.
- `experiments/pq_drift_1m_aris_main_20260522/raw/phaseB_zero_insert.jsonl` has `main_index_label_size=0`, `label_sidecar_loadable=true`, `label_storage_mode=sidecar`.

## Audit Conclusion

The code and metadata support the strong layout claim for the audited sidecar builds:

> With `--label-storage sidecar`, the main disk node record has `label_size=0`; labels are stored and loaded from `<index_prefix>_labels.densebit` sidecars, with hybrid metadata in `<index_prefix>_hybrid.meta`.

Scope caveats:

- This claim is about labels, not tag/id maps. Dynamic indexes may still have `_disk.index.tags` as a separate serving file.
- This does not by itself satisfy strict index expansion <=1x, because the sidecar removes embedded labels but the main disk layout still stores raw vectors, adjacency, and sector slack.
- The audited sidecar files are v2 mixed densebit/posting files; older v1 sidecars remain loadable by code but are not the main evidence for this run.

Machine-readable sidecar header audit: `label_sidecar_header_audit.jsonl`.
