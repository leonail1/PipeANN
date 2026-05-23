# Index Space Audit (ARIS Phase4)

## Scope and Denominators

This audit uses SIFT/BigANN-style float32 vectors with `dim=128`, so the raw vector payload is `512 bytes/vector`; the PipeANN `.bin` file denominator includes the 8-byte matrix header. Shadow indexes, source data, truthsets, logs, and build-only calibration files are not counted in serving footprint.

Two explicit footprint definitions are used:

- **Strict serving footprint**: `_disk.index` + `_pq_compressed.bin` + `_pq_pivots.bin` + `_labels.densebit` + `_hybrid.meta` + `_disk.index.tags` when present.
- **Engineering main-index footprint**: `_disk.index` only, with the raw vector payload subtracted to isolate graph/record/padding overhead. This is useful for explaining the current `~1.03x` number but is not the strict acceptance denominator.

## Measured Footprint

| Case | Points | Raw vector file | Disk index | PQ codes | Labels sidecar | Disk tags | Strict excess over raw | No-tag excess over raw | Main-index-only excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PhaseC cycle00 direct | 1,000,000 | 512.000 MB | 1024.004 MB | 16.000 MB | 0.931 MB | 0.000 MB | 1.033347x | 1.033347x | 1.000008x |
| PhaseC cycle05 no-retrain after insert | 1,000,000 | 512.000 MB | 1024.004 MB | 16.000 MB | 0.931 MB | 4.000 MB | 1.041159x | 1.033347x | 1.000008x |
| Phase12 after 60% delete merge | 400,000 | 204.800 MB | 409.604 MB | 6.400 MB | 0.373 MB | 1.600 MB | 1.041579x | 1.033766x | 1.000020x |

The `~1.03x` number is reproduced by the **no-tag excess** formula on the 1M index: `(disk + PQ codes + PQ pivots + label sidecar + hybrid meta - raw vector file) / raw vector file = 1.033347x`. The stricter dynamic serving case includes `_disk.index.tags`, which raises the excess to `1.041159x`.

## Layout Explanation

The current disk node layout is fixed-sector page packing:

- Metadata says `max_node_len=980 bytes`, `nnodes_per_sector=4`, `label_size=0` for the 1M PhaseC after-insert index.
- The record is `512B` vector payload plus `(R+1)*4 = 117*4 = 468B` adjacency/count storage, for `980B` before page slack.
- Only 4 records fit in a 4096B sector, so each sector has `4096 - 4*980 = 176B` internal slack, i.e. `44B/vector`.
- That gives a main index of `4096 + ceil(1,000,000 / 4) * 4096 = 1,024,004,096B`, matching the measured file.

Code references:

- `src/utils/index_build_utils.cpp:399-404` computes `label_size`, `max_node_len`, `nnodes_per_sector`.
- `src/utils/index_build_utils.cpp:408-454` writes full sector buffers even when internal slack remains.
- `include/ssd_index_defs.h:120-132` persists `label_size` in the disk metadata.
- `include/ssd_index_defs.h:136-138` documents the record as vector, neighbor count/ids, optional labels.

## Can Strict <=1x Be Met Now?

**No, not under the current layout and strict serving denominator.** With `R=116`, `dim=128`, and `label_size=0`, the main disk index already has essentially exactly `1.0x` extra bytes over the raw vectors: `512B/vector` for adjacency plus sector slack, plus one 4KB metadata sector. Any query-serving sidecar (`PQ`, labels, hybrid metadata, tags) pushes strict excess above `1.0x`.

Minimum current-code values:

- Main-index only, exact bytes: `1.000008x` excess over raw, because of the first metadata sector and file header denominator.
- Main-index only, amortized payload view: exactly `1.000000x` (`468B` adjacency/count + `44B` sector slack per `512B` vector payload).
- Strict serving footprint with dynamic disk tags: `1.041159x`.

## Optimization Options

| Option | Expected strict excess | Evidence status | Risk |
|---|---:|---|---|
| Keep current layout, exclude tags from denominator | 1.033347x | Measured | Does not satisfy strict serving footprint. |
| Keep current layout, include tags | 1.041159x | Measured | Fails strict <=1x. |
| Stream/packed node records, no 4KB sector slack, same `R=116` | 0.955222x | Estimated from measured metadata | Requires reader/layout change and IO benchmarking. |
| Reduce `R` enough to fit 5 records/sector (`max_node_len<=819`, roughly `R<=75`) | below 1x | Analytical only | Recall/latency not validated; likely high blast radius. |
| Compress neighbor IDs or variable-byte adjacency | potentially below 1x | Analytical only | Requires decoder changes in hot search path. |
| Move raw vector payload out of `_disk.index` and count only graph/PQ | can be below 1x by definition | Accounting/design only | Changes denominator; not a strict all-serving-files claim. |

## Phase4 Conclusion

The strict claim `index expansion <=1x` is **not supported** by current evidence. The strongest accurate statement is:

> Current label sidecar removes embedded labels from the main node record, but the current `R=116` sector-packed disk layout still leaves strict serving excess at `1.041159x` for the dynamic 1M after-insert index (`1.033347x` if disk tags are excluded). A layout change that removes sector slack can analytically bring the strict excess to about `0.955222x` without changing `R`, but that optimization is not implemented or benchmarked in this run.

Raw machine-readable measurements: `index_space_audit.csv` and `index_space_audit.jsonl`.
