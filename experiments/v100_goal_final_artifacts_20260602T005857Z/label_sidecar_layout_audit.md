# V100 Label Sidecar Layout Audit

- Repack rows audited: `10`
- Rows with `label_size=0`: `10`
- Layout violations: `0`

The Supersector32K replay evidence must preserve label storage in `_labels.densebit` sidecars. This audit marks the claim pass only when every repacked main disk index reports `label_size=0`; label sidecars, tag maps, and metadata remain counted in strict serving footprint.
