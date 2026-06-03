# V100 Index Space Audit

- Space rows audited: `10`
- Best strict total/raw: `1.980615x`
- Worst strict total/raw: `1.980615x`
- Worst strict excess/raw: `0.980615x`
- Strict口径 counts active v3 disk index, PQ codes/pivots, tag/meta files, and label sidecar.
- Engineering口径 reports excess over raw separately; transient v1 source copies and repack workspace are excluded.
- The revised acceptance target is strict total serving footprint `<2x` raw vector bytes.
