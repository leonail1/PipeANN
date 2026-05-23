# Supersector32K Index Space Audit

- Raw vector file bytes: `512000008`
- Strict v3 serving bytes: `1014075148`
- Strict total over raw: `1.980615x`
- Strict excess over raw: `0.980615x`
- Components: `{"disk_index": 993005568, "disk_tags": 4000008, "hybrid_meta": 1856, "labels_sidecar": 931416, "pq_codes": 16000008, "pq_pivots": 136292}`

- Current v3 repack rows: `11`
- Page-aware slot packing keeps the 32KB block size but reduces straddling slots from 7/33 to 5/33.

The strict denominator counts the active v3 serving prefix and loaded sidecars only; transient repack workspace and retained v1 source indexes are excluded.
