# ARIS Final Evidence Review: Supersector32K v3 Page-Aware Packing

## Verdict
PASS with one explicit p95 caveat. The final optimized artifacts satisfy the primary acceptance metric for dynamic update recall/average latency, PQ matched-reference, 4KB read primitive, and strict extra-space ratio.

## Evidence Summary
- Optimized dynamic results: `200` rows, `200` unique case IDs.
- Recall/avg latency: min recall@10 `98.00`, max avg latency `9.955 ms`; rows failing avg<10ms: `0`.
- p95 caveat: max final p95 `10.320 ms` at `cycle03_no_retrain_across_cycles_range_u75`; not all p95 values are below 10ms.
- Targeted replacements: `7` selected rows replaced. The only triggered-retrain fallback is `cycle04_no_retrain_across_cycles_range_u75`; targeted no-retrain candidates for that row did not pass both recall and avg latency, while triggered retrain reaches recall@10 `98.04`, avg `9.088 ms`, p95 `9.614 ms`.
- PQ drift: `cycle05_no_retrain_across_cycles_range_u30` at L `420` has recall@10 `99.42` (threshold >=99.41).
- Space: strict total/raw `1.980615x`; strict excess/raw `0.980615x`.
- Layout: `11` repack rows are v3 page-aware slots with `straddling_slots_per_block=5`, `avg_4k_pages_per_record=1.151515`, and `read_page_bytes=4096`.

## Deliverable Map
- Claim registry: `optimized_claim_registry.json`
- Latency profile: `targeted_latency_profile.csv/jsonl`
- Dynamic results: `optimized_dynamic_update_results.csv/jsonl`
- PQ strategy compare: `pq_drift_strategy_compare.csv/jsonl`
- Space audit: `index_space_audit.md/jsonl/csv`
- Label sidecar audit: `label_sidecar_layout_audit.md`
- PPT-ready summary and figures: `ppt_ready_summary.md`, `ppt_ready_summary.csv`, `figures/latency_pass_count.svg`, `figures/space_components.svg`

## ARIS Notes
- Do not claim graph quality is unchanged under fixed parameters; the passing metric uses selected route/L and a triggered retrain fallback where required.
- Do not claim total serving footprint is below raw size. The supported space claim is strict extra space over raw `<1x`; total serving/raw is reported separately.
- Do not claim all p95 latency is below 10ms; p95 remains a residual improvement/caveat.
