# ARIS PPT Refresh Final Review

Date: 2026-05-24

Scope:
- Re-ran Supersector32K v3 packed serving evidence for the graduation deck.
- Regenerated PPT-ready figures and recompiled `PPT/graduation-ppt/dynamic-update.pdf`.
- Large repacked indexes remain outside git under `/mnt/bak3/lzg/PipeANN-supersector32k-work/indexes`.

Code Review:
- `scripts/generate_supersector32k_ppt_refresh.py` passed fresh code review after tightening evidence validation.
- The script fails loudly on missing dynamic/PQ/space/read/maintenance evidence.
- Reused evidence charts are explicitly named `ppt_reused_*`; mixed update/v3-repack timing is named `ppt_mixed_*`.
- PPT figure overwrites require `--allow-overwrite-ppt-figures`.

Smoke Evidence Review:
- Smoke repack/search passed.
- Repack rows were v3 `page_aware_slots`, 32KB layout blocks, 33 nodes/block, 4KB read pages.
- Smoke search rows recorded `physical_read_unit_bytes=4096` and `per_node_read_request_bytes=4096`.

Full Evidence Review:
- `raw/full_selected_super32k.jsonl`: 200 rows, 200 unique cases.
- `optimized_dynamic_update_results.jsonl`: 200 unique cases, min recall@10 = 98.00, max avg latency = 9.955 ms, 0 rows with avg latency >= 10 ms.
- P95 caveat: max p95 latency = 10.320 ms, with 4 selected points at p95 >= 10 ms.
- Targeted replacements: 7 optimized rows use replacement evidence; one case uses triggered retrain fallback.
- PQ drift: exact v3 row `cycle05_no_retrain_across_cycles_range_u30`, L420, recall@10 = 99.42.
- Space audit: strict total/raw = 1.980615x; strict excess/raw = 0.980615x.
- Repack evidence: 11/11 rows v3 page-aware, `read_page_bytes=4096`, `straddling_slots_per_block=5`, `avg_4k_pages_per_record=1.151515`, and `actual_disk_bytes == expected_disk_bytes`.

PPT Review:
- `dynamic-update.pdf` compiles to 15 pages.
- Old experiment image references were replaced by `ppt_v3_*`, `ppt_reused_*`, or `ppt_mixed_*` figures.
- New requirements are plotted: dynamic latency/recall, PQ drift strategy, strict total/excess space, 4KB read granularity, and maintenance window.
- The deck does not claim fixed-parameter graph quality is unchanged.
- The deck does not claim all p95 latencies are below 10 ms.

Caveats:
- v3 packed format is a serving snapshot after merge/repack; foreground insert/delete paths still use the v1 update layout.
- Strict total serving footprint is 1.981x raw; the passing 0.981x value is the strict excess-over-raw accounting.
- Four selected points still have p95 >= 10 ms, although all selected avg latencies are below 10 ms.
