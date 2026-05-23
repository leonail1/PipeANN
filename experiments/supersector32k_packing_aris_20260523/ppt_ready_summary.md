# PPT-Ready Summary: Supersector32K v3 Page-Aware Packing

## Slide 1: Claim Status
- Dynamic selected workload: 200/200 rows meet recall@10 >= 98 and avg latency <10ms.
- PQ drift matched-reference: cycle05 range-u30 L420 recall@10 = 99.42, meeting the >=99.41 matched-reference threshold.
- Space: strict total serving footprint is 1.980615x raw; strict extra space over raw is 0.980615x raw, below the <1x extra-space acceptance metric.
- IO primitive remains 4KB: search evidence reports physical_read_unit_bytes=4096 and per_node_read_request_bytes=4096.

## Slide 2: Latency Cleanup
- Full v3 before targeted cleanup: 193/200 avg<10ms.
- Final optimized result: 200/200 avg<10ms, min recall@10 = 98.00.
- Seven targeted replacements were used; six are route/L refinements, one is a triggered-retrain fallback for cycle04 no-retrain range-u75.
- p95 caveat: max final p95 = 10.320ms, so p95 is improved but not universally below 10ms.

## Slide 3: Space And Layout
- Page-aware 32KB logical block keeps 33 nodes/block for max_node_len=980.
- Straddling nodes drop from dense 7/33 to 5/33, reducing expected 4KB reads per node from 1.212121 to 1.151515.
- Disk index bytes remain 993,005,568; large v3 indexes are stored outside git under /mnt/bak3/lzg/PipeANN-supersector32k-work/indexes.

## Slide 4: Engineering Policy
- Serving snapshots use v3 Supersector32K page-aware packing.
- Foreground dynamic insert/delete/merge remains on v1; v3 is produced by background repack.
- Triggered retrain/repack is recommended when no-retrain route/L cannot satisfy both recall and latency, as seen for cycle04 no-retrain range-u75.
