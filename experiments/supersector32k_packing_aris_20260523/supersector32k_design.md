# Supersector32K Packing Design

## Goal

Reduce the strict PipeANN serving footprint below `1.0x` raw-vector bytes without reducing `R=116` and without changing the 4KB random-read primitive.

## Current Footprint Problem

For the PhaseC 1M dynamic after-insert index, the node record is `980B`:

- `512B` vector payload (`dim=128`, float32)
- `468B` graph payload (`nnbrs + R neighbor ids`, `R=116`)
- `0B` embedded labels (`--label-storage sidecar`)

The current layout packs `floor(4096 / 980) = 4` nodes per 4KB sector and leaves `176B` slack per sector, or `44B/vector`. That makes `_disk.index` effectively `512B/vector` over raw-vector payload, before PQ, sidecar labels, hybrid metadata, and tag map. Strict dynamic serving footprint is therefore `1.041159x` over raw bytes.

## Proposed Layout: v2 Supersector32K Packing

Treat `32KB` as a logical packing block but not as the query read size:

- `layout_block_bytes = 32768`
- `layout_nodes_per_block = floor(32768 / max_node_len) = 33`
- physical byte offset for loc `i`:
  `4096 + (i / 33) * 32768 + (i % 33) * 980`
- first 4KB page:
  `floor(byte_offset / 4096)`
- offset inside first page:
  `byte_offset % 4096`
- 4KB pages to read:
  `ceil((page_offset + max_node_len) / 4096)`; this is `1` or `2` for the current 980B record.

The search path never reads a 32KB block, and it also does not represent a straddling node as a single 8KB request. For a node fully inside one page it issues one 4KB `IORequest`. For a straddling node it issues two separate 4KB `IORequest`s into adjacent scratch pages, so `DiskNode` sees a contiguous record. This preserves the 4KB random-read primitive while eliminating almost all 4KB-sector internal fragmentation.

## Expected Space

For `1,000,000` points and `max_node_len=980`:

- v1 `_disk.index`: `4096 + ceil(1,000,000 / 4) * 4096 = 1,024,004,096B`
- v2 `_disk.index`: `4096 + ceil(1,000,000 / 33) * 32768 = 993,005,568B`
- disk-index saving: `30,998,528B`

Using the previous strict serving components, the estimated strict excess drops from `1.041159x` to about `0.980616x`, below the `1.0x` acceptance line.

## Implementation Scope

This iteration implements a read/search serving layout and a repack tool:

- extend `SSDIndexMetadata` with layout-version fields while keeping old v1 indexes loadable;
- teach `SSDIndex`/`DiskNode` how to locate records in v1 or v2 layouts;
- make beam, pipe, and coro search issue one or two separate 4KB reads per node, not one 8KB or 32KB read;
- reject page-search/update-in-place operations on v2 indexes for now, because those paths assume one mutable page contains an integral number of nodes;
- add a `repack_disk_index_layout` utility that converts an existing v1 serving prefix to v2 and copies sidecars.

Dynamic delete/insert/merge continue producing v1. The engineering strategy is: foreground dynamic update remains unchanged; background/maintenance compaction repacks the serving snapshot to v2. This matches the requirement to reduce strict serving footprint without trading away `R` and recall. The strict-footprint denominator for the v2 claim counts only the active v2 serving prefix plus sidecars actually loaded by search; it excludes transient repack workspace and any retained v1 source copy.

## Claim Boundaries and Risks

- The `<1.0x` strict claim is measured for the PhaseC cycle05 no-retrain after-insert configuration (`N=1M`, `dim=128`, `R=116`, `max_node_len=980`, sidecar-label mode, PQ16, existing hybrid/tag sidecars). It is not a universal bound for every future dataset or `R`.
- The margin is small, about `9.9MB` below the `1.0x` strict line for the current file set, so the final registry must cite measured bytes rather than only the estimate.
- Seven of every 33 records straddle a 4KB boundary, so graph-style search may perform about `1.212x` physical 4KB page reads per expanded node. Latency must be remeasured.
- The v2 repacker currently hard-rejects `max_node_len > 4096`; this bounds each node to at most two separate 4KB page reads. The current target has `max_node_len=980`.
- v2 is serving-only in this patch. Page search, in-place insert, and delete merge hard-fail on v2 until their allocation/mutation model is rewritten.
- v2 does not copy v1 partition sidecars because those encode 4KB-sector page layout; repack also removes stale destination partition sidecars and the loader forces identity mapping for v2. Equal id-to-loc mapping is used for the repacked serving snapshots in this evidence set.
- The experiment runner writes large repacked indexes outside the git repo by default (`/mnt/bak3/lzg/PipeANN-supersector32k-work/indexes`); only compact JSONL/CSV/Markdown summaries under `experiments/` are commit candidates.
- Acceptance checks should scan generated requests and fail if any v2 per-node read has `len != 4096` or any read touches `32768B`.

## Acceptance Evidence

Smoke evidence must show:

- v2 metadata loads correctly;
- v2 `_disk.index` size matches the packed formula;
- representative search returns valid recall/latency;
- observed read unit count is reported as separate 4KB pages, with straddling nodes counted as two 4KB page reads.

Full evidence must show:

- strict serving footprint `< 1.0x` for PhaseC cycle05 no-retrain after-insert;
- targeted latency/PQ-drift representative searches still meet the previously accepted recall/latency policy;
- ARIS claim registry clearly distinguishes implemented v2 serving layout from the old v1 dynamic-write layout.
