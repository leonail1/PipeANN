# PQ-only Rebuild Design

Scope: retrain PQ pivots and recode PQ compressed vectors only. The serving prefix hardlinks or copies the source non-PQ files, then writes fresh `*_pq_pivots.bin` and `*_pq_compressed.bin` for the destination prefix. It does not rebuild the disk graph, compact live records, rewrite layout metadata, clean extra tombstones beyond the durable incremental merge used to materialize delete/insert operations, or change tag/id maps.

Critical ordering rule: PipeANN disk ids are not raw vector row ids. The runner reads `*_disk.index.tags`, materializes a temporary data bin in disk-id/tag order, and uses that temporary bin only as the PQ train/recode input. The temporary file is not part of the serving prefix; it exists so PQ code row `i` corresponds to disk node `i` while preserving all non-PQ serving files byte-for-byte.

Safety invariants:
- `pq_bytes` / `n_chunks` stays fixed.
- PQ code rows must equal the live vector file row count.
- Source and destination disk files must be same-inode hardlinks when possible, or byte-equivalent copies.
- PQ sidecars are written under a fresh destination prefix, so search never observes half-written source files.

Experiment caveat: the chain uses existing incremental delete/insert merge to create a durable prefix. That is not a full graph rebuild, but it means a separate no-merge in-memory experiment is needed for a pure tombstone-accumulation claim.

Configured gates: recall >= 98.0, avg latency < 10.0 ms, p95 latency < 10.0 ms.
