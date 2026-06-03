# PQ-only Space/Layout Note

PQ-only maintenance writes fresh PQ sidecars under a destination prefix and hardlinks/copies non-PQ files from the source prefix. Therefore it does not prove a new node packing or layout claim by itself. Disk graph/layout space evidence must be audited independently. The expected direct size delta is limited to replacing `*_pq_pivots.bin` and `*_pq_compressed.bin` with the same `pq_bytes`/row count.

Configured PQ bytes: `16`.
