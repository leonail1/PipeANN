# Non-PQ Degradation Diagnosis

## Outcome
PQ-only sidecar maintenance is correctly scoped and fast, but it is not sufficient for the full 5-cycle dynamic-update latency requirement.

- Smoke: PASS, recall 98.05, avg 8.442 ms, p95 8.774 ms.
- Chain before stop: 14 selected rows, 1 gate failure, min recall 98.08, max avg 10.959 ms, max p95 12.330 ms.
- Fast-fail: `cycle03_pq_only_chain_intersect_u75` reached recall 98.25, but avg 10.959 ms and p95 12.330 ms.

## Why This Is Not PQ Drift Alone
The failure curve shows graph search cannot reach recall 98 by L500; its best recall is 97.40. The first recall-passing prefilter point is L400, but it scans 750,000 candidates and reads about 400 4KB pages, pushing latency over the 10 ms gate.

The matching full-triggered rebuild row passes the same case: recall 98.14, avg 9.277227027 ms, p95 9.559427 ms. That comparison points to residual non-PQ state: graph topology after repeated replacement, incremental layout/packing effects, prefilter candidate pressure, and eventually live-record compaction/space policy.

## Decoupled Maintenance Strategy
Keep PQ-only rebuild as the frequent, low-impact PQ drift response: it only rewrites PQ pivots/codes and can run in the background. Schedule non-PQ maintenance independently:

1. Trigger graph/layout refresh when graph route cannot meet recall>=98 below the latency-safe L window, or when selected route falls back to high-candidate prefilter.
2. Trigger live-record compact/space maintenance on space or fragmentation thresholds, not on PQ drift.
3. Keep tombstone cleanup and graph rebuild out of the foreground path; run them at low peak or on shadow prefixes with an atomic publish step.
4. Continue using prior layout/space audit for 4KB packing and strict-space claims; PQ-only rebuild does not change or re-prove those claims.

## Caveats
This chain uses durable incremental delete/insert merge between cycles. It does not prove pure in-memory tombstone accumulation, live prefix swap, full graph rebuild interference, or compact/layout interference.
