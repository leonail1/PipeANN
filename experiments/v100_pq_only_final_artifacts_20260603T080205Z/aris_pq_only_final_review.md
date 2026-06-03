# PQ-only ARIS Final Review

Verdict: **PARTIAL PASS / FAIL FOR 5-CYCLE LATENCY**.

## What Passed
- PQ-only scope: PASS. The tested rebuild path changes PQ pivots/codes and preserves non-PQ serving files by hardlink/full-file hash.
- Smoke: PASS. Recall 98.05, avg 8.442 ms, p95 8.774 ms.
- Early chain rows: cycles 1-2 plus cycle3/u50 pass before fast-fail; 13 rows pass all gates.
- Background overlap: PASS_WITH_CAVEAT. One 1-core PQ-only rebuild ran while 5 foreground searches stayed under avg/p95 10 ms.

## What Failed
The full PQ-only 5-cycle acceptance failed at `cycle03_pq_only_chain_intersect_u75`. Recall remained acceptable at 98.25, but avg latency was 10.959 ms and p95 was 12.330 ms.

## Root Cause Evidence
The failing point is not solved by further PQ sidecar refresh alone. Graph route maxes out at 97.40 recall by L500; the first recall-passing prefilter point requires L400, 750k comparisons, and about 400 4KB reads. The full-triggered rebuild analog passes latency, indicating non-PQ degradation: graph topology, incremental layout/packing, high prefilter candidate pressure, and compact/space policy.

## Scope Guardrails
This PQ-only run does not prove 4KB layout, strict space ratio, label sidecar, tombstone-only accumulation, full graph rebuild interference, live compact, or live publish/swap claims. Those must remain separate evidence tracks.
