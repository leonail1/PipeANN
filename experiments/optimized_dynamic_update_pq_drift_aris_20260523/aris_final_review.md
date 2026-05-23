# ARIS Final Review

Generated: 2026-05-23T02:58:44Z

## Verdict Summary

| Area | Verdict | Evidence |
|---|---|---|
| Dynamic selected latency avg <10ms | PASS | `optimized_dynamic_update_results_summary.json` |
| Dynamic selected latency p95 <10ms | PASS | `optimized_dynamic_update_results_summary.json` |
| Recall@10 >=98 under retuned route/L | PASS | `optimized_dynamic_update_results.jsonl` |
| PQ drift matched-reference 100/100 | PASS | `pq_drift_strategy_compare_summary.json` |
| Label sidecar: main index stores no labels | PASS | `label_sidecar_layout_audit.md` |
| Strict index expansion <=1x | FAIL CURRENT LAYOUT | `index_space_audit.md` |
| Foreground 3-minute maintenance | PASS for PQ train/recode/merge; FAIL if full rebuild blocks | `phaseD_pq_core_sweep.jsonl` |
| Fixed-parameter recall unaffected | UNSUPPORTED | no fixed-parameter experiment in this run |

## Key Results

### Phase2 Dynamic Latency

The optimized selected set has 200 rows and replaces exactly 7 original slow points. It now has:

- recall pass count: 200/200
- avg latency <10ms: 200/200
- p95 latency <10ms: 200/200
- min recall@10: 98.0
- max avg latency: 9.949428ms
- max p95 latency: 9.972903ms

Caveat: the 4 range replacements are configured as graph route but report `actual_route=mixed` and `fallback_count=1000`, so they must not be described as pure graph executions.

### Phase3 PQ Drift

The sole original matched-reference miss was cycle5 range-u30. A targeted no-retrain prefilter sweep matched the reference target at L=420:

- strategy: `no_retrain_expanded_prefilter_L420`
- recall@10: 99.42 vs target 99.41
- avg latency: 5.100167ms
- p95 latency: 7.193725ms

This supports the claim that PhaseC matched-reference can be 100/100 under the retuned strategy. `partial_recode` and `larger_seed_pq` are included as design-only rows, not experimental passes.

### Phase4 Space and Layout

Strict serving footprint for the dynamic 1M after-insert index is currently above the <=1x target:

- strict excess over raw vectors: 1.041159x
- no-tag serving excess: 1.033347x
- main-index-only exact excess: 1.000008x
- no-sector-slack packed estimate: 0.955222x

The no-tag serving口径 explains the previously observed ~1.03x number. Strict <=1x is not achievable with the current R=116 4KB sector-packed layout because the main disk index already stores raw vectors plus roughly one raw-vector-width of adjacency/slack per vector.

Label layout audit supports the sidecar claim: `--label-storage sidecar` passes `main_index_label=nullptr`, audited disk metadata has `label_size=0`, and labels load from `<prefix>_labels.densebit`. This is a label claim only; tag/id maps remain separate files and must be counted under strict serving footprint when present.

### Maintenance Window

Existing 16-core PhaseD timing separates core PQ maintenance from full rebuild:

- PQ train: 2.47967s
- PQ recode: 2.01885s
- train + recode: 4.49852s
- full build: 232.91205s

Therefore the safe engineering claim is: foreground PQ train/recode can fit easily under 3 minutes; a foreground full rebuild cannot and must be background/offline.

## Independent Review Record

| Phase | Reviewer | Verdict | Notes |
|---|---|---|---|
| Phase0 | `019e528c-ba91-7de0-8202-8295067d0fe7` | WARN then fixed | Missing not-passing claims were added to registry. |
| Phase1 | `019e52a0-5c34-7193-853c-9b39c2bb084d` | WARN then fixed | Route caveat and p95 wording fixed. |
| Code review | `019e52a2-0b85-7fd2-b93a-d126ce48f33d` | PASS | Expanded L sweep / `--l-sweep` change reviewed before tests/rerun. |
| Phase2 evidence | `019e52a7-f4b7-7ab1-a6fc-ccdfb19451b6`, `019e52ac-18a6-7273-b917-c72688884367` | Metrics PASS, workflow-artifact WARN | File-visible metrics pass; node6 artifacts alone cannot timestamp the conversation-level code-review PASS. |
| Phase3 evidence | `019e52ae-9a7a-7f10-848d-59de4f73b24a` | PASS | PQ drift matched-reference fix verified. |
| Phase4 evidence | `019e52b6-c957-7422-8e2f-040baacc9bd6` | Evidence PASS, tool-flow WARN | Space/layout calculations verified; reviewer reported its own subagent/tool reconnect issue. |

## Allowed Claims

- After dynamic updates, retuned route/L recovers recall@10 >=98 for all selected PhaseC points, with avg and p95 latency <10ms in this evidence set.
- PQ drift matched-reference reaches 100/100 by retuning the sole no-retrain miss to prefilter L=420.
- Labels are sidecar-only for audited builds: the main disk node record has `label_size=0`.
- Foreground PQ train/recode is <3 minutes; full rebuild must not be foreground-blocking.

## Claims Not Allowed

- Do not claim strict total serving footprint <=1x for the current implementation.
- Do not claim fixed-parameter recall/graph quality is unaffected.
- Do not claim range replacements are pure graph executions.
- Do not count design-only partial recode or larger seed PQ rows as experimental passes.

## Deliverables

- `optimized_claim_registry.json`
- `targeted_latency_profile.csv/jsonl`
- `optimized_dynamic_update_results.csv/jsonl`
- `pq_drift_strategy_compare.csv/jsonl`
- `index_space_audit.md`
- `label_sidecar_layout_audit.md`
- `ppt_ready_conclusion_summary.md`
- `figures/*.png`
