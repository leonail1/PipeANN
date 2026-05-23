# Dynamic Delete / Merge / PQ Drift ARIS Summary

Created UTC: `20260522T084952Z`

## Evidence Roots

- `phase12`: `/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase12_rerun`
- `phase3_5cycle`: `/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase3_5cycle_bigann6m`
- `phase4_flatfinal`: `/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase4_flatfinal`

## Claim-by-Claim Status

| # | Claim | Status | Evidence-backed summary | Do not overclaim |
|---|---|---|---|---|
| 1 | Current delete is mark/lazy delete; 60% delete is sub-ms/vector | PASS | `delete_count=600000`, `delete_wall_s=0.974438`, `avg=0.001624 ms/vector`. | Claim registry C1 is still not a full paper-grade static proof. |
| 2 | Mark-delete materialization/merge time under resource cap | WARN | `merge_wall_s=31.161`, `wall_s=32.170`, `cpu_cap=16`, `allowed_cpus=0-15`. | No watt/power measurement; state CPU cap only. |
| 3 | Repeated 60% delete + equal insert | WARN | `cycles=5`, delete live counts `[400000, 400000, 400000, 400000, 400000]`, insert live counts `[1000000, 1000000, 1000000, 1000000, 1000000]`; selected min recall `98.17`, max avg latency `5.42 ms`. | This is a 1M-live BigANN/SIFT-6M-prefix experiment, not full SIFT100M. |
| 4 | PQ drift from zero insert | WARN | 100k smoke: direct train `0.286s`, recode `0.327s`; zero insert `99.516s`, merge `2.650s`, zero min recall `99.64`. | 100k flat-until-final smoke only; not 1M or long online insertion. |
| 5 | Recall means retuned search parameters can reach 98% | PASS | Selected rows use `post_hoc_retuned_fastest_feasible_recall_ge_98`; Phase3 has `20/20` selected points passing, Phase4 has `8/8`. | Do not claim fixed-parameter graph quality is unchanged or held-out generalization. |
| 6 | Labels stored in sidecar, main index label payload removed | WARN | Phase2 has `main_index_label_size=0`, `label_sidecar_loadable=True`. | Current evidence is sidecar smoke with disk format v1; full v2 format remains unsupported. |

## Key Metrics

- Phase3 delete API ms/vector: `[0.001596, 0.006047, 0.006403, 0.005532, 0.006267]`; average `0.005169`.
- Phase3 delete-side merge seconds: `[35.072, 33.775, 33.276, 32.481, 34.574]`.
- Phase3 insert seconds: `[536.895, 561.327, 562.231, 572.979, 576.593]`.
- Phase3 insert-side merge seconds: `[24.286, 22.019, 20.192, 22.312, 24.063]`.
- Phase3 selected recall min/max: `98.17` / `100.00`.
- Phase3 selected latency max avg/p95: `5.415 ms` / `5.819 ms`.

## Figures

- `figures/phase12_delete_merge.png`
- `figures/phase12_delete_merge.pdf`
- `figures/phase3_5cycle_costs.png`
- `figures/phase3_5cycle_costs.pdf`
- `figures/phase3_selected_recall_latency.png`
- `figures/phase3_selected_recall_latency.pdf`
- `figures/phase4_pq_train_recode.png`
- `figures/phase4_pq_train_recode.pdf`
- `figures/phase4_selected_latency.png`
- `figures/phase4_selected_latency.pdf`

## ARIS Review Position

- Existing evidence supports smoke/local claims with raw JSONL and logs.
- New 1M PQ drift work must create a fresh claim registry and bind direct commands, input hashes, pivot hashes, raw calibration rows, and review reports.
- Claims without frozen evidence should be labelled `UNSUPPORTED`; claims with 100k-only evidence should remain `SMOKE`.
