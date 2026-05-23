# Dynamic Delete / Merge / PQ Drift ARIS Status

Date: 2026-05-22

## Evidence Roots

- Phase1/2 rerun: `node6:/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase12_rerun`
- Phase3 smoke: `node6:/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase3_smoke`
- Phase4 flat-final smoke: `node6:/mnt/bak3/lzg/PipeANN-github/experiments/dynamic_delete_pq_drift_aris_20260522_phase4_flatfinal`

## Claim Status

| Claim | Status | Evidence |
| --- | --- | --- |
| Current delete is mark/lazy delete | PASS with C1 registry WARN | Static review and Phase1/2 raw evidence show delete loop only marks live/tombstone state; main index label payload remains size 0 in sidecar builds. |
| Delete 60% vectors in sub-ms/vector | PASS-SMOKE | Phase1 delete_count=600000, delete_wall_s=0.974438, avg=0.001624 ms/vector. |
| Merge/materialize after 60% delete | PASS-SMOKE | Phase2 delete_count=600000, merge_wall_s=31.160545, live_point_count=400000, sidecar loadable. |
| Multi-round delete 60% + insert | WARN-SMOKE | One SIFT1M fallback cycle only: delete 600k, insert 600k, live count restored to 1M; no SIFT100M main run. |
| PQ drift from zero insert | PASS-SMOKE | Phase4 flat-final: zero insert requested/live/code all 100000; seed PQ no-full-retrain selected points all recall@10 >= 99.64. |
| Label sidecar / main index label removal | PASS-SMOKE | Phase2 and Phase4 main_index_label_size=0 and densebit sidecar loadable; full v2 format claim remains unsupported. |

## Key Metrics

Phase1:
- `delete_count=600000`
- `delete_wall_s=0.974438`
- `avg_delete_ms_per_vector=0.001624`
- `live_point_count=400000`

Phase2:
- `delete_wall_s=1.009392`
- `merge_wall_s=31.160545`
- `wall_s=32.169937`
- `live_point_count=400000`
- `main_index_label_size=0`
- `label_sidecar_loadable=true`

Phase3 smoke:
- SIFT100M was absent; used wrapped SIFT1M fallback.
- One cycle only.
- Delete step: `delete_count=600000`, `live_point_count=400000`
- Insert step: `insert_count=600000`, `live_point_count=1000000`
- Selected smoke points all reached `recall@10 >= 98%`.

Phase4 flat-final smoke:
- Direct build: `requested/live/code=100000`, PQ16, full-corpus PQ train.
- Zero insert: `insert_count=100000`, `live_point_count=100000`, `code_point_count=100000`, `flat_threshold=99999`.
- Zero insert time: `insert_wall_s=99.516096`, `merge_wall_s=2.650324`, `wall_s=102.166420`.
- Zero seed PQ: `seed_points=10000`, `pq_retrained=false`, seed pivots hash equals final zero pivots hash.
- Zero selected recall: `99.98, 99.64, 99.98, 99.64`.
- All 8 direct+zero selected points: min recall `99.64`, avg latency range `0.105-1.992 ms`.

## ARIS Review Notes

- Phase4 flat-final got `PASS-SMOKE`, not full PASS.
- Route/L selection is post-hoc fastest feasible with recall@10 >= 98 on the same 1000-query/GT set; do not present it as held-out generalization.
- Phase4 zero path is `flat_until_final_materialization`; it does not measure long-running online disk insertion after 10k.
- The Phase3 main claim over SIFT100M remains unsupported until the dataset is available and main cycles are run.
- Full paper-grade PQ drift still needs all buckets, held-out evaluation or pre-registered route/L, multiple seeds, scale sweep, triggered retrain comparison, and complete artifact manifests.
