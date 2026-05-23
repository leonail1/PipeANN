# Final Data Analysis: Dynamic Delete, Merge, PQ Drift, and Label Sidecar

Scope: `R=116`, `PQ16`, CPU cap `16` unless otherwise stated. Main 1M data uses
the BigANN/SIFT 6M prefix as six consecutive 1M segments. This is not evidence
for the full SIFT100M distribution.

## 1. Current Delete Method and 60% Delete Latency

Claim status: `PASS` for mark-delete behavior and sub-millisecond delete cost
under the tested driver path.

Evidence:

- `results/dynamic_delete_pq_drift_aris_20260522_phase12_rerun/raw/phase1_delete_only.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_delete_steps.jsonl`

Observed numbers:

- Phase 1/2 rerun, 60% delete of 1M index: `600000` deletes in `0.974438s`,
  about `0.00162 ms/vector`.
- Phase C 1M main, five delete rounds:
  - cycle 1: `0.928589s`, `0.00155 ms/vector`
  - cycle 2: `3.371458s`, `0.00562 ms/vector`
  - cycle 3: `3.763496s`, `0.00627 ms/vector`
  - cycle 4: `3.397291s`, `0.00566 ms/vector`
  - cycle 5: `4.204679s`, `0.00701 ms/vector`
- Five-cycle mean: `0.00522 ms/vector`; max: `0.00701 ms/vector`.

Interpretation:

The delete path is effectively a metadata/tombstone update, not an immediate
physical rewrite of vectors and adjacency. The measured latency is far below
the requested "零点几毫秒一条" threshold. This supports the claim that current
delete is mark-delete style and can delete 60% of vectors at sub-millisecond
per-vector cost.

## 2. Merge / Materialization Time After Mark Delete

Claim status: `PASS` for measured CPU-capped merge cost; no power claim is made.

Evidence:

- `results/dynamic_delete_pq_drift_aris_20260522_phase12_rerun/raw/phase2_delete_then_merge.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_delete_steps.jsonl`

Observed numbers:

- Phase 1/2 delete-then-merge: total elapsed `32.169937s` after 60% delete.
- Phase C delete merge wall times across five cycles:
  `35.863939s`, `36.237252s`, `31.997058s`, `34.458514s`, `35.443117s`.
- Five-cycle mean merge time: `34.80s`; max: `36.24s`.

Interpretation:

Materializing 60% tombstone delete into a compacted live index is tens of
seconds at 1M scale under CPU cap 16. This is much slower than mark-delete
itself, but still clearly bounded and repeatable in the evidence. The result
should be reported as CPU-capped wall time, not power consumption.

## 3. Repeated Delete 60% + Insert Different Vectors

Claim status: `PASS` for five 1M live-set cycles; `WARN` for extrapolating to
full SIFT100M.

Evidence:

- `results/dynamic_delete_pq_drift_aris_20260522_phase3_5cycle_bigann6m/raw/phase3_cycles.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_delete_steps.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_no_retrain_cycles.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_selected_route_l.jsonl`

Observed operation cost in 1M main:

- Each cycle deletes 600k vectors, merges to 400k live points, inserts 600k
  vectors from the next segment, then merges back to 1M live points.
- Insert times:
  `578.395s`, `585.811s`, `578.707s`, `589.737s`, `595.356s`.
- Insert merge times:
  `23.312s`, `21.676s`, `21.298s`, `24.739s`, `24.182s`.
- Live count after each insert: `1000000`.

Search quality after retuning:

- Phase C selected rows: `200/200`.
- All selected rows support `recall@10 >= 98`; min recall is exactly `98.0`.
- Fastest-feasible selection came from complete calibration grids:
  `2 routes x 8 L values` per condition.
- Max selected average latency: `13.681 ms`.
- Rows over `10ms`: `7/200`, all in Phase C high-selectivity cases.

Interpretation:

The evidence supports "after repeated delete/insert, recall can still be brought
to at least 98% by retuning route/L." It does not support the stronger claim
that graph quality or latency is unchanged under identical search parameters.
The user-specified interpretation is the correct one: the system can recover
the target recall by changing search parameters within the calibration sweep.

## 4. PQ Drift From Zero Insert and Retraining Cost

Claim status: `PASS` for 1M BigANN/SIFT-prefix PQ drift measurement; `WARN` for
one matched-reference unmatched point in Phase C.

Evidence:

- `results/pq_drift_1m_aris_main_20260522/raw/phaseB_pq_drift.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseB_penalty.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseC_penalty.jsonl`
- `results/pq_drift_1m_aris_main_20260522/raw/phaseD_pq_core_sweep.jsonl`

Phase B direct retrain vs zero insert:

- Direct retrain, 1M points:
  - `pq_retrained=true`
  - PQ train `2.566s`
  - PQ recode `1.687s`
- Zero insert, no retrain:
  - inserted `1000000` points
  - insert wall `2443.381s`
  - merge wall `15.341s`
  - seed PQ points `100000`
  - `seed_pivot_hash_matches_final=true`
- Phase B selected rows: `40/40`; all `recall@10 >= 98`.
- Phase B fastest-feasible latency delta, no-retrain minus retrain:
  - min `-1.850ms`
  - median about `-0.0004ms`
  - max `+0.998ms`
  - matched-reference comparisons: `20/20` matched.

Phase C repeated drift:

- Phase C selected rows: `200/200`; all `recall@10 >= 98`.
- No-retrain vs retrain fastest-feasible latency delta:
  - min `-0.767ms`
  - median `+0.113ms`
  - max `+5.406ms`
- Matched-reference status: `99/100` matched.
- One unmatched point:
  - `cycle_idx=5`, `selector_type=range`, `bucket=u30`
  - retrain recall `99.51`, matched target `99.41`
  - no-retrain selected recall `98.85`
  - fastest-feasible latency delta still only `+0.0557ms`

PQ retrain / recode core sweep:

| Cores | Total build wall time | PQ train | PQ recode |
|---:|---:|---:|---:|
| 1 | `2094.99s` | `20.322s` | `7.343s` |
| 4 | `762.62s` | `6.269s` | `2.477s` |
| 8 | `368.73s` | `3.601s` | `1.872s` |
| 16 | `232.91s` | `2.480s` | `2.019s` |

Interpretation:

On the tested 1M SIFT-prefix data, PQ drift did not prevent reaching
`recall@10 >= 98` after retuning. The cost is mainly in insertion/build, not PQ
training itself; 16-core PQ training is only about `2.48s`, and recoding is
about `2.02s` in this evidence. The caveat is that matched-reference recall is
not perfect across every point: one Phase C point could not match
`reference recall - 0.1pp`.

## 5. Correct Meaning of "Insert/Delete Does Not Affect Recall"

Claim status: `PASS` under the corrected definition.

Correct definition:

After insert/delete, the system is allowed to recalibrate search parameters.
The claim is that there exists a feasible `route/L` under the tested sweep that
achieves `recall@10 >= 98%`, not that recall stays unchanged at the same
parameters.

Evidence:

- Phase 3 original 5-cycle selected rows: `20/20`, all pass recall target.
- Phase C 1M main selected rows: `200/200`, all pass recall target.
- Complete calibration evidence retained: `3840` calibration rows for the 1M
  main, with failed/non-feasible rows retained.

Interpretation:

The claim should be written as "after dynamic updates, retuning route/L can
recover recall@10 >= 98% on the tested workloads." Do not write "graph quality
does not degrade" or "same search parameters preserve recall" unless a separate
fixed-parameter experiment is run.

## 6. Label Sidecar / Main Index Format

Claim status: `PARTIAL PASS` for implemented sidecar path and evidence; `WARN`
until a dedicated binary-layout audit is kept with the final commit.

Evidence:

- 1M main commands consistently pass `--label-storage sidecar`.
- Driver and build logs record sidecar mode and separate label input.
- Earlier storage-design work moved labels out of the main index path by using
  sidecar metadata for label/filter data.

Interpretation:

The experiments used sidecar label storage in command provenance. That supports
the operational claim that the run was configured with sidecar labels. For a
paper or final defense claim that "main index contains only raw vector +
adjacency", keep a separate code-level audit showing the exact v2 record
layout, sector packing, and absence of embedded label payload in the main index.

## ARIS Review Result

Independent subagent review of the 1M main run: `WARN`.

Reasons:

- One Phase C matched-reference point is unmatched.
- The local evidence bundle intentionally omits large data/index/truth
  artifacts; provenance is through inventories, hashes, commands, and logs.
- Large input hashes in the pulled evidence are prefix hashes, not full local
  rehashes.

Overall conclusion:

The main technical claims are supported with caveats:

- Mark delete is extremely fast at 60% delete.
- Physical merge is tens of seconds at 1M scale under CPU cap 16.
- Five cycles of delete/insert keep live count at 1M and can recover
  `recall@10 >= 98` after retuning.
- PQ drift does not block the 98% recall target in this 1M SIFT-prefix test,
  but it can increase latency in some high-selectivity points and has one
  unmatched matched-reference comparison.
- Dataset scope must be stated as BigANN/SIFT 6M prefix, not full SIFT100M.
