# PPT-Ready Conclusions and Figures

## Slide 1: Dynamic Update Latency Now Meets Target

- Original PhaseC selected set had 7/200 points with avg latency >10ms; worst was 13.681ms avg and 18.353ms p95.
- Targeted route/L reselection replaces exactly those 7 points; merged selected set is 200/200 recall@10 >=98, avg <10ms, and p95 <10ms.
- New max avg latency: 9.949ms; new max p95 latency: 9.973ms.
- Caveat: range replacements are configured graph but actual route is mixed/fallback, not pure graph.

Figure: `figures/latency_before_after_avg.png`
Figure: `figures/latency_before_after_p95.png`

## Slide 2: PQ Drift Matched-Reference Fixed Without Foreground Rebuild

- Original matched-reference result was 99/100; only miss was cycle5 range-u30.
- No-retrain expanded prefilter L=420 reaches recall@10 99.42 vs target 99.41, avg 5.100ms, p95 7.194ms.
- This makes PhaseC matched-reference 100/100 under the retuned strategy.
- Triggered retrain remains a fallback, but full rebuild is not needed for this miss.

Figure: `figures/pq_drift_unmatched_l_sweep.png`

## Slide 3: Space Claim Must Use Two Denominators

- Strict serving footprint for dynamic 1M after-insert is 1.041x excess over raw vectors, so strict <=1x is not currently met.
- The previously observed ~1.03x is the no-tag serving口径: 1.033x.
- Main-index-only is essentially exactly 1x extra (1.000008x) because R=116 fixed-sector packing leaves 512B/vector of adjacency+slack over a 512B vector.
- Removing sector slack is analytically enough: same R packed estimate is 0.955x, but this is not implemented in this run.

Figure: `figures/index_space_excess_audit.png`

## Slide 4: 3-Minute Maintenance Window

- Existing evidence separates PQ train/recode from full rebuild: 16-core train 2.480s, recode 2.019s, combined 4.499s.
- Full build at 16 cores is 232.912s, so a foreground full rebuild violates the 3-minute window.
- Engineering strategy: keep foreground work to delete/merge and PQ train/recode; run full rebuild/repacking in background or during scheduled maintenance.

## Recommended Wording

- Say: "After dynamic updates, retuned route/L recovers recall@10 >=98 with selected avg and p95 latency <10ms in this 1M PhaseC evidence set."
- Say: "PQ drift matched-reference reaches 100/100 by retuning the sole no-retrain miss."
- Do not say: "same search parameters preserve graph quality" unless a fixed-parameter experiment is added.
- Say: "label sidecar removes embedded labels from the main node record; strict total serving footprint is still >1x under the current sector-packed layout."
