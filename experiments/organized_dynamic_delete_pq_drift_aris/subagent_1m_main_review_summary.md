# Independent ARIS Review Summary: 1M PQ Drift Main

Reviewer: fresh subagent `019e51bf-1868-7013-bef7-a119d0dd9386`.

Overall verdict: `WARN`.

Checked evidence:

- `claim_registry.json`
- `summary.json`
- `figures_manifest.json`
- `evidence/input_inventory.json`
- `evidence/runner_config.json`
- `evidence/phase0_inventory.json`
- `evidence/tool_inventory.json`
- `evidence/driver_contract.json`
- `raw/commands.jsonl`
- `raw/selected_route_l.jsonl`
- `raw/phaseB_*`
- `raw/phaseC_*`
- `raw/phaseD_pq_core_sweep.jsonl`
- PhaseD build logs and phase0 env/git/cpu logs

Findings:

- Row counts are self-consistent:
  - PhaseB selected `40`
  - PhaseB penalty `20`
  - PhaseC selected `200`
  - PhaseC penalty `100`
  - PhaseC delete `5`
  - PhaseC no-retrain cycles `5`
  - PhaseD core sweep `4`
  - selected total `240`
- Selected route/L comes from complete calibration grids:
  `240` calibration JSONL files, `3840` rows total,
  `16` rows per condition (`2 routes x 8 L values`).
- Failed/non-feasible rows were retained:
  `2540` calibration rows had recall below `98`.
- All selected rows have `recall@10 >= 98` and
  `supports_recall_claim=true`; min recall is `98.0`.
- Max selected average latency is `13.681 ms`.
- `7` selected rows are above `10ms`, all in Phase C high-selectivity cases.
- No-retrain pivots stayed unchanged in Phase C cycles 1-5.
- PhaseB seed pivot matches final zero-insert no-retrain pivot.
- PhaseD core sweep is credible from logs and JSON:
  - 1 core: train `20.322s`, recode `7.343s`, total build `2094.99s`
  - 4 cores: train `6.269s`, recode `2.477s`, total build `762.62s`
  - 8 cores: train `3.601s`, recode `1.872s`, total build `368.73s`
  - 16 cores: train `2.480s`, recode `2.019s`, total build `232.91s`

Warnings:

- Phase C matched-reference penalty has one unmatched row:
  `selector_type=range`, `cycle_idx=5`, `bucket=u30`.
  Any matched-reference claim must say `99/100` matched, not all 100.
- The local pulled evidence package does not include large `data/`, `labels/`,
  `truth/`, or `indexes/` artifacts. Provenance is carried by inventories,
  hashes, and logged commands.
- Large inputs use prefix hashes in the pulled inventory, not full local
  rehashes.

Claim guidance:

- Supported:
  1M PQ drift against direct retrain; selected recall feasibility; Phase C
  delete/insert cycles; no-retrain pivot immutability; PhaseD PQ retrain/recode
  core sweep.
- Must qualify:
  dataset scope is BigANN/SIFT 6M prefix with 1M live corpus plus replacements,
  not full SIFT100M.
- Must downgrade:
  matched-reference PQ drift penalty is `99/100` matched with one explicitly
  reported unmatched row.
