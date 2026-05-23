# ARIS Dynamic Delete / Merge / PQ Drift Experiments

This directory is a commit-friendly index for the dynamic-delete and PQ-drift
experiment evidence. Large artifacts are intentionally excluded by `.gitignore`:
raw vector data, generated truth sets, indexes, PQ code files, and other binary
corpora should be regenerated from the recorded commands and inventories.

## Evidence Roots

- Phase 1/2 delete and merge rerun:
  `results/dynamic_delete_pq_drift_aris_20260522_phase12_rerun`
- Phase 3 original 5-cycle delete/insert run:
  `results/dynamic_delete_pq_drift_aris_20260522_phase3_5cycle_bigann6m`
- Phase 4 100k PQ drift smoke:
  `results/dynamic_delete_pq_drift_aris_20260522_phase4_flatfinal`
- Phase B/C/D 1M PQ drift main:
  `results/pq_drift_1m_aris_main_20260522`
- 1M main figures:
  `results/pq_drift_1m_aris_main_20260522/figures`

## Commit Scope

Commit these file classes:

- `claim_registry.json`
- `summary.json`
- `figures_manifest.json`
- `raw/*.jsonl` and `raw/*.csv`
- `evidence/*.json` and `evidence/*.md`
- `logs/*.log` when they are small enough for GitHub
- `figures/*.png` and `figures/*.pdf`
- review summaries and final Markdown reports

Do not commit these file classes:

- `data/`, `indexes/`, `truth/`
- generated vector/index/PQ binaries such as `*.bin`, `*.index`, `*.spmat`
- any single file over the GitHub limit or over the project-local large-file threshold

## Main Review Status

The 1M main run completed and passed local self-check. The independent ARIS
subagent review verdict was `WARN`, not `FAIL`.

Important caveats:

- Phase C matched-reference penalty has one unmatched row:
  `cycle_idx=5`, `selector_type=range`, `bucket=u30`.
- Local pulled evidence does not include the large data/index/truth artifacts.
  Provenance is through inventory, hashes, commands, and logs.
- Large input hashes are prefix hashes in the pulled inventory, not full
  rehashable local artifacts.
