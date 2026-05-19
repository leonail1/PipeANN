# Cleanup Notes

日期：2026-05-19

This directory is the retained final ARIS-style CPU run for r116.

## Kept

- `exp6_aris_cpu_clean/`: clean 320-job run with `u75` recalibrated to `graph L=75`.
- `exp4_intersect_range_selectivity/`: existing r116 requirement-3 result table and refreshed high-resolution plot.
- Existing `exp2`, `exp3`, `exp4_delete_reinsert_selectivity`, `exp5`, and `exp_baseline` directories: historical r116 suite results already tracked by the repository.

## Removed

- `exp6_aris_cpu/`: first ARIS-style CPU run. It contained mixed ledgers because `u75` was rerun after `L=50` failed recall.
- `exp6_query_thread_budget/`: interrupted temporary exp6 run and generated 1M disk index files. Its compact conclusions are superseded by `exp6_aris_cpu_clean/`; the large index files are reproducible and intentionally not tracked.

## Ignored Large Artifacts

The repository `.gitignore` already excludes large/generated experiment files such as:

- `*.bin`
- `*.densebit`
- `*.index`
- `*.index.tags`
- `*.log`
- `*.meta`
- `*.spmat`

Only compact result files, plots, reviews, manifests, and the CPU runner are kept for GitHub.
