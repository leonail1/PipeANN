# ARIS-Style Experiment Review: r116 CPU Thread Sweep

日期：2026-05-19

## Scope

- ARIS source: `/Users/zhengganglin/Downloads/Auto-claude-code-research-in-sleep` at `60a8c70`
- Skills used as protocol: `experiment-queue`, `experiment-integrity`
- Executor/reviewer note: this is an ARIS-style self review in Codex Desktop, not an independent cross-model reviewer run.
- Evaluation type: `real_gt`
- Resource mode: CPU, `max_parallel=1`
- Rows: `320` / `320`

## Experiment Settings Review

- PASS: The experiment uses real filtered ground truth from `compute_groundtruth`, not synthetic or self-normalized labels.
- PASS: Route and `L` are inherited from `exp4_intersect_range_selectivity` except documented overrides.
- PASS: `l_overrides.json` documents post-review L overrides. In this run, `u75` was recalibrated from graph `L=50` to graph `L=75` after the first sweep found recall@10 below 98%.
- PASS: The sweep covers both `intersect` and `range`, all 10 selectivity buckets, and foreground query threads 1-16.
- PASS: `search_disk_index_hybrid` emits `avg`, `p90`, `p95`, and `p99` latency fields.
- PASS: The runner writes `manifest.json`, `queue_state.json`, per-job logs, per-job JSON outputs, `table.csv`, and summary plots.
- WATCH: The source index prefix is `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite/exp6_query_thread_budget/tmp/direct_1m`. It is a completed r116/PQ32 build from the interrupted exp6 run and is referenced explicitly in the manifest.
- WATCH: The 10ms acceptance in the user's latest instruction is average latency. Percentiles are plotted and reported, but p99 is expected to be stricter and may exceed 10ms.
- WATCH: Original ARIS GPU queue manager is not used because node6 has no `nvidia-smi` and this is a CPU/SSD benchmark.

## Result Review

- Average latency budget: `PASS`; failed rows above 10ms = `0`.
- p90 latency budget: `WARN`; failed rows above 10ms = `19`.
- p95 latency budget: `WARN`; failed rows above 10ms = `23`.
- p99 latency budget: `WARN`; failed rows above 10ms = `86`.
- Minimum recall@10: `98.8400`.
- Recall@10 rows below 98%: `0`.
- Max avg latency: `9.064 ms`.
- Max p90 latency: `10.584 ms`.
- Max p95 latency: `10.786 ms`.
- Max p99 latency: `38.309 ms`.

## Artifacts

- Manifest: `manifest.json`
- Queue state: `queue_state.json`
- Full table: `table.csv`
- Thread summary: `thread_summary.csv`
- L overrides: `l_overrides.json`
- U75 recalibration log: `recalibration_u75.jsonl`
- Worst-case plot: `latency_percentiles_worstcase_highres.png`
- Selector plot: `latency_percentiles_by_selector_highres.png`
