# io_representative_20260515_bpftrace_rerun Results

Generated: 2026-05-16T14:57:50Z

## Representative Search

- rows: 93
- fashion_mnist784: 24 rows
- gist960: 24 rows
- glove100: 24 rows
- yfcc10m: 21 rows
- block latency status counts: {'ok_bpftrace': 65, 'unavailable_no_events': 28}

## fio Baseline

- rows: 96

## Comparison Table

- rows: 93
- disk metrics status counts: {'ok': 93}
- bottleneck conclusion counts: {'io_latency_or_queue_underfilled': 14, 'cpu_or_algorithm_bound': 62, 'inconclusive_mixed': 17}

## Key Artifacts

- `representative_results.csv`
- `representative_results.jsonl`
- `comparison_table.csv`
- `comparison_table.jsonl`
- `fio_baseline.csv`
- `fio_baseline.jsonl`
- `representative_qps_latency.png`
- `representative_cpu.png`
- `representative_disk.png`
- `representative_vs_fio.png`

## Notes

Per-run stdout/stderr, iostat, and block trace logs are generated details and are intentionally ignored.
