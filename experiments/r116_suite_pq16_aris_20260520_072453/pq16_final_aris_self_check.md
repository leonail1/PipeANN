# PQ16 r116 ARIS Self-Check

- Repo: `/mnt/bak3/lzg/PipeANN-github`
- Output root: `/mnt/bak3/lzg/PipeANN-github/experiments/r116_suite_pq16_aris_20260520_072453`
- Git commit: `fc07195c82d749a452f77cbf771fadab0cd7e2c0`
- RSS formula: `adjusted_rss_kb = measured_peak_rss_kb - ceil(query_file_bytes/1024) - ceil(gt_file_bytes/1024); labels and PQ resident/mmap pages are counted`

## Claims
- WARN: PQ16 lowers PQ-resident adjusted RSS to around 30 MiB (pq_memory max adjusted RSS = 32.41 MiB)
- PASS: PQ16 mmap/drop-cache lowers adjusted RSS materially (pq_disk_no_cache max adjusted RSS = 18.71 MiB)
- PASS: PPT calibrated route/L reaches recall@10 >= 98 in exp4 (exp4 rows=20, failed_recall_rows=0, min recall=98.550003)
- FAIL: Exp6 covers full 1..16 thread sweep (exp6 rows=319 expected=320)
- PASS: 16-thread avg latency remains below 10 ms (thread16 max avg latency = 8.751 ms)

## Phase 1 PQ Residency
- Rows: 40
- pq_disk_no_cache: RSS 12.74-18.71 MiB; latency 0.163-53.539 ms; min recall 98.440
- pq_memory: RSS 27.98-32.41 MiB; latency 0.083-6.373 ms; min recall 98.450

## Exp4
- Rows: 20
- Failed recall rows: 0
- Min recall@10: 98.550003
- Max avg latency: 5.99676123 ms

## Exp6
- Rows: 319 / 320
- Threads: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- Min recall@10: 98.379997
- Max avg latency: 10.328325195 ms
- Thread-16 max avg latency: 8.751087891 ms

## Provenance
- Input hashes: `pq16_final_input_hashes.csv`
- Phase status log: `phase2_status.jsonl`
- Raw command logs: `logs/phase2/*.log`
