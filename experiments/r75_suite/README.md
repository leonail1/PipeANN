# R75 Dynamic Update Suite

本 suite 使用同一组参数完整运行动态更新实验：

- `R=75`
- `build_L=150`
- `PQ=32`
- `beamwidth=4`
- `k=10`

运行入口：

```bash
./experiments/r75_suite/start.sh
```

脚本会按顺序生成 5 个实验和 baseline：

1. `exp1_insert_vs_build_threads`
2. `exp2_stage_recall_build_vs_insert`
3. `exp3_search_during_insert`
4. `exp4_intersect_range_selectivity`
5. `exp5_index_bloat_by_size`
6. `exp_baseline`

`exp3` 使用 `data/bigann/sift_base_2m_float.bin` 和
`data/bigann/sift_query_10000_float.bin`，从 1M 初始索引向 2M 插入时测前台查询。
`exp4` 使用直接构建的 SIFT1M 1M 索引，对 intersect/range 两类过滤查询分别测
latency、QPS 和单查询进程 RSS。其余实验使用 SIFT1M。每个实验目录只保留
JSON/JSONL/CSV/PNG/README 等小结果文件。
