# Experiments

当前动态更新实验按图构建参数拆成三套独立 suite：

1. `r75_suite`
2. `r96_suite`
3. `r116_suite`

每套 suite 都包含同样的 5 个实验和 baseline：

1. `exp1_insert_vs_build_threads`
2. `exp2_stage_recall_build_vs_insert`
3. `exp3_search_during_insert`
4. `exp4_intersect_range_selectivity`
5. `exp5_index_bloat_by_size`
6. `exp_baseline`

运行方式：

```bash
./experiments/r75_suite/start.sh
./experiments/r96_suite/start.sh
./experiments/r116_suite/start.sh
```

参数配置：

| suite | R | build_L | PQ bytes | 说明 |
| --- | ---: | ---: | ---: | --- |
| `r75_suite` | 75 | 150 | 32 | 原始对照配置 |
| `r96_suite` | 96 | 150 | 32 | 中等 R 配置 |
| `r116_suite` | 116 | 220 | 32 | 当前满足 `total/raw <= 2.1x` 的高 R 配置 |

`exp3` 使用 `data/bigann/sift_base_2m_float.bin` 和
`data/bigann/sift_query_10000_float.bin` 做 1M 到 2M 插入期间前台查询实验。
`exp4` 使用直接构建的 SIFT1M 1M 索引，对 intersect/range 两类过滤查询分别测
latency、QPS 和单查询进程 RSS。其余实验使用 SIFT1M。

每个 suite 的 `start.sh` 都会完成实验运行、绘图和结果表生成。实验目录只应保留
`json/jsonl/csv/png/md/sh` 等小文件；索引、truthset、临时 workload 等大文件应在运行后清理。
