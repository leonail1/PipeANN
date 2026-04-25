# PipeANN 性能 Benchmark 条件

作者：GitHub Copilot  
更新日期：2026年4月25日

本文档只记录当前主树中正式使用的 PipeANN benchmark 条件，避免后续阅读结果时混淆查询规模、线程数和内存口径。

## 统一约定

当前正式 benchmark 统一遵循以下条件：

1. 只运行 PipeANN 主树，不包含其他系统；
1. 正式搜索 binary 是 `build/tests/search_disk_index_hybrid`；
1. 正式 auto-route 搜索默认参数是 `threads=52`、`beamwidth=4`、`k=10`、`similarity=l2`、`nbr_type=pq`、`mem_L=0`、`L=100`；
1. PQ 压缩向量默认使用 mmap 模式，即 `PIPEANN_PQ_MMAP=1`；
1. 所有正式实验产物统一写到 `experiments/`，不写入 `build/`。

## 延迟与 QPS

### SIFT1M uniform exact-selectivity

当前正式 SIFT1M 输入约定如下：

1. base/query 数据来自 `data/sift1m/sift_base.bin` 和 `data/sift1m/sift_query.bin`；
1. bucket 定义来自 `data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json`；
1. 当前共 11 个选择性点：`1e-5`、`3e-5`、`1e-4`、`3e-4`、`1e-3`、`3e-3`、`1e-2`、`1e-1`、`0.5`、`0.75`、`1.0`；
1. 每个 bucket 都复用完整 `sift_query.bin`，因此正式 latency/QPS 跑的是每桶 10000 条查询；
1. 2026年4月25日起，prefilter 路径统一使用按选择性校准后的 rerank 值，目标是 `recall@10 >= 98%`；
1. 当前正式 calibration 结果如下：`u1e-05=10`、`u3e-05=11`、`u1e-04=12`、`u3e-04=14`、`u1e-03=16`、`u3e-03=18`、`u1e-02=22`、`u1e-01=28`；
1. `u50`、`u75`、`u100` 不注入 rerank override，因为 auto route 在这些 bucket 上走 graph-only；
1. 当前正式结果输出在 `experiments/sift1m_uniform_rerank_calibrated/`。

当前正式重跑命令如下：

```bash
/mnt/data/lzg/PipeANN/.venv/bin/python scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json \
  --index-prefix data/sift1m/sift1m_pipeann_uniform \
  --index-type float \
  --selector-type intersect \
  --manifest experiments/sift1m_uniform_rerank_calibrated/manifest.json

/mnt/data/lzg/PipeANN/.venv/bin/python scripts/pipeann_hybrid_experiment.py run \
  --manifest experiments/sift1m_uniform_rerank_calibrated/manifest.json \
  --out-dir experiments/sift1m_uniform_rerank_calibrated/run \
  --dataset-name sift1m \
  --threads 52 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes auto \
  --l-values 100 \
  --prefilter-rerank-json experiments/sift1m_uniform_rerank_calibrated/calibration/prefilter_rerank_calibration.json

/mnt/data/lzg/PipeANN/.venv/bin/python scripts/pipeann_hybrid_experiment.py plot \
  --results-jsonl experiments/sift1m_uniform_rerank_calibrated/run/results.jsonl \
  --output experiments/sift1m_uniform_rerank_calibrated/sift1m_uniform_calibrated_auto_l100.png \
  --plot-l 100 \
  --title 'PipeANN sift1m uniform selectivity (calibrated rerank, auto, L=100)'
```

### YFCC-10M single-label

当前正式 YFCC-10M 输入约定如下：

1. 真实单标签 workload 位于 `data/yfcc100M/random_single_label_workloads/real_selected_labels/`；
1. 人工 50% / 75% / 100% workload 位于 `data/yfcc100M/random_single_label_workloads/synthetic_high_selectivity/`；
1. 当前正式 real/synth run 都复用完整公开 query bin，因此正式 latency/QPS 统计是每桶 100000 条查询；
1. 线程数与其他主线 benchmark 一致，统一使用 `threads=52`；
1. 当前正式结果分别位于 `experiments/yfcc10m_real_run/` 与 `experiments/yfcc10m_synth_run/`。

## Rerank 校准

当前 prefilter rerank 校准条件如下：

1. 校准入口是 `scripts/pipeann_hybrid_experiment.py calibrate-rerank`；
1. 校准只对当前 auto route 会走 prefilter 的 bucket 进行，默认上限是 `selectivity <= 0.1`；
1. 每个可校准 bucket 默认抽取 200 条查询；
1. exact reference 不是复用外部 groundtruth，而是直接从 base 向量与标签候选集计算 filtered exact top-10；
1. 目标是找到满足 `recall@10 >= 98%` 的最小 rerank 值；
1. 校准结果输出到 `prefilter_rerank_calibration.json`，并由 `run --prefilter-rerank-json ...` 消费。

当前正式 SIFT1M 校准命令如下：

```bash
/mnt/data/lzg/PipeANN/.venv/bin/python scripts/pipeann_hybrid_experiment.py calibrate-rerank \
  --summary-json data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json \
  --index-prefix data/sift1m/sift1m_pipeann_uniform \
  --out-dir experiments/sift1m_uniform_rerank_calibrated/calibration \
  --threads 52 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --search-l 100 \
  --target-recall 98
```

## 峰值内存与单查询内存分解

当前内存口径统一如下：

1. `scripts/pipeann_hybrid_experiment.py run` 中的 `peak_memory_kb` 由 `/usr/bin/time -v` 的 `Maximum resident set size` 提供；
1. 这个峰值内存测量固定使用每个 bucket 的 `probe_query.bin` 与 `probe_query.spmat`，即单查询口径；
1. `scripts/pipeann_memory_breakdown.py` 的内置 `sift1m` 预设现在默认指向 `data/sift1m/uniform_exact_selectivity/u1e-03/probe_query.bin` 与对应 `probe_query.spmat`；
1. `scripts/pipeann_memory_breakdown.py` 的内置 `yfcc10m` 预设现在默认指向 `data/yfcc100M/random_single_label_workloads/real_selected_labels/real_t1e-03_l8636/probe_query.bin` 与对应 `probe_query.spmat`；
1. 若 resident 采样需要更稳定的 post-load snapshot，可以在实验目录中额外构造 repeated single-query workload，但正式 preset 默认仍是单查询输入。

## 结果落盘位置

当前与 benchmark 条件直接相关的正式产物如下：

1. SIFT1M calibrated rerank 配置：`experiments/sift1m_uniform_rerank_calibrated/calibration/prefilter_rerank_calibration.json`；
1. SIFT1M calibrated latency/QPS 结果：`experiments/sift1m_uniform_rerank_calibrated/run/results.jsonl`；
1. SIFT1M calibrated 图：`experiments/sift1m_uniform_rerank_calibrated/sift1m_uniform_calibrated_auto_l100.png`；
1. YFCC-10M real run 结果：`experiments/yfcc10m_real_run/results.jsonl`；
1. YFCC-10M synth run 结果：`experiments/yfcc10m_synth_run/results.jsonl`；
1. 正式单查询内存分解输出根目录：`experiments/memory_breakdown/`。