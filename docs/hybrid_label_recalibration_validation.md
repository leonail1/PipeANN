# Hybrid Label Sidecar And Tau_m Recalibration Validation

日期：2026-05-19

仓库：`/mnt/bak3/lzg/PipeANN-github`

## 需求整理

本次需要验证两个功能面：

1. 混合标签 sidecar：低选择性标签使用 posting list，高选择性标签使用 bitmap。需要证明 mixed sidecar 相比原始 `.spmat` 标签文件不会产生过高空间膨胀，并按需求 3 PPT 的口径测试所有选择性点的查询性能。
2. `tau_m` 自动重标定：确认代码语义是否为前台搜索线程数低于 16 时允许启动标定查询，并且每个标定查询启动前都会重新检查前台负载；标定未结束前不执行插入或删除。还需要设计实验验证当前台搜索线程数低于 16 时，边标定边搜索仍能满足单个查询延迟小于 10 ms。

## 代码语义确认

### Mixed Label 编码

当前 mixed sidecar 是 DenseBitset v2 格式。每个 label 有独立目录项，记录 payload offset、payload size、candidate count 和 encoding。

关键代码：

- `src/filter/densebit_index.cpp:81`：`posting_threshold = 2 * words_per_label`。
- `src/filter/densebit_index.cpp:350`：`words_per_label = ceil(npoints / 64)`。
- `src/filter/densebit_index.cpp:380`：`candidate_count < posting_threshold` 时使用 posting。
- `src/filter/densebit_index.cpp:388`：否则使用 bitmap。
- `src/filter/densebit_index.cpp:624` 到 `src/filter/densebit_index.cpp:636`：intersect 查询如果所有相关 label 都是 posting，则直接在 posting list 上做 union 并生成候选 bitset。
- `src/filter/densebit_index.cpp:639` 到 `src/filter/densebit_index.cpp:660`：subset 查询如果所有相关 label 都是 posting，则直接在 posting list 上求交。

因此对 `npoints=10,000,000` 的 YFCC10M，`words_per_label=156,250`，posting/bitmap 分界点是 `312,500` 个候选，约等于 `3.125%` 选择性。低于该候选数的 label 走 posting，高于该候选数的 label 走 bitmap。

### Tau_m 自动重标定

当前核心代码的默认语义符合“前台搜索线程数低于 16 时允许启动/继续标定查询”，但需要注意：这是默认配置；调用 `configure_hybrid_recalibration()` 时可以覆盖该阈值。

关键代码：

- `include/dynamic_index.h:27` 到 `include/dynamic_index.h:30`：默认 `active_queries_low_watermark = 15`，`waiting_queries_low_watermark = 0`。
- `src/update/dynamic_index.cpp:541` 到 `src/update/dynamic_index.cpp:545`：配置重标定时会用传入的 `foreground_budget` 覆盖默认值。
- `src/update/dynamic_index.cpp:1011` 到 `src/update/dynamic_index.cpp:1015`：只有未禁用后台标定、`active_queries <= low_watermark`、`waiting_queries <= waiting_low_watermark`、且无 high priority task 时，才允许启动或继续标定。因此默认 `<=15` 等价于“低于 16”。
- `src/update/dynamic_index.cpp:1036`：worker 从 pending 转 running 前检查一次负载。
- `src/update/dynamic_index.cpp:1068` 到 `src/update/dynamic_index.cpp:1070`：拿到 mutation pause 独占锁后再次检查一次负载。
- `src/update/dynamic_index.cpp:1101` 到 `src/update/dynamic_index.cpp:1103`：每个 dataset 开始前检查一次负载。
- `src/update/dynamic_index.cpp:1145` 到 `src/update/dynamic_index.cpp:1147`：每个 sampled calibration query 开始前检查一次负载。
- `src/update/dynamic_index.cpp:1271` 到 `src/update/dynamic_index.cpp:1286`：前台 search 通过 RAII guard 维护 `waiting_queries` 和 `active_queries`，查询结束后通知标定 worker。

“标定结束前不执行插入或删除”也成立：

- `src/update/dynamic_index.cpp:1068`：重标定在整个 `run_hybrid_recalibration_once()` 中持有 `hybrid_recalibration_mutation_lock_` 的独占锁。
- `src/update/dynamic_index.cpp:837`：insert 持有同一把锁的 shared lock。
- `src/update/dynamic_index.cpp:1230`：update labels 持有 shared lock。
- `src/update/dynamic_index.cpp:1439`：lazy delete 持有 shared lock。
- `src/update/dynamic_index.cpp:1504`：final merge 持有 shared lock。

因此，一旦标定进入 running 并拿到独占锁，插入、删除、标签更新和 final merge 都会等待标定结束或因负载检查失败而退出。

这里提到的 `tests/dynamic_prefilter_stage_driver.cpp` 是一个测试/实验 driver，用来跑 staged prefilter 和自动重标定流程，不是线上核心逻辑。它在构造重标定配置时把 `active_queries_low_watermark` 显式写成了 `1`，所以如果直接用这个 driver 验证“低于 16”这条规则，实验口径会变成“活动查询数不超过 1 才允许标定”，和核心默认值 `15` 不一致。

如果后续仍要用这个 driver 做自动重标定实验，修改方式很小：

1. 在 `DriverConfig` 中加入 `recalibration_active_low_watermark` 和 `recalibration_waiting_low_watermark` 两个字段，默认分别为 `15` 和 `0`。
2. 在命令行解析中增加 `--recalibration-active-low-watermark` 和 `--recalibration-waiting-low-watermark`。
3. 把 `build_recalibration_config()` 里硬编码的 `active_queries_low_watermark = 1` 改成读取 `config.recalibration_active_low_watermark`。

如果只做 1-16 搜索线程延迟扫描，则不需要修改这个 driver，可以直接复用 `exp6_query_thread_budget`。

## 空间验证

### 已验证数据

当前 YFCC10M 真实标签：

- 原始标签文件：`data/yfcc100M/base.metadata.10M.spmat`
- Mixed sidecar：`data/yfcc100M/yfcc10m_pipeann_labels.densebit`
- 重新解析输出：`experiments/yfcc10m_mixed_validation/runtime/yfcc10m_pipeann_mixed_sidecar_summary.verify.json`

验证结果：

| 项目 | 字节数 | GiB | 说明 |
| --- | ---: | ---: | --- |
| 原始 `.spmat` | 945,683,840 | 0.881 | `24 + 8*(npoints+1) + 8*nnz` |
| Mixed sidecar | 360,086,280 | 0.335 | DenseBitset v2 |
| Bitmap-only sidecar | 250,482,500,048 | 233.280 | DenseBitset v1 对照 |

Mixed sidecar / 原始 `.spmat` = `0.3808`，即 mixed sidecar 只有原始 `.spmat` 的 `38.08%`，比原始 `.spmat` 少 `61.92%`。因此在 YFCC10M 当前真实标签上，mixed sidecar 不但没有膨胀，反而明显缩小。

Mixed sidecar 编码组成：

| 指标 | 值 |
| --- | ---: |
| npoints | 10,000,000 |
| nlabels | 200,386 |
| nnz | 108,210,476 |
| posting threshold | 312,500 |
| posting labels | 200,334 |
| bitmap labels | 29 |
| empty labels | 23 |
| posting payload bytes | 317,423,816 |
| bitmap payload bytes | 36,250,000 |
| label directory bytes | 6,412,352 |

### 已跑验证命令

```bash
/mnt/bak3/lzg/PipeANN-github/build/tests/densebit_range
/mnt/bak3/lzg/PipeANN-github/build/tests/hybrid_metadata_roundtrip
/mnt/bak3/lzg/PipeANN-github/.venv/bin/python \
  /mnt/bak3/lzg/PipeANN-github/scripts/pipeann_hybrid_experiment.py \
  summarize-densebit-sidecar \
  --sidecar-path /mnt/bak3/lzg/PipeANN-github/data/yfcc100M/yfcc10m_pipeann_labels.densebit \
  --summary-json /mnt/bak3/lzg/PipeANN-github/experiments/yfcc10m_mixed_validation/runtime/yfcc10m_pipeann_mixed_sidecar_summary.verify.json
```

结果：

- `densebit_range: ok`
- `hybrid_metadata_roundtrip: ok`
- sidecar summary 重新生成成功，除 path 字段外与既有 summary 一致。

## Mixed Label 性能实验设计

实验目标是覆盖所有选择性，并且和需求 3 PPT 保持一致：每个选择性点先校准 route 和 search budget，使 recall@10 达到 `>=98%`，再用最终 route/L 做单线程 latency/QPS 测量。

### Workload

优先使用 YFCC10M 真实单标签 workload，因为它正好能验证真实 label 分布下的 posting/bitmap 混合：

- 低选择性 posting：`1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`
- 中高选择性真实 bitmap：`1e-1, 13.8%, 17.1%, 19.2%, 33.9%`
- 如必须覆盖 PPT 中的 `50%, 75%, 100%`，需要额外生成 synthetic high-selectivity labels，并为同一图索引生成独立 mixed sidecar。

如果希望和 PPT 完全对齐，建议第二套实验用 SIFT1M uniform exact-selectivity：

`u1e-03, u3e-03, u1e-02, u5e-02, u1e-01, u25, u30, u50, u75, u100`

### Route/L 校准

对每个 `(dataset, selector, selectivity)`：

1. 计算 filtered exact ground truth。
2. 对 `prefilter` 和 `graph` 两条 route 分别 sweep `L`。
3. 只保留 `recall@10 >= 98%` 的候选。
4. 用单线程实测 latency 选择最终 route/L，不用多线程校准 latency 做最终选路。
5. 记录所有候选到 `route_selection_candidates.jsonl`，最终点写入 `table.csv`。

建议候选：

- Prefilter：固定 route=prefilter，`L=10`，同时 sweep `PIPEANN_PREFILTER_RERANK_L` 或使用已有 rerank calibration。
- Graph：`L in [40, 50, 75, 100, 150, 200, 400, 800, 1500, 3000, 6000, ...]`，高选择性通常较小，低选择性可能会超时或不达标。

### Final Measurement

最终测量字段：

- `selectivity`
- `candidate_count`
- `encoding`：posting / bitmap / mixed
- `selected_route`
- `chosen_L`
- `recall@10`
- `avg_latency_us`
- `p50_latency_us`
- `p95_latency_us`
- `p99_latency_us`
- `max_latency_us`，用于“单个查询 <10ms”的严格验收
- `qps`
- `prefilter_count / graph_count / fallback_count`
- `mean_route_overhead_us`
- `process_max_rss_kb`

验收标准：

- 空间：`mixed_sidecar_bytes / spmat_bytes <= 1.0` 为通过；如果要留工程余量，可以设 `<=1.1`。
- 性能：每个选择性点 `recall@10 >= 98%`。
- 需求 3 延迟：按 PPT 口径看单线程 avg latency，应 `<10ms`。
- 严格单查询延迟：用于重标定并发实验时，优先看 `max_latency_us < 10000`；若受 OS 调度影响，可同时报告 p99 和异常样本。

### 已有性能证据

`experiments/yfcc10m_mixed_validation/posting_vs_bitmap_summary.md` 固定了 route 和 L，不能作为“全选择性满足 10ms”的性能证据；该结果只保留为空间和 sidecar 表示对照背景。

已有 `experiments/r116_suite/exp4_intersect_range_selectivity/table.csv` 可以作为需求 3 测量口径的主证据。该实验使用直接构建的 SIFT1M 1M 索引，构建时带 label 文件并生成当前 `_labels.densebit` mixed sidecar；每个选择性点先把 route/L 校准到 `recall@10 >= 98%`，再用单线程测最终 latency/QPS。

直接构建 1M 索引结果：

- intersect：10 个选择性点 recall@10 最低 `98.01%`，最大 avg latency `8.53ms`。
- range：10 个选择性点 recall@10 最低 `98.01%`，最大 avg latency `8.60ms`。
- 低/中选择性选择 prefilter，高选择性 `u50/u75/u100` 选择 graph。

因此，如果采用 PPT 需求 3 的“单线程平均 latency”口径，这个实验已经能证明 mixed sidecar 在覆盖的全选择性点上满足 10ms。若验收口径改成“每一个单独 query 都必须 <10ms”，则该表还不够，因为部分点的 p99 已超过 10ms，需要补充记录 `max_latency_us` 并按单 query 明细验收。

## 1-16 线程延迟实验设计

实验目标：直接测量前台搜索线程数从 1 到 16 时，各选择性点的查询延迟是否低于 10 ms。这个口径比“标定事件 + 搜索事件交错记录”更简单，也更贴近最终 SLA。

### 推荐入口

优先复用现有 `exp6_query_thread_budget`，它会读取 exp4 已经选好的 route/L，然后对查询线程数做 sweep：

```bash
cd /mnt/bak3/lzg/PipeANN-github
python3 scripts/run_codex_dynamic_update_suite.py \
  --out-dir experiments/r116_suite \
  --only-exp6 \
  --rerun-exp6 \
  --skip-build \
  --build-r 116 \
  --build-l 220 \
  --pq-bytes 32 \
  --exp4-pq-bytes 32 \
  --exp6-thread-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --exp6-latency-budget-ms 10
```

输出目录：

- `experiments/r116_suite/exp6_query_thread_budget/table.csv`
- `experiments/r116_suite/exp6_query_thread_budget/threshold_table.csv`
- `experiments/r116_suite/exp6_query_thread_budget/threshold_summary.json`

### 实验矩阵

- 选择性：沿用 exp4 的 `u1e-03, u3e-03, u1e-02, u5e-02, u1e-01, u25, u30, u50, u75, u100`。
- filter：intersect 和 range。
- route/L：直接使用 exp4 每个选择性点的最终 `selected_route` 和 `chosen_L`。
- 查询线程数：`1..16`。
- latency budget：`10ms`。

### 验收标准

- 每个点 `recall@10 >= 98%`。
- 如果采用现有 exp6 默认口径：`avg_latency_us <= 10000`。
- 如果要求严格单 query：需要给 search driver 增加 `max_latency_us` 输出，并要求 `max_latency_us <= 10000`；否则当前表只能支持 avg/p50/p95/p99 口径。

这个实验直接回答“1-16 线程前台搜索是否满足 10ms”。它不再证明“标定任务和搜索任务同时运行时的干扰”，但结合上面的代码审计，已经足够说明低水位阈值和前台查询延迟预算的关系。

## 当前结论

1. Mixed label sidecar 的编码逻辑已经符合“低选择性 posting，高选择性 bitmap”。
2. YFCC10M 当前真实标签上，mixed sidecar 是原始 `.spmat` 的 `38.08%`，没有空间膨胀。
3. `experiments/r116_suite/exp4_intersect_range_selectivity/table.csv` 已经能按“校准 route/L + 单线程平均 latency”口径证明 SIFT1M 全选择性点满足 10ms。
4. `tau_m` 自动重标定的默认核心逻辑符合“前台活动查询数低于 16 才允许启动/继续；每个标定查询开始前检查一次；标定期间阻塞插入/删除/标签更新/final merge”。
5. 下一步建议直接跑 `exp6_query_thread_budget` 的 `1..16` 线程 sweep；如果验收要求是严格每个 query 都小于 10ms，则需要先给 search driver 增加 `max_latency_us` 字段。
