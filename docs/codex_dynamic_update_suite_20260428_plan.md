# Codex 动态索引实验计划

日期：2026-04-28

本文档记录当前已经实现并正在执行的动态索引实验方案。实验入口和结果目录位于：

`experiments/`

## 运行入口

当前不再使用 suite 根目录下的统一 `start.sh`。每个实验目录都有自己的 `start.sh`，在仓库根目录运行对应入口即可：

```bash
./experiments/exp1_insert_vs_build_threads/start.sh
./experiments/exp2_stage_recall_build_vs_insert/start.sh
./experiments/exp3_search_during_insert/start.sh
./experiments/exp4_delete_reinsert_selectivity/start.sh
./experiments/exp5_index_bloat_by_size/start.sh
./experiments/exp_baseline/start.sh
```

每个 `start.sh` 都会优先使用仓库里的 `.venv/bin/python`，因为绘图依赖 `matplotlib`。脚本会调用：

- `scripts/run_codex_dynamic_update_suite.py`
- `scripts/plot_codex_dynamic_update_suite.py`
- `build/tests/dynamic_update_suite_driver`
- `build/tests/build_disk_index`
- `build/tests/utils/compute_groundtruth`
- `build/tests/calibrate_hybrid_threshold`（仍作为构建目标保留；当前 `exp4` 不再依赖它做最终路由判定）

如果只想运行某个实验，直接进入对应目录或调用对应路径的 `start.sh`。例如：

```bash
./experiments/exp2_stage_recall_build_vs_insert/start.sh --skip-build
./experiments/exp4_delete_reinsert_selectivity/start.sh --skip-build --exp4-pq-bytes 32
./experiments/exp_baseline/start.sh --skip-build --rerun-baseline --baseline-query-count 1000 --baseline-pq-bytes 32
```

实验完成后会自动调用 `scripts/plot_codex_dynamic_update_suite.py` 绘图，并清理临时索引、GT、临时数据和标签等大文件。

## 通用配置

- 数据集：SIFT1M。
- Base 数据：`data/sift1m/sift_base.bin`。
- Query 数据：`data/sift1m/sift_query.bin`。
- 距离度量：L2。
- 构建参数：
  - `R=64`
  - `L=96`
  - `PQ_bytes=16`
  - `memory_gb=64`
  - `nbr_type=pq`
- `exp4_delete_reinsert_selectivity` 单独使用 `--exp4-pq-bytes 32`，用于对齐 SIFT1M PipeANN/ref 图中的 high-selectivity graph-search 口径。主动态实验默认 `--pq-bytes 16` 仍保留。
- 动态索引 seed：10k 点。
- 查询集：前 10,000 个 SIFT query。
- 每个实验点运行 1 次。
- truthset 计算：绑定 NUMA node 1，并使用该 NUMA 节点的全部逻辑 CPU。当前机器 NUMA1 为 52 个逻辑 CPU，脚本会设置 `OMP_NUM_THREADS=52`，并通过 `numactl --cpunodebind=1 --membind=1` 启动 `compute_groundtruth`。
- 运行中会生成临时子数据集、标签、truthset 和索引文件。
- 实验结束后只保留小结果文件：`json/jsonl/csv/png/md/sh`。

选择性 bucket 由脚本确定性生成：

`u1e-03,u3e-03,u1e-02,u5e-02,u1e-01,u25,u30,u50,u75,u100`

对应选择性分别为：

`0.1%, 0.3%, 1%, 5%, 10%, 25%, 30%, 50%, 75%, 100%`

## 当前执行调整

原计划中 `exp1_insert_vs_build_threads` 的线程列表是：

`32,16,8,4,2,1`

实际执行中发现低线程动态插入耗时过长，因此已按运行时决策调整为只保留：

`32,16,8,4`

`2` 和 `1` 线程点跳过。后续实验已改为各自独立入口，不再依赖 `--resume-after-exp*` 串行续跑。

`exp4_delete_reinsert_selectivity` 的选择性查询已按 2026-04-29 到 2026-04-30 的排查结果调整：

- 不再通过手工修改 `_hybrid.meta` 的 `tau_m` 来改变路由。`tau_m` 只表示候选规模阈值，不能代表动态更新后的图质量。
- 不再在每个状态前调用 `calibrate_hybrid_threshold` 作为最终依据；重插后曾出现 `_hybrid.meta` 缺失或不稳定，继续依赖 auto-route 会把低选择性错误送入 graph-search。
- 每个状态、每个选择性 bucket 都先用 32 查询线程实测 `prefilter` 路径，并在候选 `L` 列表中找满足 `recall@10 >= 98` 的点。
- 如果 `prefilter` 已达标、选择性低于 50%、且候选量不超过 live 点数的 50%，直接选择 `prefilter`，不再浪费时间扫 graph。这样 5%、10%、25%、30% 等低/中选择性不会误走 graph-search。
- `u50/u75/u100` 必须额外测 graph 候选。32 线程阶段只用于校准各 route 的达标 `L`；最终 route 选择必须用单线程实测 latency 决定，避免 32 线程下 prefilter 吞吐更高而误导单线程图。
- 图中的 latency/QPS 是单线程比较所有达标候选后选出的最终 route + `L`；CSV/JSONL 中记录 `selected_route`、`chosen_L`、`calibration_recall@10` 和 `calibration_avg_latency_us`。
- 如果只想重新生成 `exp4`，使用 `exp4_delete_reinsert_selectivity/start.sh --rerun-exp4 --exp4-pq-bytes 32`。

## 实验 1：插入 vs 直接构建耗时

目录：

`exp1_insert_vs_build_threads/`

保留线程数：

`32,16,8,4`

每个线程数执行：

1. 构建 10k seed 索引。
2. 从 10k 动态插入到 1M。
3. 执行 `final_merge`。
4. 记录：
   - `seed_build_s`
   - `insert_remaining_s`
   - `insert_total_s = seed_build_s + insert_remaining_s`
   - `build_1m_s`
5. 用同样线程数直接构建 1M 索引作为对比。

输出：

- `results.jsonl`
- `table.csv`
- `insert_vs_build_threads.png`

## 实验 2：不同阶段的构建索引 vs 插入索引召回

目录：

`exp2_stage_recall_build_vs_insert/`

阶段规模：

`250k,500k,750k,1M`

主图展示三条曲线：

- 直接构建：分别构建 250k、500k、750k、1M 索引。
- 250k seed 插入：先直接构建 250k 初始索引，再插入到 500k、750k、1M。
- 500k seed 插入：先直接构建 500k 初始索引，再插入到 750k、1M。

10k seed 插入结果仍保留在历史 `results.jsonl/table.csv` 中用于对照，但不再作为 `stage_recall_build_vs_insert.png` 的主曲线。

每个阶段单独生成 truthset，避免用 1M ground truth 去评估 250k/500k/750k 导致 recall 失真。

搜索 `L` 选择规则：

1. 在 direct 1M 和 inserted 1M 上 sweep `[40,60,80,100]`。
2. 选择第一个满足两条路径 recall 都在 `(80, 99.8)` 范围内的 `L`。
3. 如果没有满足条件的 `L`，回退为 `L=60`。
4. 当前已选出的主图对比 `L=40`；250k/500k seed 插入曲线固定使用该 `L`，保证和原始 exp2 可比。

输出：

- `results.jsonl`
- `table.csv`
- `stage_recall_build_vs_insert.png`
- `seed_sweep_results.jsonl`
- `seed_sweep_table.csv`
- `seed_sweep_recall.png`

## 实验 3：插入期间前台查询性能

目录：

`exp3_search_during_insert/`

插入线程数：

`4,2,1`

查询线程数：

`32,16,8,4,2,1`

每组配置执行：

1. 构建新的 500k 起点索引。
2. 后台从 500k 向 1M 插入。
3. 前台运行 1,000 个查询。
4. 前台查询完成后停止本轮后台插入。
5. 每个查询线程数都重新初始化 500k 起点，避免插入进度互相污染。

记录：

- 平均延迟
- p50/p95/p99 延迟
- QPS
- recall@10
- 查询结束时已插入点数
- live point count

输出：

- `results.jsonl`
- `table.csv`
- `search_during_insert_ins4.png`
- `search_during_insert_ins2.png`
- `search_during_insert_ins1.png`

## 实验 4：删除到 750k 后再插回 1M 的选择性查询

目录：

`exp4_delete_reinsert_selectivity/`

状态序列：

1. `1m_initial`：直接构建 1M。
2. `750k_after_delete`：删除 tag `[750000,1000000)` 后 `final_merge`。
3. `1m_after_reinsert`：重新插入 tag `[750000,1000000)` 后 `final_merge`。

每个状态、每个选择性 bucket：

1. 使用 exp4 专用的 1000-query bin 和对应 query label 生成 truthset，避免 GT 阶段误用 10000-query 文件。
2. 用 32 查询线程分别校准可用 route 和 `L`，目标是 `recall@10 >= 98`。
3. 先测 `prefilter`；如果 `prefilter` 达标、选择性低于 50%、且候选量不超过 live 点数的 50%，直接选择 `prefilter`。
4. 对高选择性或 `prefilter` 不达标的情况，再测 graph 候选。`u50/u75/u100` 不允许因为 `prefilter` 已达标而提前退出。
5. 候选列表：
   - `[10,20,30,40,50,75,100,125,150,200,300,400,600,800,1000,1500,2000]`
6. 对所有 32 线程校准达标的候选 route + `L` 做单线程查询，再选择单线程平均延迟最低的一组作为最终结果。
7. 记录延迟、QPS、recall、chosen `L`、selected route、RSS、live point count 和实际路由计数。所有参与单线程 route 选择的候选写入 `route_selection_candidates.jsonl`。

删除态标签语义：

- `750k_after_delete` 是从 1M 索引删除尾部 `[750000,1000000)` 得到的，因此 live 点保留的是 1M 构建时的选择性标签语义。
- 该状态的 truthset 不能使用重新按 750k 缩放的 `base_750k.spmat`，否则 `u50` 会从“原始前 500k”错误变成“当前前 375k”，导致 recall 统计失真。
- 脚本会为删除态生成 `base_750k_from_1m_semantics.spmat`，即把 1M 选择性语义裁剪到 live 750k 后再计算 truthset。

输出：

- `results.jsonl`
- `table.csv`
- `selectivity_1m_initial.png`
- `selectivity_750k_after_delete.png`
- `selectivity_1m_after_reinsert.png`

## 实验 5：不同规模的索引膨胀率

目录：

`exp5_index_bloat_by_size/`

阶段规模：

`250k,500k,750k,1M`

每个阶段：

1. 直接构建索引。
2. 统计以下文件大小：
   - `_disk.index`
   - `_disk.index.tags`
   - `_pq_compressed.bin`
   - `_pq_pivots.bin`
   - `_labels.densebit`
   - `_hybrid.meta`
3. 计算：
   - `raw_vector_bytes = npoints * 128 * 4`
   - `extra_over_raw_ratio = (total_index_bytes - raw_vector_bytes) / raw_vector_bytes`
   - `total_to_raw_ratio = total_index_bytes / raw_vector_bytes`

输出：

- `results.jsonl`
- `table.csv`
- `index_bloat_by_size.png`

## 基准实验：直接构建 1M 的 forced route 性能

目录：

`exp_baseline/`

这个实验用于回答一个单独问题：在没有删除、没有重插、没有 auto-route 干预的直接构建 1M 索引上，`prefilter` 和 `graph` 两条路径各自的真实性能是多少。

运行方式：

```bash
./experiments/exp_baseline/start.sh --rerun-baseline --baseline-query-count 1000 --baseline-pq-bytes 32
```

配置：

- 直接构建 1M 索引。
- baseline 直接构建索引使用 `PQ_bytes=32`。动态更新主实验仍保留通用配置里的 `PQ_bytes=16`，但这个 forced-route baseline 要对齐 `ref.png` 和正式 SIFT1M PipeANN 基线，因此单独使用 PQ32。
- 查询数：默认 1000，可通过 `--baseline-query-count` 调整到 10000。
- 查询线程数：只跑单线程 `1`。此前多线程 baseline 会把问题混在并发调度里，不利于和 `ref.png` 这种单线程曲线对齐。
- 选择性 bucket：`u1e-03,u3e-03,u1e-02,u5e-02,u1e-01,u25,u30,u50,u75,u100`。
- 强制 route：
  - `prefilter`
  - `graph`
- 不使用 auto-route。
- 搜索驱动：使用静态 `build/tests/search_disk_index_hybrid`，强制 `prefilter` 或 `graph`。旧版本曾用动态 driver 加载直接构建索引，`graph_count=0` 且 `fallback_count=1000`，说明并没有真正走静态 graph-only 路径，因此该口径已废弃。
- 每个 route、每个选择性先在单线程下 sweep `L`，目标是 `recall@10 >= 98`。
- 如果某条 route 在达到 98% recall 前，平均搜索延迟已经超过 100 ms，则标记为 skipped，图中不画该点。

输出：

- `results.jsonl`
- `skipped.jsonl`
- `calibration.jsonl`
- `table.csv`
- `baseline_single_thread_latency.png`
- `baseline_fixed_l100_ref_like.png`
- `baseline_prefilter.png`
- `baseline_graph.png`

## 已完成的基础验证

在全量运行前已完成：

- 编译通过：
  - `dynamic_update_suite_driver`
  - `build_disk_index`
  - `compute_groundtruth`
- Python 语法检查通过：
  - `scripts/run_codex_dynamic_update_suite.py`
  - `scripts/plot_codex_dynamic_update_suite.py`
- 在 `/tmp` 下跑过 200 点 seed 到 300 点的动态插入 sanity test，确认 driver 可以输出合法 JSONL。

## 2026-04-29 重跑结论

注意：本节记录的是 2026-04-29 的中间排查结果，其中“30 个点最终都选择 `prefilter L=10`”后来被证明仍有口径问题：当时用 32 线程校准阶段的 latency 决定最终 route，但图中展示的是单线程 latency。2026-04-30 已修复为“32 线程只校准 recall/L，单线程实测后再选最快 route”，最终结论以后文为准。

本轮先使用：

```bash
./experiments/exp4_delete_reinsert_selectivity/start.sh --rerun-exp4
```

排查中发现旧的 `calibrate_hybrid_threshold` 阶段会在 `1m_after_reinsert` 上长时间运行，而且重插后的索引加载时出现 `_hybrid.meta` 缺失提示。随后移除 `exp4` 对该阶段的依赖，并使用：

```bash
./experiments/exp4_delete_reinsert_selectivity/start.sh
```

从已有 `exp4` 进度继续完成剩余点。最终主脚本正常退出，绘图完成，清理检查中实验目录下没有残留 `.bin/.index/.spmat/.densebit/.meta/.tags/.log` 大文件。

本轮不再手工修改 `tau_m`，也不再把 auto-route 阈值作为最终依据。每个状态、每个选择性 bucket 都先测 `prefilter`，必要时再测 graph；最终图展示的是满足 `recall@10 >= 98` 的最低延迟 route + `L` 的单线程 latency 和 QPS。三张图已经重新生成，并在每个点标注 route 和 chosen `L`。

- `exp4_delete_reinsert_selectivity` 30 个点全部 `status=ok`，三个状态的 `recall@10` 都不低于 98%：
  - `1m_initial`：最低 98.97%，最大平均延迟约 13.99 ms。
  - `750k_after_delete`：最低 99.07%，最大平均延迟约 11.02 ms。
  - `1m_after_reinsert`：最低 98.93%，最大平均延迟约 13.73 ms。
- 三个状态、全部 30 个点最终都选择了 `prefilter L=10`。关键点：
  - `1m_initial/u5e-02`：recall 99.82%，平均延迟约 1.42 ms，QPS 约 703.89。
  - `1m_after_reinsert/u5e-02`：recall 99.82%，平均延迟约 1.72 ms，QPS 约 582.84。
  - `1m_after_reinsert/u50`：recall 99.14%，平均延迟约 8.27 ms，QPS 约 120.88。
  - `1m_after_reinsert/u100`：recall 98.93%，平均延迟约 13.73 ms，QPS 约 72.83。
- 这说明旧图中 `1m_initial` 接近 30 ms、`1m_after_reinsert/u5e-02` 走 graph 的现象不是直接构建 1M 或重插后选择性查询的真实代价，而是计时口径和 route 选择共同造成的偏差。修正后，5% 选择性明确走 `prefilter`，并且满足 98% 召回。
- `exp5_index_bloat_by_size` 的 `extra_over_raw_ratio` 约为 0.634，低于 1.0 阈值；`total_to_raw_ratio` 约为 1.634。

随后追加运行了直接构建 1M 的 forced route baseline：

```bash
./experiments/exp_baseline/start.sh --rerun-baseline --baseline-query-count 1000 --baseline-pq-bytes 32
```

本次 baseline 正常退出，绘图完成，清理检查中实验目录下没有残留 `.bin/.index/.spmat/.densebit/.meta/.tags/.log` 大文件。脚本也已改成所有选择性都实测 forced graph；低选择性点只有在“平均延迟超过 100 ms 仍未达到 98% recall”时才跳过，不再人工跳过。输出位于：

- `exp_baseline/results.jsonl`
- `exp_baseline/skipped.jsonl`
- `exp_baseline/table.csv`
- `exp_baseline/baseline_single_thread_latency.png`
- `exp_baseline/baseline_fixed_l100_ref_like.png`
- `exp_baseline/baseline_prefilter.png`
- `exp_baseline/baseline_graph.png`

关键结果：

- 直接构建 1M baseline 使用 PQ32 后，`ref.png` 所暗示的高选择性 graph 单线程延迟是合理的：在满足 `recall@10 >= 98` 的口径下，`u50/u75/u100` 的 graph 平均延迟分别约为 `9.02 ms / 6.89 ms / 7.04 ms`。
- `prefilter` 在所有选择性上都用 `L=10` 达到 98% recall 以上；单线程延迟从 `0.37 ms` 增长到 `19.89 ms`。
- `graph` 的最终达标点如下：
  - `u5e-02`：`L=800`，recall 99.12%，单线程平均延迟约 68.61 ms。
  - `u1e-01`：`L=400`，recall 99.02%，单线程平均延迟约 34.22 ms。
  - `u25`：`L=150`，recall 98.02%，单线程平均延迟约 13.63 ms。
  - `u30`：`L=150`，recall 98.29%，单线程平均延迟约 13.48 ms。
  - `u50`：`L=100`，recall 98.52%，单线程平均延迟约 9.02 ms。
  - `u75`：`L=75`，recall 98.67%，单线程平均延迟约 6.89 ms。
  - `u100`：`L=75`，recall 99.09%，单线程平均延迟约 7.04 ms。
- `graph` 在 `u1e-03/u3e-03/u1e-02` 三个低选择性 bucket 上按规则跳过：`L=1500` 时平均延迟已经分别约 123.94 ms、123.69 ms、129.88 ms，但 recall 仍只有 18.40%、41.40%、86.44%。

本轮纠正了前一版分析中的关键错误：之前临时 baseline 使用过 PQ16，导致 graph 为达到 98% recall 需要显著更大的 `L`，进而把高选择性延迟误判为十几到几十毫秒。正式 SIFT1M PipeANN 基线和 `ref.png` 对应的是 PQ32 量级；使用仓库已有正式索引 `data/sift1m/sift1m_pipeann_uniform` 额外验证时，`u100`、`L=100` 的 graph 单线程平均延迟约 8.69 ms，recall 约 99.60%。因此，“高选择性 graph 单线程延迟不应大于 10ms”这个判断在 PQ32 baseline 上是成立的。

新的结论是：直接构建 1M 的 graph 路径本身没有异常慢；异常来自实验口径不一致，主要是 PQ bytes 配置不一致，以及 filtered graph 在低/中选择性下确实需要更大的 `L` 才能达到 98% filtered recall。对 `u50/u75/u100` 这类高选择性，PQ32 下 graph 已经恢复到符合 `ref.png` 预期的低延迟范围。

为对齐 `ref.png` 的视觉口径，已额外生成 `baseline_fixed_l100_ref_like.png`：

- `prefilter` 使用 `L=10`。
- `graph` 固定使用 `L=100`。
- 左轴画 latency，右轴画 graph 的 `recall@10`。
- 这张图用于显示固定 `L=100` 下的延迟和 recall，不替代 `baseline_single_thread_latency.png` 中“先校准到 98% recall 再画延迟”的达标结果。

仍然存在的算法层面问题：

- `exp2_stage_recall_build_vs_insert` 的 graph-only 阶段召回显示，增量插入索引在相同 `L=40` 下低于直接构建索引；1M 点时直接构建为 88.13%，增量插入为 82.875%。这不是本轮选择性 prefilter 实验能覆盖的问题，而是动态插入生成的图结构质量与直接构建存在差距。
- `exp4` 当前结论主要说明选择性查询应按候选量走 `prefilter`，删除/重插不应让 5% 这类选择性误入 graph-search。它不能反证 graph-only 路径没有质量下降；graph 质量问题仍需用 `exp2` 或新增 graph-only 对照实验继续分析。

## 2026-04-30 exp4 修复与重跑结论

修复内容：

- `exp4` 默认改为 `--exp4-pq-bytes 32`，避免用 PQ16 解释 high-selectivity graph latency。
- `u50/u75/u100` 不再因为 `prefilter` 已达标而提前退出，必须进入 graph 候选校准。
- route 选择口径改为两阶段：
  - 32 查询线程只用于找每条 route 达到 `recall@10 >= 98` 的最小 `L`。
  - 所有达标候选再用单线程实测，最终选择单线程平均延迟最低的 route + `L`。
- 新增 `route_selection_candidates.jsonl`，记录参与单线程 route 选择的所有候选，便于复核。

重跑命令：

```bash
./experiments/exp4_delete_reinsert_selectivity/start.sh \
  --rerun-exp4 \
  --skip-build \
  --exp4-pq-bytes 32
```

本次重跑正常退出，三张图已重新生成，清理检查中实验目录下没有残留 `.bin/.index/.spmat/.densebit/.meta/.tags/.log` 大文件。`exp4_delete_reinsert_selectivity/table.csv` 共 30 个点，全部 `status=ok` 且满足 `recall@10 >= 98`。

关键结果如下：

- `1m_initial`：
  - `u50` 选择 `graph L=100`，recall 98.46%，平均延迟 8.75 ms。
  - `u75` 选择 `graph L=75`，recall 98.72%，平均延迟 6.74 ms。
  - `u100` 选择 `graph L=75`，recall 99.01%，平均延迟 6.71 ms。
- `750k_after_delete`：
  - `u50` 选择 `graph L=125`，recall 98.48%，平均延迟 10.63 ms。
  - `u75` 选择 `graph L=100`，recall 98.29%，平均延迟 8.68 ms。
  - `u100` 选择 `graph L=100`，recall 98.29%，平均延迟 8.66 ms。
- `1m_after_reinsert`：
  - `u50` 选择 `graph L=125`，recall 98.18%，平均延迟 10.73 ms。
  - `u75` 选择 `graph L=100`，recall 98.09%，平均延迟 8.68 ms。
  - `u100` 选择 `graph L=100`，recall 98.26%，平均延迟 8.72 ms。

对比候选验证：

- `1m_initial/u75`：`prefilter L=10` 为 16.25 ms、recall 99.90%；`graph L=75` 为 6.74 ms、recall 98.72%，最终应选 graph。
- `750k_after_delete/u75`：`prefilter L=10` 为 16.32 ms、recall 99.87%；`graph L=100` 为 8.68 ms、recall 98.29%，最终应选 graph。
- `1m_after_reinsert/u75`：`prefilter L=10` 为 16.74 ms、recall 99.86%；`graph L=100` 为 8.68 ms、recall 98.09%，最终应选 graph。
- `1m_after_reinsert/u100`：`prefilter L=10` 为 21.13 ms、recall 99.90%；`graph L=100` 为 8.72 ms、recall 98.26%，最终应选 graph。

新的结论：

- 之前 exp4 高选择性图中“选择性越高 graph 延迟越高/总是 prefilter”的现象不是算法真实性能，而是实验脚本选路口径错误叠加 PQ bytes 不一致造成的。
- 修复后，直接构建、删除到 750k、再重插回 1M 的三个状态，在 `u50/u75/u100` 上都会选择 graph；`u75/u100` 单线程延迟回到约 6.7-8.7 ms，和 baseline/ref 图一致。
- 删除/重插后 graph 达标所需 `L` 从直接构建的 `75/100` 上升到 `100/125`，说明动态更新后的 graph 仍有一定质量退化；但通过提高 `L` 可以满足 98% recall，单线程延迟仍基本在 10 ms 左右。

## 2026-04-30 exp2 初始索引规模 sweep

本轮补充实验用于回答：如果不从 10k seed 开始插入，而是先直接构建 250k 或 500k 初始索引，再继续增量插入，是否能减轻增量插入对 graph 质量的影响。

运行命令：

```bash
./experiments/exp2_stage_recall_build_vs_insert/start.sh \
  --rerun-exp2-seed-sweep \
  --skip-build \
  --exp2-seed-sweep-starts 250000,500000 \
  --exp2-seed-sweep-l 40
```

输出目录仍是 exp2 主目录：

`exp2_stage_recall_build_vs_insert/`

输出文件：

- `seed_sweep_results.jsonl`
- `seed_sweep_table.csv`
- `seed_sweep_recall.png`

口径：

- 搜索固定使用 `L=40`，与原始 `exp2_stage_recall_build_vs_insert` 保持可比。
- 每个阶段仍使用对应规模的 truthset。
- 查询单线程执行；这里的目标是比较 graph-only recall，而不是优化 latency。

结果：

| 路径 | 阶段 | recall@10 | 平均延迟 |
| --- | ---: | ---: | ---: |
| 250k 直接构建起点 | 250k | 91.674% | 5.123 ms |
| 250k 起点插入 | 500k | 88.923% | 4.547 ms |
| 250k 起点插入 | 750k | 87.140% | 4.504 ms |
| 250k 起点插入 | 1M | 85.991% | 4.774 ms |
| 500k 直接构建起点 | 500k | 89.917% | 4.336 ms |
| 500k 起点插入 | 750k | 88.184% | 4.857 ms |
| 500k 起点插入 | 1M | 86.832% | 4.644 ms |

与原始 exp2 对比：

| 路径 | 1M recall@10 | 相对直接构建 1M 的差距 |
| --- | ---: | ---: |
| 直接构建 1M | 88.130% | 0 |
| 10k seed 插入到 1M | 82.875% | -5.255 pct |
| 250k seed 插入到 1M | 85.991% | -2.139 pct |
| 500k seed 插入到 1M | 86.832% | -1.298 pct |

结论：

- 增大初始直接构建索引规模确实能改善动态插入后的 graph 质量。
- 250k 起点已经把 1M recall 从 82.875% 提升到 85.991%；500k 起点进一步提升到 86.832%。
- 但 500k 起点仍低于直接构建 1M 的 88.130%，说明“从较大 seed 开始插入”只能缓解 graph 质量差距，不能完全消除差距。
- 这支持当前判断：exp2 的 recall 差异不是 `tau_m` 或选择性路由问题，而是动态插入形成的 graph 结构与一次性构建的 graph 结构仍有质量差异；提高查询 `L` 可以补 recall，但会带来对应 latency 成本。

## 验收口径修正

- `10 ms` 延迟目标只用于单线程搜索结果。多线程实验中的平均延迟、p95、p99 会受到并发调度、I/O 竞争和后台插入影响，不能直接按单线程 10 ms 判失败。
- RSS 必须按单查询模式测量。脚本已调整为：`exp4` 的最终 latency/QPS 仍使用 1000 query 单线程统计，但 RSS 会额外用 `query_limit=1` 跑单查询测量，并把 `max_rss_kb` 改为单查询 RSS；普通 1000-query 搜索进程 RSS 保留在 `search_max_rss_kb`。
- 上述 RSS 字段调整已经写入脚本，但已有 `exp4` 表格是在调整前生成的，因此当前 `exp4_delete_reinsert_selectivity/table.csv` 中的 RSS 不能作为最终验收 RSS。需要重新运行 `exp4` 后再用 RSS 字段做验收。

## 最终检查

实验结束后，应检查实验目录中是否还残留大文件：

```bash
find experiments -type f \
  | grep -E '\\.(bin|index|spmat|densebit|meta|tags|log)$'
```

正常情况下，清理后该命令不应输出任何文件。
