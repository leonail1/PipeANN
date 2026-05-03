# PipeANN 动态更新实验汇报

数据集：SIFT1M；需求 4 使用 BIGANN/SIFT 前 2M 向量  
实验目录：`experiments/`

---

## 结论总览

| 需求点 | 当前结论 | 证据 | 状态 |
|---|---|---|---|
| 动态增量索引能否从低数据量增长到 1M | 可以完成从 seed 到 1M 的插入流程 | exp1、exp2、exp4 均完成 1M 动态插入/重插 | 通过 |
| 插入索引与直接构建索引 recall 是否一致 | 不一致；seed 越大差距越小，但 1M 仍有差距 | exp2：10k seed 差 5.255 pct；250k seed 差 2.139 pct；500k seed 差 1.298 pct | 未达标 |
| 插入期间前台查询 latency/QPS 是否稳定 | route/L 校准到 recall@10 >= 98 后，单线程平均延迟 <10ms | exp3：BIGANN/SIFT 前 2M，40 个选择性点 min recall@10 = 98.12，max avg latency = 8.54ms | 通过 |
| 删除到 750k、再插回 1M 后 recall 是否保持 | 在按选择性校准 route/L 后，30 个点全部 recall@10 >= 98 | exp4 最新重跑：min recall@10 = 98.16 | 通过 |
| 删除/重插后的单线程 latency 是否 <10ms | 部分高选择性点超过 10ms | exp4：最大 avg latency 16.35ms；删除后 u50/u75、重插后 u50/u75 超过 10ms | 部分通过 |
| RSS 是否满足 30MB | 已按干净单查询进程口径修正，峰值低于 30MB | exp4：rss_single_query_kb 最大 20468 KB，约 19.99 MB | 通过 |
| 索引膨胀率是否 <= 1x raw | 满足 | exp5：extra_over_raw_ratio 约 0.634 | 通过 |
| 低选择性是否走 auto route/prefilter | 最新 exp4 低/中选择性均选择 prefilter | exp4：u1e-03 到 u30 全部 prefilter | 通过 |

---

## 实验入口和产物结构

每个实验已经拆成独立目录和独立入口：

| 实验 | 入口 | 主要结果 |
|---|---|---|
| exp1 插入 vs 直接构建 | `exp1_insert_vs_build_threads/start.sh` | `table.csv`, `insert_vs_build_threads.png` |
| exp2 分阶段 recall | `exp2_stage_recall_build_vs_insert/start.sh` | `table.csv`, `seed_sweep_table.csv`, `stage_recall_build_vs_insert.png` |
| exp3 插入期间查询 | `exp3_search_during_insert/start.sh` | `table.csv`, `search_during_insert_selectivity.png` |
| exp4 删除/重插选择性 | `exp4_delete_reinsert_selectivity/start.sh` | `table.csv`, `selectivity_*.png` |
| exp5 索引膨胀率 | `exp5_index_bloat_by_size/start.sh` | `table.csv`, `index_bloat_by_size.png` |
| exp_baseline 强制 route 基准 | `exp_baseline/start.sh` | `table.csv`, `baseline_prefilter_vs_graph.png` |

本轮只重跑了 exp4：

```bash
experiments/exp4_delete_reinsert_selectivity/start.sh --skip-build --rerun-exp4 --exp4-pq-bytes 32
```

---

## 需求 1：动态索引能否增长到 1M

结论：流程可行，但增量构建质量不等同于直接构建。

证据：

| 实验 | 路径 | 结果 |
|---|---|---|
| exp1 | 10k seed + 插入剩余向量 | 线程 32/16/8/4 均完成到 1M |
| exp2 | seed 后分阶段插入 | 250k、500k、750k、1M 阶段均可查询 |
| exp4 | 1M -> 删除到 750k -> 重插回 1M | 删除、final_merge、重插、final_merge 全流程完成 |

注意：exp1 按中途决策只跑到 4 线程，2 线程和 1 线程未继续。

---

## 需求 2：插入 vs 直接构建用时

exp1 已完成 32/16/8/4 线程点。结果显示直接构建明显快于从 10k seed 插入到 1M。

| 线程数 | seed+insert 总时间 s | 直接构建 1M s | 插入/构建倍率 |
|---:|---:|---:|---:|
| 32 | 534.64 | 56.55 | 9.45x |
| 16 | 836.31 | 72.86 | 11.48x |
| 8 | 1221.64 | 139.12 | 8.78x |
| 4 | 2219.03 | 227.50 | 9.75x |

![exp1](../experiments/exp1_insert_vs_build_threads/insert_vs_build_threads.png)

---

## 需求 3：直接构建 vs 插入索引 recall 是否一致

结论：未达标。增量插入索引 recall 低于直接构建。

固定 `L=40` 的原始 exp2：

| 点数 | 直接构建 recall@10 | 10k seed 插入 recall@10 | 差距 |
|---:|---:|---:|---:|
| 250k | 91.640 | 87.910 | -3.730 pct |
| 500k | 89.868 | 85.258 | -4.610 pct |
| 750k | 88.896 | 83.958 | -4.938 pct |
| 1M | 88.130 | 82.875 | -5.255 pct |

后续 seed sweep：

| 起始 seed | 1M recall@10 | 相对直接构建 1M 差距 |
|---:|---:|---:|
| 10k | 82.875 | -5.255 pct |
| 250k | 85.991 | -2.139 pct |
| 500k | 86.832 | -1.298 pct |

结论很朴素：更大的初始图能减少质量损失，但当前增量插入仍没有完全追平直接构建。

---

## exp2 图

![exp2 stage](../experiments/exp2_stage_recall_build_vs_insert/stage_recall_build_vs_insert.png)

![exp2 seed sweep](../experiments/exp2_stage_recall_build_vs_insert/seed_sweep_recall.png)

---

## 需求 4：插入期间前台查询稳定性

结论：按需求 5 的方式逐选择性校准 route/L 后，从 1M initial 向 2M 后台插入期间，单线程前台查询平均延迟满足 <10ms。

本版不再固定 `L=100`，而是在每个选择性上先找满足 recall@10 >= 98 的 route/L 候选，再用单线程实测选择最快路径。图中圆点表示 `L=10`/prefilter，三角表示 `L>10`/graph。

数据口径：使用 BIGANN/SIFT public 数据集，脚本通过 HTTP Range 只下载 base 前 2M 向量和 query 10k，转换为 PipeANN float `.bin`。无插入曲线是直接构建 1M initial；1/2/4 插入线程曲线均从同一个 1M initial 起点开始，后台插入 `[1M,2M)`。每个选择性查询结束后立即停止插入、删除运行期索引，下一个选择性重新从干净 1M initial 开始。

| 插入线程 | 点数 | min recall@10 | max avg latency ms | 说明 |
|---:|---:|---:|---:|---|
| 0 | 10 | 98.12 | 7.58 | 直接构建 1M initial |
| 1 | 10 | 98.12 | 7.91 | 1M initial 起点，后台插入到 2M |
| 2 | 10 | 98.12 | 8.43 | 1M initial 起点，后台插入到 2M |
| 4 | 10 | 98.12 | 8.54 | 1M initial 起点，后台插入到 2M |

---

## exp3 图

![exp3 selected route](../experiments/exp3_search_during_insert/search_during_insert_selectivity.png)

---

## 需求 5：删除/重插后 recall 是否稳定

本轮已重跑 exp4，使用 PQ32，并且按照 route/L 校准后再测单线程 latency/QPS。

校准方式：

| 步骤 | 设置 |
|---|---|
| 状态 | 1M initial、750k after delete、1M after reinsert |
| 选择性 | u1e-03 到 u100 共 10 个 |
| recall 目标 | recall@10 >= 98 |
| route 候选 | prefilter / graph |
| L 候选 | 10,20,30,40,50,75,100,125,150,200,300 |
| 最终测量 | 单线程、1000 查询 |
| RSS | 单查询模式单独测量 |

最新结果：

| 指标 | 数值 |
|---|---:|
| 点数 | 30 |
| 最低 recall@10 | 98.16 |
| RSS 模式 | single_query |
| 大文件清理 | 已清理，未发现 `.bin/.index/.spmat/.densebit/.meta/.tags/.log` 残留 |

结论：按校准后的最小可用 route/L，删除和重插后的 recall 可以保持在 98 以上。

---

## exp4 高选择性 latency

| 状态 | 选择性 | route | L | recall@10 | avg ms | p99 ms | QPS |
|---|---|---|---:|---:|---:|---:|---:|
| 1M initial | u50 | graph | 100 | 98.52 | 8.716 | 9.420 | 114.73 |
| 1M initial | u75 | graph | 75 | 98.65 | 6.712 | 7.293 | 148.98 |
| 1M initial | u100 | graph | 75 | 98.96 | 6.691 | 7.151 | 149.44 |
| 750k after delete | u50 | graph | 125 | 98.28 | 10.603 | 11.353 | 94.31 |
| 750k after delete | u75 | graph | 125 | 98.54 | 10.637 | 11.394 | 94.01 |
| 750k after delete | u100 | prefilter | 10 | 99.90 | 16.351 | 19.733 | 61.16 |
| 1M after reinsert | u50 | prefilter | 10 | 99.87 | 11.890 | 13.716 | 84.10 |
| 1M after reinsert | u75 | graph | 125 | 98.38 | 10.690 | 11.361 | 93.54 |
| 1M after reinsert | u100 | graph | 100 | 98.16 | 8.672 | 9.360 | 115.31 |

判断：直接构建 1M 的高选择性图搜索已经恢复到合理区间；删除/重插后仍存在高选择性延迟劣化。

---

## exp4 图

![exp4 1m initial](../experiments/exp4_delete_reinsert_selectivity/selectivity_1m_initial.png)

![exp4 750k after delete](../experiments/exp4_delete_reinsert_selectivity/selectivity_750k_after_delete.png)

![exp4 1m after reinsert](../experiments/exp4_delete_reinsert_selectivity/selectivity_1m_after_reinsert.png)

---

## 需求 6：低选择性 auto route

结论：最新 exp4 已经避免低选择性误走图搜索。

| 状态 | u1e-03 到 u30 route |
|---|---|
| 1M initial | 全部 prefilter |
| 750k after delete | 全部 prefilter |
| 1M after reinsert | 全部 prefilter |

高选择性 route 不是固定 graph，而是按满足 recall 后的单线程 latency 选择。因此会出现 u50/u100 选 prefilter 的点：这不是 tau_m 单独决定，而是最终 route/L 竞选结果。

---

## 需求 7：RSS

结论：当前结果满足 `RSS <= 30MB`。

本轮 exp4 已按要求改成干净单查询进程 RSS 测量：

| 状态 | 单查询进程 RSS 范围 |
|---|---:|
| 1M initial | 约 12.55-15.76 MB |
| 750k after delete | 约 15.51-18.62 MB |
| 1M after reinsert | 约 16.71-19.99 MB |

这个数值是单查询 driver 进程总 RSS。RSS 子进程不加载 query file、groundtruth、query spmat 或批量查询缓存，只通过命令行接收一个 query vector 和一个过滤 label；PQ compressed 走 mmap，并启用低内存搜索模式。因此该字段可以作为“单进程总内存峰值”的验收口径。

`1M initial` 和 `1M after reinsert` 的 RSS 不完全相同，主要原因是 tag 映射状态不同：初始直接构建索引没有 `_disk.index.tags` 时走 equal mapping，不分配 dense tag table；删除/重插后 tag 顺序不再 identity，加载 `_disk.index.tags` 时会常驻一份 dense tag table，因此 RSS 会系统性高一些。当前实现已经把 tag 加载改成 move 语义，避免加载阶段额外复制一份 tag vector，所以差距回落到 4MB 左右。

![exp4 rss](../experiments/exp4_delete_reinsert_selectivity/rss_by_selectivity.png)

---

## 需求 8：索引膨胀率

结论：满足 `extra_over_raw_ratio <= 1.0`。

| 点数 | total/raw | extra/raw |
|---:|---:|---:|
| 250k | 1.6348 | 0.6348 |
| 500k | 1.6342 | 0.6342 |
| 750k | 1.6341 | 0.6341 |
| 1M | 1.6340 | 0.6340 |

![exp5](../experiments/exp5_index_bloat_by_size/index_bloat_by_size.png)

---

## Baseline：直接构建 1M 强制 route

exp_baseline 用于回答“直接构建 1M 时 graph 高选择性是否本应 <10ms”。

结论：是的，PQ32 下直接构建 1M 的高选择性 graph 可以低于 10ms。

| route | 选择性 | L | recall@10 | avg ms | QPS |
|---|---|---:|---:|---:|---:|
| graph | u50 | 100 | 98.52 | 9.022 | 108.87 |
| graph | u75 | 75 | 98.67 | 6.895 | 142.31 |
| graph | u100 | 75 | 99.09 | 7.042 | 139.34 |
| prefilter | u50 | 10 | 99.85 | 10.762 | 87.84 |
| prefilter | u75 | 10 | 99.91 | 15.111 | 62.50 |
| prefilter | u100 | 10 | 99.92 | 19.892 | 47.33 |

![baseline](../experiments/exp_baseline/baseline_prefilter_vs_graph.png)

---

## 当前未完成/未达标项

| 项目 | 状态 | 说明 |
|---|---|---|
| exp1 2/1 线程完整点 | 未完成 | 按中途决策，4 线程完成后停止 exp1 |
| 插入索引最终 recall 等同直接构建 | 未达标 | 500k seed 仍低 1.298 pct |
| 删除/重插后所有单线程 avg latency <10ms | 部分未达标 | 删除后 u50/u75，重插后 u50/u75，重插 u50 prefilter 超过 10ms |
| RSS <=30MB | 通过 | 干净单查询进程 RSS 峰值 19.99MB |
| 插入期间严格 recall 验证 | 未覆盖 | 需要按 live_count checkpoint 生成 truthset |
| OpenHarmony/Mate60 真机 | 未覆盖 | 当前全部是本机环境 |
| 范围查询、多标签组合查询 | 未覆盖 | 当前主要是等值选择性 workload |

---

## 下一步建议

1. 针对 exp2：继续追 graph 质量差异，而不是只调 search L。建议比较增量插入和直接构建的入边/出边分布、连通性、入口点覆盖、删除节点导航保留策略，以及 final_merge 后图修复效果。
2. 针对 exp4 latency：删除/重插后高选择性需要更大的 L 才能到 98 recall，说明图质量或导航路径变差；优先定位重插后图边质量，而不是继续调 tau_m。
3. 针对 RSS：当前单查询进程总 RSS 已低于 30MB；后续只需防止回归，保留 `rss_single_query_driver.jsonl` 作为诊断依据。
4. 针对验收矩阵：补一个 checkpoint recall 实验，固定 500k/750k/1M live_count 后生成 truthset，再验证插入期间 recall。

---

## 一句话版本

动态更新链路已经跑通；PQ32 修复了直接构建 1M 高选择性 graph 延迟异常；删除/重插后 recall 可以靠 route/L 校准保持 98 以上；单查询进程总 RSS 已降到 19.99MB 峰值；但增量图质量仍低于直接构建，删除/重插后高选择性延迟还有劣化。
