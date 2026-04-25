# PipeANN Hybrid 过滤

作者：GitHub Copilot  
更新日期：2026年4月24日

# 文档定位

本文档描述当前主树中已经实现的 PipeANN hybrid 过滤架构，而不是实施计划、阶段推进记录或需求草案。文中的文件格式、运行时控制流、状态机、持久化语义和测试入口均以当前代码为准。

本文档只覆盖 PipeANN 主树中的正式实现，重点说明以下内容：

1. 静态 hybrid 路由如何构建、加载和执行；
1. 动态索引如何维护 live labels、live candidate counting 和后台阈值重标定；
1. 正式 binary、正式测试和 PipeANN-only 实验入口如何组织。

本文档明确不再承担以下职责：

1. 不再描述已经删除的 `hybrid_rebuild_context` 实验目录；
1. 不再按阶段性待交付事项来组织叙述；
1. 不再讨论 Milvus、Qdrant、FilteredVamana 或其他系统的对比实验方案。

# 架构总览

当前 hybrid 过滤架构由两条运行时路径组成：

1. **静态路径**：`SSDIndex` 负责只读 hybrid runtime 的加载、候选规模估计、自动路由和 prefilter / graph-only 查询执行；
1. **动态路径**：`DynamicSSDIndex` 在 `SSDIndex` 之上维护 live labels、live bitsets、checkpoint / merge 写回和后台阈值重标定。

在语义上，当前正式实现只支持以下两类标签过滤：

1. `intersect`
1. `subset`

`range` 或其他 selector 不接入 hybrid 自动路由，统一回退到 graph-only 路径。

当前路由决策统一围绕候选规模阈值 $\tau_m$ 展开。对任意一条 hybrid 查询，系统先估计候选规模 $m(F)$，然后执行如下决策：

$$
m(F)=0 \Rightarrow \text{prefilter-fast-return}
$$

$$
0 < m(F) \le \tau_m \Rightarrow \text{prefilter}
$$

$$
m(F) > \tau_m \Rightarrow \text{graph-only}
$$

其中：

1. `prefilter-fast-return` 表示候选集合为空，直接清空结果缓冲并返回；
1. `prefilter` 表示先物化候选 ID，再执行 PQ shortlist 和 SSD 精排；
1. `graph-only` 表示继续走现有 `pipe_search(...)` 过滤图查询路径。

# 子系统组成

当前实现可分为六个正式子系统。

## 1. DenseBitset 运行时

对应文件：

1. `include/filter/densebit_index.h`
1. `src/filter/densebit_index.cpp`

职责：

1. 加载 `<index_prefix>_labels.densebit`；
1. 以 bitset 方式对 `intersect` / `subset` 查询做候选计数；
1. 在需要走 prefilter 时物化候选 ID；
1. 支持从标签 sidecar 回写与恢复 `labels_by_point`。

## 2. Hybrid metadata

对应文件：

1. `include/filter/hybrid_metadata.h`
1. `src/filter/hybrid_metadata.cpp`

职责：

1. 持久化 `tau_m`、selector mask、标定版本、densebit header 快照和 calibration buckets；
1. 在 load 时校验 metadata 与 sidecar / npoints 的一致性；
1. 在 checkpoint、merge 和后台重标定期间原子发布 metadata 更新。

## 3. 统一 hybrid 查询入口

对应文件：

1. `include/ssd_index.h`
1. `src/ssd_index.cpp`
1. `src/search/hybrid_search.cpp`

职责：

1. 在 `SSDIndex::load()` 期间自动加载 hybrid runtime；
1. 暴露统一的 `hybrid_search(...)`；
1. 根据 `candidate_count`、`tau_m`、metadata flags 和 route override 做路径决策；
1. 返回 `HybridQueryStats` 作为路由侧统计面。

## 4. Prefilter 执行器

对应文件：

1. `include/filter/hybrid_route.h`
1. `src/search/hybrid_prefilter.cpp`

职责：

1. 对候选 ID 列表做 PQ 近似打分；
1. 维护 shortlist；
1. 对 shortlist 做精确向量读取和精排；
1. 输出标准 `QueryStats`。

## 5. Dynamic live state

对应文件：

1. `include/dynamic_index.h`
1. `src/update/dynamic_index.cpp`
1. `src/update/delete_merge.cpp`

职责：

1. 从静态 sidecar 初始化 live labels；
1. 在 insert / delete / label update 后维护 live IDs、live labels 和 live bitsets；
1. 在查询期间优先消费 live bitsets，而不是磁盘 sidecar 的旧快照；
1. 在 checkpoint / merge 后把 live state 原子写回正式 sidecar 与 metadata。

## 6. 后台阈值重标定

对应文件：

1. `include/dynamic_index.h`
1. `src/update/dynamic_index.cpp`
1. `tests/calibrate_hybrid_threshold.cpp`

职责：

1. 用 live labels 快照重新估计 `tau_m`；
1. 使用单线程 worker 在前台负载允许时执行重标定；
1. 在 metadata 中维护 pending / running 状态位；
1. 成功后推进 `threshold_version`、`n_calib` 和 `n_live_snapshot`。

# 正式产物与命名规则

当前主树只使用以下 hybrid 相关正式产物：

```text
<index_prefix>_disk.index
<index_prefix>_disk.index.tags
<index_prefix>_mem.index
<index_prefix>_labels.densebit
<index_prefix>_hybrid.meta
<index_prefix>_labels.densebit.tmp
<index_prefix>_hybrid.meta.tmp
```

命名规则如下：

1. `<index_prefix>_disk.index` 是磁盘主索引；
1. `<index_prefix>_labels.densebit` 是 hybrid 路由使用的标签 sidecar；
1. `<index_prefix>_hybrid.meta` 是阈值与标定 metadata；
1. `.tmp` 文件只在原子写入窗口中出现，不是长期产物；
1. build 完成后，运行时不再依赖原始 `.spmat` 路径，只依赖 `index_prefix` 派生的 sidecar 和 metadata。

所有 hybrid 运行时资产均从 `index_prefix` 派生：

1. `DenseBitsetIndex::default_sidecar_path(index_prefix)` 生成 sidecar 路径；
1. `HybridMetadata::default_metadata_path(index_prefix)` 生成 metadata 路径；
1. `SSDIndex::load_hybrid_runtime(index_prefix)` 以此为唯一加载入口。

# Build、Calibration 与 Load 生命周期

## Build：sidecar 在正式构建内生成

当前 build 过程已经把 densebit sidecar 生成并入正式主链路。

关键代码位于 `src/utils/index_build_utils.cpp`：

1. `build_disk_index(...)` 接收 `label_source_file`；
1. `create_disk_layout(...)` 在写完磁盘 metadata 后，若 `label_source_file` 非空，则调用 `build_densebit_sidecar(...)`；
1. `build_densebit_sidecar(...)` 读取 `.spmat`，写出固定命名的 `<index_prefix>_labels.densebit`。

因此当前 build 生命周期是：

1. 构建主索引；
1. 写出 `<index_prefix>_disk.index`；
1. 基于 `label_source_file` 写出 `<index_prefix>_labels.densebit`；
1. build 不会自动生成 `<index_prefix>_hybrid.meta`。

当前实现中，threshold calibration 仍是显式的 build 后步骤，而不是 build 内部的隐式副作用。

## Calibration：metadata 是显式后处理产物

当前正式标定入口是 `tests/calibrate_hybrid_threshold.cpp`。

该工具负责：

1. 加载已构建的 disk index 与 densebit sidecar；
1. 读取查询向量和查询标签；
1. 用固定随机种子 `20260423` 采样查询；
1. 对每条采样查询分别测量 prefilter 和 graph-only 延迟；
1. 按 `next_power_of_two(candidate_count)` 聚合 buckets；
1. 生成 `tau_m` 与 calibration buckets；
1. 写出 `<index_prefix>_hybrid.meta`。

当前主树中不存在“没有 metadata 也自动启用 hybrid 自动路由”的路径。

## Load：`SSDIndex::load()` 自动装配 hybrid runtime

`SSDIndex::load()` 在加载磁盘索引、page layout 和 tags 之后，会直接调用 `load_hybrid_runtime(index_prefix)`。

当前 load 行为如下：

1. 若找不到 `<index_prefix>_labels.densebit`，则 hybrid runtime 关闭；
1. 若 sidecar 加载失败，hybrid runtime 关闭；
1. 若找不到 `<index_prefix>_hybrid.meta`，hybrid runtime 关闭；
1. 若 metadata 与 densebit header 或当前 `meta_.npoints` 不一致，hybrid runtime 关闭；
1. 只有 sidecar 和 metadata 都通过校验时，`hybrid_runtime_enabled_` 才置为 true。

因此，当前自动路由依赖以下三个条件同时成立：

1. 磁盘索引可正常加载；
1. densebit sidecar 可正常加载；
1. hybrid metadata 可正常加载且校验通过。

# DenseBitset sidecar 结构与语义

当前 sidecar 文件头定义在 `include/filter/densebit_index.h`：

```cpp
struct DenseBitsetFileHeaderV1 {
  uint64_t magic = 0;
  uint64_t version = 0;
  uint64_t npoints = 0;
  uint64_t nlabels = 0;
  uint64_t words_per_label = 0;
  uint64_t nnz = 0;
};
```

payload 组织方式为 label-major：

1. 先写 header；
1. 再写 `nlabels * words_per_label` 个 `uint64_t`；
1. 每个 label 对应一整行 bitset；
1. bit 位置直接映射 point ID；
1. tail word 会施加 mask，防止越界点位参与计数。

当前运行时语义如下：

1. `subset`：对多个 label 的 bitset 做按位与；
1. `intersect`：对多个 label 的 bitset 做按位或；
1. 查询标签会先排序和去重；
1. `count_candidates(...)` 只做计数，不物化候选 ID；
1. `materialize_candidates(...)` 在已经选择 prefilter 路径后才会把 bitset 展开成 point ID 数组。

查询期 scratch 结构为：

```cpp
struct HybridQueryScratch {
  std::vector<uint64_t> bitset_words;
  std::vector<uint32_t> candidate_ids;
  std::vector<uint32_t> normalized_labels;
};
```

当前实现有两个重要优化点：

1. 先 count 再 route，避免所有查询都做 candidate materialization；
1. 只有 `candidate_count <= tau_m` 的查询才真正生成 candidate IDs。

# Hybrid metadata 结构与语义

当前 metadata 定义在 `include/filter/hybrid_metadata.h`，由 256-byte header 和可变长 bucket 数组组成。

header：

```cpp
struct HybridMetadataHeaderV1 {
  uint64_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint64_t flags;
  uint64_t route_selector_mask;
  uint64_t tau_m;
  uint64_t n_calib;
  uint64_t n_live_snapshot;
  uint64_t threshold_version;
  uint64_t calib_epoch_sec;
  uint64_t calib_query_count;
  uint64_t calib_bucket_count;
  uint64_t calib_k;
  uint64_t calib_mem_L;
  uint64_t calib_beamwidth;
  uint64_t calib_l_search;
  uint64_t densebit_npoints;
  uint64_t densebit_nlabels;
  uint64_t densebit_words_per_label;
  uint64_t densebit_nnz;
  uint64_t reserved[13];
};
```

bucket：

```cpp
struct HybridCalibrationBucketV1 {
  uint64_t candidate_upper_bound;
  uint64_t query_count;
  uint64_t prefilter_p50_us;
  uint64_t graph_p50_us;
  uint64_t reserved;
};
```

当前 flags 语义如下：

1. bit0：metadata 有效；
1. bit1：calibration 有效；
1. bit2：允许 prefilter 路由；
1. bit3：pending recalibration；
1. bit4：running recalibration。

当前 `route_selector_mask` 语义如下：

1. bit0：`intersect` 已标定；
1. bit1：`subset` 已标定。

当前 metadata 在 load 时执行以下校验：

1. `magic`、`version`、`header_bytes`；
1. 文件长度与 bucket 数组长度一致；
1. routing-ready 模式下要求 bit0 和 bit1 已置位；
1. selector mask 只允许使用低两位；
1. densebit header 与 sidecar 一致；
1. `n_calib == meta_.npoints`。

当前 metadata 的持久化策略是统一的 `.tmp + fsync + rename` 原子发布。

# 静态查询路径：统一 hybrid route

当前正式查询入口是 `SSDIndex::hybrid_search(...)`，实现位于 `src/search/hybrid_search.cpp`。

其控制流如下：

1. 将 `filter_data` 解码为 `[count][label_0...label_n]`；
1. 根据 `HybridFilterKind` 选择 `LabelIntersectionSelector` 或 `LabelSubsetSelector`；
1. 若 filter kind 不支持，则直接走 graph-only fallback；
1. 若 hybrid runtime 未启用，则直接走 graph-only fallback；
1. 若 metadata 不允许该 selector 走 prefilter，则直接走 graph-only fallback；
1. 用 densebit sidecar 计算 `candidate_count`；
1. 若 `candidate_count == 0`，则执行 fast-return；
1. 若 `route_override == kForcePrefilter`，则直接走 prefilter；
1. 若 `route_override == kForceGraphOnly`，则直接走 graph-only；
1. 否则按 `candidate_count <= tau_m` 决定 prefilter 或 graph-only。

当前 `HybridQueryStats` 包含以下运行时可观测字段：

1. `candidate_count`
1. `threshold`
1. `threshold_version`
1. `decision`
1. `route_overhead_us`

因此当前 hybrid 查询不仅返回 ANN 结果，还显式暴露了路由面统计。

# Prefilter 执行器

当前 prefilter 执行器由 `SSDIndex::hybrid_prefilter_search(...)` 实现，文件位于 `src/search/hybrid_prefilter.cpp`。

控制流如下：

1. 若 `candidate_ids` 为空，则直接返回 0；
1. 从 query buffer pool 取出 `QueryBuffer`；
1. 用 `nbr_handler->compute_dists(...)` 批量计算候选近似距离；
1. 用最大堆维护 shortlist；
1. 对 shortlist 中的 point IDs 调用 `get_vector_by_id(...)` 读原始向量；
1. 用 `dist_cmp->compare(...)` 做精确距离计算；
1. `partial_sort` 后回填最终的 `tags` 和 `distances`。

当前 shortlist 大小通过 `compute_prefilter_rerank_l(...)` 自适应控制，其依据是：

1. `candidate_count`
1. `total_points`
1. `k_search`

动态路径中还支持 `total_points_override`，因此 live prefilter 的 rerank 规模是基于 `live_point_count_` 而不是静态 `meta_.npoints`。

# Dynamic live state 架构

当前动态 hybrid 运行时构建在 `DynamicSSDIndex` 上。

## 初始化

构造期间会执行 `initialize_live_state_from_disk(_disk_index_prefix_in)`，从当前 sidecar 和 tag 映射初始化：

1. `live_ids_by_tag_`
1. `live_labels_by_tag_`
1. `live_labels_by_id_`
1. `live_label_bitsets_`
1. `live_present_bitset_`
1. `live_point_count_`

因此，dynamic path 的 label 真源在进入运行态之后不再是磁盘节点尾部的旧 label 区域，而是内存中的 live 状态结构。

## 核心 live 状态

当前 `DynamicSSDIndex` 中最关键的 live 字段如下：

1. `live_ids_by_tag_`：tag 到 live point ID 的映射；
1. `live_labels_by_tag_`：tag 到 labels 的映射；
1. `live_labels_by_id_`：point ID 到 labels 的映射；
1. `live_label_bitsets_`：label ID 到 live bitset 的映射；
1. `live_present_bitset_`：当前存在的 live IDs；
1. `live_point_count_`：唯一的 live vector count 信号。

## 更新语义

当前动态更新语义如下：

1. `insert(...)`：新增 live ID，更新 `live_ids_by_tag_`、`live_present_bitset_` 和 `live_point_count_`；
1. `lazy_delete(...)`：清理 live labels、清位 present bit、移除 tag 映射并递减 `live_point_count_`；
1. `update_labels(...)`：更新 live labels 和 live bitsets，但不改变 `live_point_count_`；
1. `final_merge(...)`：在 merge 完成后重建 live IDs，并刷新 sidecar / metadata；
1. `checkpoint()`：当 live IDs 仍是紧凑连续编号时，把当前 live labels 原子写回 sidecar。

## 动态查询路径

`DynamicSSDIndex::search(...)` 会优先消费 live bitsets，而不是静态 densebit sidecar：

1. prefilter 路径通过 `compute_live_candidate_bitset(...)` 直接对 `live_label_bitsets_` 做 candidate counting；
1. graph 路径使用 `LiveLabelSelector`，它从 `live_labels_by_id_` 读取标签真源；
1. 若走 prefilter，则通过 `materialize_candidate_ids(...)` 生成 live candidate IDs；
1. 调用 `_disk_index->hybrid_prefilter_search(...)` 时传入 `live_point_count_` 作为 `total_points_override`。

因此，dynamic hybrid query 的路由输入与过滤语义在查询时已经是 live-consistent 的。

# 后台阈值重标定状态机

当前后台重标定已经是正式实现的一部分，而不是未来计划。

## 配置面

当前配置对象包括：

1. `HybridForegroundCounters`
1. `HybridForegroundBudget`
1. `HybridRecalibrationState`
1. `HybridRecalibrationDataset`
1. `HybridRecalibrationConfig`

其中 `HybridForegroundCounters` 当前包含四个前台门限信号：

1. `active_queries`
1. `waiting_queries`
1. `active_high_priority_tasks`
1. `background_recalibration_disabled`

## 触发条件

当前重标定触发条件实现为：

$$
\left|n_{calib} - n_{live}\right| \times 10 > n_{calib}
$$

即 live 向量数相对上次 calibration 的偏移超过 10%。

只有在以下前提同时成立时，系统才会从 `kIdle` 进入 `kPending`：

1. 重标定已配置；
1. metadata 文件存在；
1. 偏移超过 10%；
1. 当前状态仍为 `kIdle`。

## Worker 生命周期

当前 worker 生命周期如下：

1. `configure_hybrid_recalibration(...)` 写入配置并懒启动 worker；
1. `ensure_hybrid_recalibration_worker_started()` 在第一次需要时创建后台线程；
1. worker 等待 `hybrid_recalibration_signal_count_` 变化；
1. 观察到 `kPending` 且前台负载允许时，CAS 进入 `kRunning`；
1. `run_hybrid_recalibration_once()` 重新生成 calibration buckets 和 `tau_m`；
1. 成功则发布新 metadata，并把状态回到 `kIdle`；
1. 失败则回到 `kPending`，等待下一次可运行窗口；
1. 析构时 `stop_hybrid_recalibration_worker()` 会 join worker。

## 前台负载门限

当前 `can_run_hybrid_recalibration_now()` 使用如下门限：

1. `background_recalibration_disabled == false`
1. `active_queries <= active_queries_low_watermark`
1. `waiting_queries <= waiting_queries_low_watermark`
1. `active_high_priority_tasks == 0`

前台查询路径通过 `ForegroundQueryGuard` 自动维护 `waiting_queries` 和 `active_queries`；
merge 路径通过 `HighPriorityTaskGuard` 维护 `active_high_priority_tasks`。

## 重标定算法与静态 calibration 的关系

当前后台重标定复用了正式 calibration 工具的分桶语义，而不是重新实现第二套 metadata 格式：

1. 读取 `HybridRecalibrationConfig.datasets`；
1. 以固定随机种子采样 query IDs；
1. 对每条采样 query 先基于 live bitsets 计算 `candidate_count`；
1. 再分别测量 graph-only 和 prefilter 延迟；
1. 按 `next_power_of_two(candidate_count)` 聚合 bucket；
1. 取满足 `prefilter_p50_us <= graph_p50_us` 的最大 bucket 作为新的 `tau_m`；
1. 刷新 `n_calib`、`n_live_snapshot` 和 `threshold_version`。

# Sidecar 与 metadata 的持久化语义

当前系统中，live sidecar 写回与 metadata 发布是同一类高一致性操作。

## Checkpoint

`checkpoint()` 的当前语义：

1. 先持久化 journal；
1. 若 live IDs 仍为紧凑连续编号，则调用 `persist_live_hybrid_state(...)`；
1. 该函数先把 live labels 写成新的 `<index_prefix>_labels.densebit`；
1. 再刷新 metadata 中的 densebit header 和 `n_live_snapshot`；
1. 若 `n_calib != live_count`，则调用 `disable_routing()` 清空自动路由能力；
1. 最后重新 `load_hybrid_runtime(...)`。

若 live IDs 不紧凑，则 checkpoint 明确跳过 hybrid sidecar 写回。

## Merge

`final_merge(...)` 的当前语义：

1. 调用底层 `merge_deletes(...)`；
1. `src/update/delete_merge.cpp` 会把旧前缀的 `_hybrid.meta` 一并复制到新前缀；
1. merge 后重建 live IDs；
1. 再次通过 `persist_live_hybrid_state(...)` 刷新 sidecar / metadata；
1. 若当前 live count 与 `n_calib` 不一致，则自动路由按设计被禁用，等待后台重标定恢复。

## Metadata I/O 串行化

当前 `DynamicSSDIndex` 内部引入了 `hybrid_metadata_io_lock_`，用于串行化以下动作：

1. pending / running 状态位写回；
1. checkpoint / merge 的 sidecar + metadata 刷新；
1. 后台重标定发布新 metadata；
1. 相关 runtime reload。

这样可以避免 worker 和 checkpoint / merge 在 sidecar / metadata 更新窗口内互相覆盖。

# 正式测试与验证面

当前主树已有四类正式验证入口。

## 1. Metadata round-trip

文件：`tests/hybrid_metadata_roundtrip.cpp`

覆盖：

1. metadata create / write / load；
1. densebit header 校验；
1. npoints 校验；
1. 损坏 metadata 的异常路径。

## 2. 静态 hybrid 查询驱动

文件：`tests/search_disk_index_hybrid.cpp`

覆盖：

1. `auto`
1. `validate-auto`
1. `prefilter`
1. `graph`

该驱动是当前正式的 hybrid benchmark 入口。

## 3. Dynamic live state regression

文件：`tests/dynamic_hybrid_live_state.cpp`

覆盖：

1. pre-checkpoint live label update；
1. insert 后 live hybrid prefilter；
1. delete 后 fast-return；
1. checkpoint sidecar 写回；
1. merge 后 metadata 与 live state 保持一致。

## 4. Dynamic recalibration regression

文件：`tests/dynamic_hybrid_recalibration.cpp`

覆盖：

1. 10% drift 触发 `kPending`；
1. 禁用后台时只进入 pending，不推进版本；
1. 恢复后台后会经过 running 并回到 idle；
1. 成功后清除 bit3 / bit4，推进 `threshold_version` 并刷新 `n_calib`。

# PipeANN-only 实验入口

本文档中的实验范围只保留 PipeANN，不包含任何其他系统。

## 实验范围

当前实验约束如下：

1. 只构建和运行 PipeANN；
1. 只使用主树中的正式 binary 和脚本；
1. 不运行 FilteredVamana、Milvus、Qdrant 或其他系统；
1. 所有实验产物都以 PipeANN 的 `index_prefix`、densebit sidecar 和 hybrid metadata 为中心。

## 当前正式实验入口

当前主树已有四类正式实验入口：

1. `build/tests/build_disk_index`
1. `build/tests/calibrate_hybrid_threshold`
1. `build/tests/search_disk_index_hybrid`
1. `scripts/pipeann_hybrid_experiment.py`
1. `scripts/pipeann_memory_breakdown.py`

对应的最小流程为：

1. 用 `build_disk_index` 构建磁盘索引与 densebit sidecar；
1. 用 `calibrate_hybrid_threshold` 生成 `<index_prefix>_hybrid.meta`；
1. 用 `search_disk_index_hybrid` 运行 `auto`、`validate-auto`、`prefilter` 或 `graph` 模式的正式查询；
1. 用 `pipeann_hybrid_experiment.py` 基于 `<index_prefix>_labels.densebit` 或原始 `spmat` 标签扫描真实选择性，生成正式实验用的随机单标签 query workload，并运行 PipeANN-only 搜索与绘图。
1. 用 `pipeann_memory_breakdown.py` 直接基于 `sift1m` 与 `yfcc10m` 正式预设运行 resident、massif / heaptrack、perf 三类外部内存分析。

当前面向 YFCC-10M 的正式实验编排脚本是 `scripts/pipeann_hybrid_experiment.py`，其子命令职责如下：

1. `scan-single-label`：扫描 densebit sidecar 与 query labels，输出单标签频次、query 覆盖和推荐的单标签选择性点；
1. `generate-random-single-label-workloads`：为扫描选中的真实单标签以及人工 50% / 75% / 100% 高选择性标签，复用完整原始 query bin，并按 bucket 重写 one-hot query label；
1. `generate-synthetic-high-selectivity`：单独生成 50% / 75% / 100% 的人工 base label，并为完整原始 query bin 生成对应的重标记 query labels；
1. `generate-uniform-exact-selectivity-workloads`：针对无标签数据集，按目标选择性精确采样 base 向量生成 uniform 分布标签，并为完整原始 query bin 生成对应的重标记 query labels；
1. `prepare-index-prefix-for-labels`：复用既有图索引 prefix，生成一套与新标签文件匹配的 densebit sidecar；
1. `build-manifest-from-summary`：把 workload summary 转换成正式可运行的 selectivity manifest；
1. `prepare`：基于 densebit sidecar 计算真实候选规模并按选择性分桶；
1. `calibrate-rerank`：基于 base 向量与标签候选集，为 prefilter bucket 搜索满足 `recall@10 >= 98%` 的最小 rerank 值，并输出 calibration JSON；
1. `run`：调用 `search_disk_index_hybrid`，同时采集 JSONL 指标和单查询峰值内存；
1. `plot`：直接消费 JSONL 聚合结果并输出延迟、QPS、峰值内存图；
1. `all`：按 `prepare -> run -> plot` 顺序串行执行。

`scripts/pipeann_memory_breakdown.py` 的正式内置预设现在统一使用单查询 `probe_query.bin` / `probe_query.spmat`，不再默认使用整批 query 输入。

## YFCC-10M 输入约定

当前主树 README 已给出 YFCC-10M 的基本命名约定：

1. base label 使用 `base.metadata.10M.spmat` 类文件；
1. query label 使用 `query.metadata.public.100K.spmat` 类文件；
1. hybrid build / calibration / query 都以这些正式输入为起点。

当前单标签实验进一步收敛为两段式输入流程：

1. 原始 `query.metadata.public.100K.spmat` 只用于 `scan-single-label` 阶段，负责做真实标签分布扫描与点位推荐；
1. 正式单标签实验统一复用原始 `query.public.100K.u8bin`，但为每个选择性单独写一份 one-hot query label 文件；
1. `data/yfcc100M/random_single_label_workloads/real_selected_labels` 子目录保存真实标签对应的重标记 query labels；
1. `data/yfcc100M/random_single_label_workloads/synthetic_high_selectivity` 子目录保存人工 50% / 75% / 100% 的 query label 与 probe label；
1. 运行时真正切换 base 标签语义靠的是 prefix 绑定的 `_labels.densebit` 与 `_hybrid.meta`，而不是仅切换 query label 文件；
1. `random_single_label_workloads_summary.json` 是后续实验选取 query 文件的总索引。

## SIFT1M 输入与标签约定

SIFT1M 没有原生标签，因此当前实验采用精确选择性的 uniform synthetic 标签：

1. 原始 texmex 数据已转换为 `data/sift1m/sift_base.bin`、`data/sift1m/sift_query.bin` 和 `data/sift1m/sift_groundtruth.ibin`；
1. `generate-uniform-exact-selectivity-workloads` 在 `data/sift1m/uniform_exact_selectivity` 下生成 `base.uniform_exact_selectivity.spmat`；
1. 当前默认点位为 11 档：`1e-5`、`3e-5`、`1e-4`、`3e-4`、`1e-3`、`3e-3`、`1e-2`、`1e-1`、`0.5`、`0.75`、`1.0`；
1. 每个 bucket 都复用完整 `sift_query.bin`，只改对应的 `queries.spmat` 与 `probe_query.spmat`；
1. 这套 workload 已通过 `build-manifest-from-summary` 转成 `data/sift1m/sift1m_uniform_manifest.json`。

本文档不再为多系统比较保留额外 schema。后续若执行 YFCC-10M 实验，默认仅产出 PipeANN 自身的：

1. latency
1. QPS
1. peak memory
1. selectivity 相关聚合结果

## 当前实验输出面

当前 `search_disk_index_hybrid` 已能输出：

1. `L`
1. `QPS`
1. `AvgLat(us)`
1. `P99 Lat`
1. `Mean Hops`
1. `Mean IOs`
1. `MeanCand`
1. `MeanRouteUs`
1. route decision breakdown
1. `Recall@K`

在保持上述表格输出兼容的同时，当前 driver 还支持可选的 `--jsonl-output` 参数，用于把每个 `L` 对应的一条机器可读记录追加写入 JSONL 文件。

当 `scripts/pipeann_hybrid_experiment.py run` 传入 `--prefilter-rerank-json` 时，JSONL 聚合结果还会额外记录：

1. `prefilter_rerank_l`
1. `prefilter_rerank_source`

因此当前 PipeANN-only 实验的正式搜索面已经具备：

1. 路由统计；
1. 搜索延迟；
1. 吞吐；
1. recall；
1. bucket 级 rerank 配置回填。

本文档不对其他系统的输出格式做任何约束。

# 当前实现的关键事实汇总

1. densebit sidecar 已在正式 build 链路内生成，而不是外部预处理物；
1. hybrid metadata 仍是显式 calibration 后处理产物，而不是 build 内隐式副作用；
1. `SSDIndex::load()` 会自动尝试装配 hybrid runtime；
1. 自动路由只在 sidecar + metadata 全部存在且通过校验时启用；
1. 动态路径查询使用 live bitsets，不依赖磁盘 sidecar 的旧快照；
1. checkpoint / merge 会在可能时把 live state 持久化回正式 sidecar；
1. live count 与 `n_calib` 偏移超过 10% 时，后台 worker 会触发重标定；
1. background recalibration 与 sidecar / metadata 发布已通过同一把 I/O 锁串行化；
1. 正式实验范围当前只保留 PipeANN 主树自己的 build / calibration / hybrid search 流程；
1. PipeANN-only 实验脚本已能基于 densebit sidecar 做真实选择性分桶，并输出延迟、QPS、峰值内存图；
1. `pipeann_hybrid_experiment.py` 已支持 `calibrate-rerank`，可为 prefilter bucket 输出最小 rerank 配置并直接供 `run` 消费；
1. 正式实验结果统一落盘到 `experiments/`，不再写入 `build/`；
1. 当前 YFCC-10M 单标签正式实验默认复用原始公开 query bin，并按选择性重写 query label；
1. 当前 SIFT1M 实验默认使用 uniform exact-selectivity synthetic labels，并复用完整原始 query bin；
1. `pipeann_memory_breakdown.py` 的正式 preset 已统一切到单查询 probe 输入。

# 当前实验产物

当前已落盘的核心实验产物如下：

1. YFCC-10M 真实与人工高选择性合并结果：`experiments/yfcc10m_combined_results.jsonl`；
1. YFCC-10M 总图：`experiments/yfcc10m_selectivity_auto_l100.png`；
1. SIFT1M uniform exact-selectivity 结果：`experiments/sift1m_uniform_run/results.jsonl`；
1. SIFT1M 总图：`experiments/sift1m_uniform_auto_l100.png`；
1. SIFT1M calibrated rerank 配置：`experiments/sift1m_uniform_rerank_calibrated/calibration/prefilter_rerank_calibration.json`；
1. SIFT1M calibrated rerank 结果：`experiments/sift1m_uniform_rerank_calibrated/run/results.jsonl`；
1. SIFT1M calibrated rerank 总图：`experiments/sift1m_uniform_rerank_calibrated/sift1m_uniform_calibrated_auto_l100.png`；
1. SIFT1M 当前 calibration 结果对应 `tau_m = 131072`，并已写入 `data/sift1m/sift1m_pipeann_uniform_hybrid.meta`。

# 代码事实来源

本文档的主要事实来源如下：

1. `src/utils/index_build_utils.cpp`
1. `src/filter/densebit_index.cpp`
1. `src/filter/hybrid_metadata.cpp`
1. `src/ssd_index.cpp`
1. `src/search/hybrid_search.cpp`
1. `src/search/hybrid_prefilter.cpp`
1. `src/update/dynamic_index.cpp`
1. `src/update/delete_merge.cpp`
1. `tests/calibrate_hybrid_threshold.cpp`
1. `tests/search_disk_index_hybrid.cpp`
1. `scripts/pipeann_hybrid_experiment.py`
1. `scripts/pipeann_memory_breakdown.py`

后续如果代码行为发生变化，应优先更新上述实现文件，再同步更新本文档；本文档本身不再作为计划文件维护。