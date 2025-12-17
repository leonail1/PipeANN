# PipeANN 索引构建分析

## 1. 性能统计现状

### 当前统计粒度

PipeANN索引构建已具备**细粒度的时间统计**，可以分析各部分工作的时间占比。

| 统计项 | 位置 | 说明 |
|--------|------|------|
| `Link time` | `src/index.cpp:648` | 整个link过程耗时 |
| `CPU Time Statistics` | `src/index.cpp:651` | 各操作的CPU时间统计 |
| `Wall Time Statistics` | `src/index.cpp:652` | 各操作的墙钟时间统计 |
| `Distance Calculation` | `src/index.cpp:651` | 距离计算时间及占比 |
| `Vamana index built in` | `src/utils/index_build_utils.cpp:434` | Vamana索引构建时间 |
| `Indexing time` | `src/utils/index_build_utils.cpp:457` | 总索引时间 |

### 查询阶段的统计（参考）

查询阶段有较详细的统计结构 (`include/utils/percentile_stats.h`)：

```cpp
struct QueryStats {
    double total_us = 0;    // 总查询时间
    double io_us = 0;       // IO时间
    double cpu_us = 0;      // CPU计算时间
    double n_cmps = 0;      // 距离计算次数
    // ...
};
```

### 索引构建细粒度统计

通过 `scripts/rebuild_index.py` 脚本可以触发索引构建并获取详细的性能统计。

以下是 SIFT1M (100万点, 128维) 磁盘索引构建的实测数据：

#### CPU 时间统计 (所有线程累计)

| 操作 | CPU时间(s) | 占比 | 调用次数 | 平均(us) |
|------|-----------|------|----------|----------|
| search | 1189.038 | 44.4% | 1000000 | 1189.0 |
| inter_insert | 1069.951 | 39.9% | 1000000 | 1070.0 |
| prune | 234.787 | 8.8% | 1000000 | 234.8 |
| cleanup_prune | 185.115 | 6.9% | 938365 | 197.3 |
| lock_update | 0.605 | 0.0% | 1000000 | 0.6 |
| **总计** | **2679.496** | 100% | - | - |

**距离计算**: 2584.342s，占总CPU时间的 **96.4%**

#### 墙钟时间统计 (并行执行)

| 操作 | 墙钟时间(s) | 占比 |
|------|------------|------|
| inter_insert | 39.049 | 72.2% |
| search | 39.044 | 72.2% |
| prune | 38.730 | 71.6% |
| cleanup_prune | 15.001 | 27.7% |
| **总墙钟时间** | **54.079** | - |

> 注：由于主循环中 search、prune、inter_insert 在同一并行区域内串行执行，墙钟时间百分比之和会超过100%。

---

## 2. 并行化部分

索引构建中被并行化的操作 (`src/index.cpp`)：

| 代码位置 | 操作 | 调度策略 |
|----------|------|----------|
| L336 | `calculate_entry_point()` - 计算入口点 | `static, 65536` |
| L559 | **主构建循环** - 搜索+修剪+插入 | `dynamic` |
| L590 | **最终清理** - 度数超限节点重新修剪 | `dynamic, 65536` |
| L889 | 邻居列表更新(compaction) | `dynamic, 1024` |

核心构建过程 (L559-585) 是完全并行的：每个点独立执行搜索、修剪、反向插入。

---

## 3. 时间度量选择

评估各部分时间占比时，需要选择合适的时间度量方式：

| 度量方式 | 适用场景 |
|----------|----------|
| **真实世界时间 (wall time)** | 评估端到端性能、用户体验、与其他系统对比 |
| **总线程时间 (CPU time)** | 评估某部分计算量占比、优化瓶颈分析 |

**建议使用总线程时间**来评估各部分时间占比，原因：

1. 并行循环内各线程工作量不均（`dynamic`调度），wall time会被最慢线程主导
2. CPU time能准确反映**计算量占比**，例如距离计算占总计算的百分比
3. 方便识别真正的优化目标（即使某部分wall time短，若CPU time占比高仍值得优化）

**获取方式示例**：
```cpp
#include <omp.h>
// 在并行区域内用 thread-local 累加器统计各部分时间
double cpu_start = omp_get_wtime();
// ... 代码 ...
```

---

## 4. 索引构建完整阶段

### 磁盘索引构建流程 (`build_disk_index`)

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段1: 数据预处理                                                │
│    - Cosine距离: 向量归一化 (normalize_data_file)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段2: 邻居编码构建 (nbr_handler->build)                         │
│    - PQ/RaBitQ 量化表训练                                        │
│    - 向量编码                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段3: Vamana图构建 (build_merged_vamana_index)                  │
│    ├─ 小数据: 直接构建                                           │
│    └─ 大数据: 分片构建 + 合并 (partition → build → merge_shards) │
│                                                                  │
│    3.1 Index::build()                                            │
│        - 加载数据                                                │
│        - link() ← 核心图构建                                     │
│                                                                  │
│    3.2 link() 内部:                                              │
│        a) calculate_entry_point() - 计算入口点                   │
│        b) 主循环 (并行):                                         │
│           - get_expanded_nodes() → 搜索候选邻居                  │
│           - prune_neighbors() → 邻居修剪                         │
│           - inter_insert() → 反向边插入                          │
│        c) final cleanup - 超度数节点重新修剪                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段4: 磁盘布局创建 (create_disk_layout)                         │
│    - 将图+向量+编码数据按页对齐写入磁盘                          │
│    - 生成元数据                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 各阶段时间占比 (实测)

基于 SIFT1M 数据集的实测结果：

| 阶段 | 主要操作 | 时间 | 占比 |
|------|----------|------|------|
| 邻居编码构建 | PQ训练+压缩 | 6.4s | 10.3% |
| **Vamana图构建** | link() | **55.3s** | **88.5%** |
| 磁盘布局 | I/O写入 | 0.7s | 1.2% |
| **总计** | - | **62.5s** | 100% |

**结论**：Vamana图构建占绝对主体（88.5%），其中距离计算占CPU时间的96.4%。

---

## 5. 测量脚本使用方式

### 重建索引并查看统计

```bash
python3 scripts/rebuild_index.py --no-labels --force
```

参数说明：
- `--no-labels`: 构建无标签索引
- `--force`: 强制删除现有索引后重建
- `--labels`: 构建带标签索引（与 `--no-labels` 互斥）

脚本会自动：
1. 删除现有索引文件
2. 调用 `build_disk_index` 构建磁盘索引
3. 调用 `build_memory_index` 构建内存采样索引
4. 输出详细的 CPU 时间和墙钟时间统计

---

## 6. link() 函数工作原理

`link()` 对每个节点执行一次查询，但查询是在**逐步构建的图**上进行：

```cpp
for (node = 0; node < n_vecs_to_visit; node++) {
    // 1. 在当前已构建的图上搜索，找到node的候选邻居
    get_expanded_nodes(node, L, pool);  
    
    // 2. 修剪候选邻居
    prune_neighbors(pool, pruned_list, ...);
    
    // 3. 设置node的出边
    _final_graph[node] = pruned_list;
    
    // 4. 反向插入：让邻居也指向node（可能触发邻居的修剪）
    inter_insert(node, pruned_list, params);
}
```

### 关键特性

| 特性 | 说明 |
|------|------|
| 查询目标 | 当前节点 `node` 的向量 |
| 查询的图 | **部分构建的图**（节点0~node-1已有边，node~N暂无边） |
| 起点 | 入口点 `_ep`（通过`calculate_entry_point`选取的中心点） |
| 结果 | 找到距离`node`最近的L个候选邻居 |

这是 **Vamana/DiskANN 的增量构建策略**：每个新节点通过在已有图上搜索来找邻居，同时通过`inter_insert`添加反向边，使图逐渐变得更稠密和连通。

---

## 7. 相关代码位置

| 功能 | 文件 | 主要函数 |
|------|------|----------|
| 内存索引构建 | `src/index.cpp` | `Index::build()`, `Index::link()` |
| 磁盘索引构建 | `src/utils/index_build_utils.cpp` | `build_disk_index()`, `build_merged_vamana_index()` |
| 邻居修剪 | `include/utils/prune_neighbors.h` | `prune_neighbors()` |
| 性能统计结构 | `include/utils/percentile_stats.h` | `QueryStats` |
| 距离计算 | `src/utils/distance.cpp` | 各距离函数实现 |