# Hybrid Combined Curve 复现说明

这个目录现在是一个独立子工程，目标是：

- 所有 hybrid rebuild context 相关的新代码都留在这里
- upstream 已有的头文件、源码、测试入口都直接复用原位置
- 在纯 upstream PipeANN 基线上，只靠这个目录就能重新编译并运行 hybrid 实验

最终产物仍然围绕这张图：

- 图：`hybrid_rebuild_context/artifacts/images/hybrid_combined.png`
- 数据：`hybrid_rebuild_context/artifacts/results_v2/hybrid_results.csv`

实验过程中生成的中间资产统一放在：

- `hybrid_rebuild_context/assets/sift1m/`
- `hybrid_rebuild_context/assets/labels/`
- `hybrid_rebuild_context/assets/cache/`
- `hybrid_rebuild_context/assets/index/`

## 目录说明

- `CMakeLists.txt`
  独立构建入口。它会把仓库根目录作为 upstream 子工程引入，并在本目录下生成 hybrid 需要的二进制。
- `src/search_disk_index_filtered_prefilter.cpp`
  本目录唯一的 hybrid 搜索 C++ 实现。它直接使用 upstream 的 `SSDIndex` / `PQNeighbor` / `SpmatLabel`。
- `include/cblas.h`
  本地 fallback 头文件，仅用于补齐某些机器上缺失的 `cblas.h`。
- `exact_hybrid_common.py`
  共享路径、命令构造、RSS 解析、cache helper。
- `build_exact_indices.py`
  构建 exact 索引。
- `build_exact_gt_batch.py`
  生成 exact GT。
- `gen_exact_labels.py`
  生成 exact labels。
- `prepare_hybrid_exact_cache.py`
  生成 `n100` / `n1` 查询子集缓存。
- `smoke_test_hybrid_exact.py`
  运行 prefilter / graph-only smoke。
- `run_hybrid_curve.py`
  生成 `hybrid_results.csv`。
- `plot_hybrid_curve.py`
  从 `hybrid_results.csv` 重绘 `hybrid_combined.png`。

## 二进制布局

执行本目录的 CMake 之后，会得到：

- `hybrid_rebuild_context/build/bin/hybrid_build_disk_index`
  来自 upstream `tests/build_disk_index.cpp` 的 staged copy。
- `hybrid_rebuild_context/build/bin/hybrid_search_disk_index_filtered`
  来自 upstream `tests/search_disk_index_filtered.cpp` 的 staged copy。
- `hybrid_rebuild_context/build/bin/hybrid_search_disk_index_filtered_prefilter`
  本目录新增的 standalone prefilter 可执行文件。

也就是说，`graph-only` 直接跑 upstream 逻辑，`prefilter` 只新增了 upstream 里没有的那部分。

## 运行前提

本目录会优先使用：

- `hybrid_rebuild_context/assets/sift1m/`
- `hybrid_rebuild_context/assets/labels/`
- `hybrid_rebuild_context/assets/cache/`
- `hybrid_rebuild_context/assets/index/`

如果实验目录里还没有 `sift_base.bin` / `sift_query.bin` / `sift_groundtruth.bin`，
helper 会从仓库已有的 `data/sift1m/` 或 `data/sift/` 复制一份到 `assets/sift1m/`。
也就是说，生成后的实验资产和最终结果都会收拢在 `hybrid_rebuild_context/` 下面。

## 从零复现

在仓库根目录执行。

1. 配置本目录的独立构建

```bash
cmake -S hybrid_rebuild_context -B hybrid_rebuild_context/build -DCMAKE_BUILD_TYPE=Release
```

2. 编译 hybrid 所需目标

```bash
cmake --build hybrid_rebuild_context/build -j $(nproc) --target \
  hybrid_build_disk_index \
  hybrid_search_disk_index_filtered \
  hybrid_search_disk_index_filtered_prefilter
```

3. 如需从零重建 exact labels / GT / cache

```bash
python3 hybrid_rebuild_context/gen_exact_labels.py
python3 hybrid_rebuild_context/build_exact_gt_batch.py
python3 hybrid_rebuild_context/prepare_hybrid_exact_cache.py
```

4. 如需重建 exact 索引

```bash
python3 hybrid_rebuild_context/build_exact_indices.py
```

5. 先跑 smoke

```bash
python3 hybrid_rebuild_context/smoke_test_hybrid_exact.py
```

6. 生成最新曲线数据

```bash
python3 hybrid_rebuild_context/run_hybrid_curve.py
```

7. 重绘图片

```bash
python3 hybrid_rebuild_context/plot_hybrid_curve.py
```

## 当前实验定义

- 数据集：SIFT1M exact
- selectivity：`0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250, 0.500, 1.000`
- prefilter：`PQ16 + local standalone prefilter binary`
- graph-only：`PQ16 + upstream filtered search binary`
- 内存指标：`/usr/bin/time -v` 的 whole-process peak RSS
- I/O：`io_uring`

## 备注

- 运行 helper 默认会设置 `PIPEANN_PQ_MMAP=1`，避免 PQ 压缩码整体常驻堆内存。
- 这个目录现在不再依赖 `overlay/` 形式的源码覆盖。
