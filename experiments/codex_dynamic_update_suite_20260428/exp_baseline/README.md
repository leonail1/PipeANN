# exp_baseline 使用说明

本目录用于单独运行直接构建 SIFT1M 1M 索引上的 forced-route baseline。实验不走 auto-route，分别强制 `prefilter` 和 `graph`，对每个选择性 bucket 校准满足 `recall@10 >= 98` 的最小 `L`，再记录单线程 latency 和 QPS。

## 一键运行

在本目录或仓库任意目录运行都可以：

```bash
/mnt/data/lzg/PipeANN/experiments/codex_dynamic_update_suite_20260428/exp_baseline/start.sh --rerun-baseline
```

常用完整命令：

```bash
./experiments/codex_dynamic_update_suite_20260428/exp_baseline/start.sh \
  --rerun-baseline \
  --baseline-query-count 1000 \
  --baseline-pq-bytes 32
```

`exp_baseline/start.sh` 默认会完整重跑 baseline，并重新生成 CSV/summary/图片；实验结束后会清理临时大文件。

## 输出文件

- `results.jsonl`：所有达到 `recall@10 >= 98` 的测量点。
- `skipped.jsonl`：超过延迟阈值仍未达到 98% recall 的点。
- `calibration.jsonl`：每个 route、bucket、候选 `L` 的校准过程。
- `measure_driver.jsonl`：最终测量过程的原始 JSONL。
- `table.csv`：汇总表。
- `baseline_prefilter.png`：prefilter 单独图，含 latency 和 QPS。
- `baseline_graph.png`：graph 单独图，含 latency 和 QPS。
- `baseline_prefilter_vs_graph.png`：prefilter 与 graph 合并对比图。
- `baseline_single_thread_latency.png`：单线程 latency 汇总图。
- `baseline_fixed_l100_ref_like.png`：固定 `graph L=100` 的参考风格图，同时标出 graph recall。

## 参数取值

### 实验控制

- `--rerun-baseline`：删除并重跑本目录已有 baseline 结果。`exp_baseline/start.sh` 已默认传入该参数。
- `--skip-build`：跳过 CMake target 编译。适合确认二进制已经是最新时节省时间。
- `--baseline-query-count N`：baseline 使用的查询数量。默认 `1000`；建议取 `1000` 做快速稳定对比，取 `10000` 做最终报告。

### 数据路径

- `--base-bin PATH`：base 向量文件。默认 `data/sift1m/sift_base.bin`。
- `--query-bin PATH`：query 向量文件。默认 `data/sift1m/sift_query.bin`。
- `--out-dir PATH`：实验套件根目录。`exp_baseline/start.sh` 已固定传入上级 suite 目录，通常不需要手动设置。

### 索引构建

- `--baseline-pq-bytes N`：baseline 直接构建索引的 PQ bytes。默认 `32`。为了对齐正式 SIFT1M PipeANN/ref 图，建议使用 `32`；不要用主动态实验默认的 `16` 来解释 high-selectivity graph latency。
- `--build-r N`：图最大出度 `R`。默认 `64`。
- `--build-l N`：构建搜索列表大小 `L`。默认 `96`。
- `--memory-gb N`：构建内存预算。默认 `64`。
- `--beamwidth N`：搜索 beamwidth。默认 `4`。
- `--nbr-type pq|float`：邻居距离处理类型。默认 `pq`，本实验建议保持默认。

### 搜索与 GT

- `--k N`：召回评估的 top-k。默认 `10`。
- `--metric l2`：距离度量。SIFT1M 默认 `l2`。
- `--gt-numa-node N`：计算 ground truth 时绑定的 NUMA 节点。默认 `1`。
- `--gt-threads N`：计算 ground truth 的线程数。默认 `0`，表示使用 `--gt-numa-node` 上所有逻辑 CPU。

## 固定选择性 bucket

baseline 固定使用以下选择性：

```text
u1e-03, u3e-03, u1e-02, u5e-02, u1e-01, u25, u30, u50, u75, u100
```

分别对应：

```text
0.1%, 0.3%, 1%, 5%, 10%, 25%, 30%, 50%, 75%, 100%
```

## 判定规则

- `prefilter` 和 `graph` 都强制 route，不使用 auto-route。
- 每个点从候选 `L=[10,20,30,40,50,75,100,125,150,200,300,400,600,800,1000,1500,2000]` 中选择第一个满足 `recall@10 >= 98` 的值。
- 如果某条 route 在达到 98% recall 前，平均延迟已经超过 `100 ms`，该点写入 `skipped.jsonl`，图中不画。
- 实验结束会删除临时索引、GT、临时数据和标签，只保留 `json/jsonl/csv/png/md/sh` 等小文件。
