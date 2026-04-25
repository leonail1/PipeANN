# Experiments

## SIFT1M uniform final mixed-PQ run

本目录中的最终 SIFT1M uniform 实验使用混合 PQ 配置：

1. prefilter 路径使用 `data/sift1m/sift1m_pipeann_uniform_pq32`
2. graph 路径使用 `data/sift1m/sift1m_pipeann_uniform_pq16`
3. 多查询性能统一使用单线程 `threads=1`
4. RSS 统一使用单查询 `probe_query.bin` 和 `probe_query.spmat`
5. `k=10`
6. `beamwidth=4`
7. `mem_L=0`
8. `L=100`
9. `similarity=l2`
10. `nbr_type=pq`

bucket 分配如下：

1. prefilter + PQ32: `u1e-05`, `u3e-05`, `u1e-04`, `u3e-04`, `u1e-03`, `u3e-03`, `u1e-02`, `u1e-01`
2. graph + PQ16: `u50`, `u75`, `u100`

实验入口脚本：

1. `scripts/run_sift1m_uniform_final_mixed_pq.sh`

脚本行为：

1. 删除旧的 `experiments/sift1m_uniform_pq32`
2. 删除旧的 `experiments/sift1m_uniform_pq16`
3. 删除旧的 `experiments/sift1m_uniform_pq8`
4. 重建 `experiments/sift1m_uniform_final_mixed_pq`
5. 为 PQ32 生成 manifest，并只保留低选择性 prefilter bucket
6. 重新校准 PQ32 prefilter rerank，使 `recall@10 >= 98%`
7. 跑 PQ32 单线程 prefilter 多查询性能，并对每个 bucket 额外测一次单查询 RSS
8. 为 PQ16 生成 manifest，并只保留高选择性 graph bucket
9. 跑 PQ16 单线程 graph 多查询性能，并对每个 bucket 额外测一次单查询 RSS
10. 合并两部分结果为单一路由 `mixed`
11. 输出最终图

最终产物路径：

1. `experiments/sift1m_uniform_final_mixed_pq/calibration_prefilter_pq32/prefilter_rerank_calibration.json`
2. `experiments/sift1m_uniform_final_mixed_pq/prefilter_pq32_run/results.jsonl`
3. `experiments/sift1m_uniform_final_mixed_pq/graph_pq16_run/results.jsonl`
4. `experiments/sift1m_uniform_final_mixed_pq/results.jsonl`
5. `experiments/sift1m_uniform_final_mixed_pq/sift1m_uniform_final_mixed_pq_l100.png`

说明：

1. `prefilter_pq32_run/results.jsonl` 和 `graph_pq16_run/results.jsonl` 保留原始 route
2. `results.jsonl` 会把最终选中的记录统一改写为 `route=mixed`，用于最终汇总和出图