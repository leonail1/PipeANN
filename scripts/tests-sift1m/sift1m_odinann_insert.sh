#!/bin/bash

echo "This script tests OdinANN insert performance on sift1m dataset."
echo "Starting from 100k index, incrementally inserting to 1M"

# 清理之前的索引文件
rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_shadow*
rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_merge*
rm -f /data/lzg/sift-pipeann/sift100k/indices/100ktemp0*
rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_mem*

# 创建数据输出目录
mkdir -p ./data-insert-search

CWD=$(pwd)

# 设置环境变量 - 使用100k索引
INDEX_PREFIX="/data/lzg/sift-pipeann/sift100k/indices/100k"

# 构建内存索引 (从100k索引的1%采样)
echo "Building memory index from 100k dataset..."
${CWD}/build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2 24 l2

echo "Starting OdinANN insert test: 100k -> 1M"
echo "Data file: /data/lzg/sift-pipeann/sift1m_pq/bigann_1m.bin (1M vectors, will use 100k-1M range)"
echo "Query file: /data/lzg/sift-pipeann/sift1m_pq/bigann_query.bin"
echo "Initial index: /data/lzg/sift-pipeann/sift100k/indices/100k (100k vectors)"
echo "Ground truth folder: /data/lzg/sift-pipeann/indices_upd/5m_topk"

# 运行OdinANN插入测试
# 用法示例：
# build/tests/test_insert_search <data_type> <data_file> <L_disk> <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode> <index_prefix> <query_file> <groundtruth_file> <truthset_l_offset> <recall_at> <beam_width> <search_beam_width> <mem_L> <L_search_values...>
#
# 参数说明：
# <data_type>: 数据类型 uint8
# <data_file>: 数据文件路径 (包含1M向量,索引0-999999)
# <L_disk>: L_disk参数 128
# <vecs_per_step>: 每步插入的向量数 100000 (100k)
# <num_steps>: 总步数 9 (100k->200k->...->1M, 共9步)
# <insert_threads>: 插入线程数 4
# <search_threads>: 搜索线程数 16
# <search_mode>: 搜索模式 0 (beam search for OdinANN)
# <index_prefix>: 索引文件前缀 (100k初始索引)
# <query_file>: 查询文件路径
# <groundtruth_file>: 真值文件夹路径 (包含gt_0.bin, gt_100000.bin, ..., gt_900000.bin)
# <truthset_l_offset>: 真值偏移量 0 (groundtruth文件名是gt_100000.bin表示索引已经插入到100k+100000=200k)
# <recall_at>: 计算recall@K的K值 10
# <beam_width>: 插入时的beam宽度 4
# <search_beam_width>: 搜索时的beam宽度 4
# <mem_L>: 内存索引L参数 10 (从内存索引获取10个起始搜索点)
# <L_search_values...>: 不同的L搜索参数列表
#
# 关键逻辑说明:
# - 初始磁盘索引包含100k个向量 (索引0-99999)
# - get_trace函数在每个batch i时:
#   - 删除向量: 索引 i*100k 到 (i+1)*100k-1  (这些向量在初始索引中不存在,所以是空操作)
#   - 插入向量: 从数据文件读取索引 (i*100k + 100k) 到 (i+1)*100k + 100k - 1
# - Batch 0: 插入索引 100000-199999 (第2个100k)
# - Batch 1: 插入索引 200000-299999 (第3个100k)
# - ...
# - Batch 8: 插入索引 900000-999999 (第10个100k)
# - 总共9个batch,从100k增长到1M
~/PipeANN/build/tests/test_insert_search uint8 /data/lzg/sift-pipeann/sift1m_pq/bigann_1m.bin 128 100000 10 1 1 0 ${INDEX_PREFIX} /data/lzg/sift-pipeann/sift1m_pq/bigann_query.bin /data/lzg/sift-pipeann/indices_upd/5m_topk 0 10 4 4 10 20 25 30 40 50 |& tee $CWD/data-insert-search/OdinANN-insert-sift1m.txt

echo "OdinANN sift1m insert test completed."
echo "Results saved to: $CWD/data-insert-search/OdinANN-insert-sift1m.txt"

# 清理测试后的临时文件（保留内存索引文件）
rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_shadow*
rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_merge*
# rm -f /data/lzg/sift-pipeann/sift100k/indices/100k_mem*  # 保留内存索引文件，避免下次重新构建
rm -f /data/lzg/sift-pipeann/sift100k/indices/100ktemp0*

echo "Cleanup completed."