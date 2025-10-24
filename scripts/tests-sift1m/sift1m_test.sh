#!/bin/bash

echo "This script tests PipeANN performance on SIFT1M dataset."

# 创建数据输出目录
mkdir -p ./data

# run top-10 evaluation for SIFT1M dataset.
function run_10_1M() {
    NTHREADS=$5
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5: num threads, $6: SIFT1M index file prefix
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $5 SIFT1M" 
    /home/dell/PipeANN/build/tests/search_disk_index uint8 $6 $NTHREADS $2 /data/sift-pipeann/sift1m/bigann_query.bin /data/sift-pipeann/sift1m/groundtruth_1m.bin $1 l2 pq $3 $4 25 30 35 40 50
}

echo "Run PipeANN on SIFT1M..."
# 参数说明：
# 10: top-K (recall@10)
# 32: I/O pipeline width (PipeANN使用32)
# 2: search mode (2 = pipe search)
# 10: mem_L (PipeANN使用10)
# 1: num threads
# /data/sift1M/indices/1m: SIFT1M索引前缀路径
run_10_1M 10 32 2 0 1 /data/sift-pipeann/sift1m/indices/1m | tee ./data/sift-pipeann/sift1m/sift1m_pipeann_results.txt

echo "Test completed. Results saved to ./data/sift1m_pipeann_results.txt"