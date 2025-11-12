#!/bin/bash

echo "This script tests PipeANN performance on SIFT1M dataset."

# 参数解析
COMPRESSION_TYPE=${1:-pq}  # 默认使用 pq

if [[ "$COMPRESSION_TYPE" != "pq" && "$COMPRESSION_TYPE" != "rabitq" ]]; then
    echo "Error: Invalid compression type '$COMPRESSION_TYPE'. Use 'pq' or 'rabitq'."
    echo "Usage: $0 [pq|rabitq]"
    exit 1
fi

echo "Using compression type: $COMPRESSION_TYPE"

# 根据压缩类型设置路径
if [[ "$COMPRESSION_TYPE" == "pq" ]]; then
    INDEX_PREFIX="/data/lzg/sift-pipeann/sift1m_pq/indices/1m"
    QUERY_BIN="/data/lzg/sift-pipeann/sift1m_pq/bigann_query.bin"
    GROUNDTRUTH_BIN="/data/lzg/sift-pipeann/sift1m_pq/groundtruth_1m.bin"
else
    INDEX_PREFIX="/data/lzg/sift-pipeann/sift1m_rabitq/indices/1m"
    QUERY_BIN="/data/lzg/sift-pipeann/sift1m_rabitq/bigann_query.bin"
    GROUNDTRUTH_BIN="/data/lzg/sift-pipeann/sift1m_rabitq/groundtruth_1m.bin"
fi

# 基本存在性检查，避免读取到错误文件导致维度解析异常
if [[ ! -f "$QUERY_BIN" ]]; then
    echo "Error: Query BIN not found: $QUERY_BIN"
    exit 1
fi
if [[ ! -f "$GROUNDTRUTH_BIN" ]]; then
    echo "Warning: Groundtruth BIN not found: $GROUNDTRUTH_BIN (recall 计算将不可用)"
fi

# 创建数据输出目录
mkdir -p ./data

# run top-10 evaluation for SIFT1M dataset.
function run_10_1M() {
    NTHREADS=$5
    COMPRESSION=$7
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5: num threads, $6: SIFT1M index file prefix, $7: compression type (pq or rabitq)

    # 根据压缩类型选择不同的 L 值列表
    if [[ "$COMPRESSION" == "pq" ]]; then
        L_VALUES="25 30 35 40 50"
    else
        L_VALUES="10 500 550 600 650 700"
    fi

    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $5 COMPRESSION $COMPRESSION SIFT1M"
    ~/PipeANN/build/tests/search_disk_index uint8 $6 $NTHREADS $2 "$QUERY_BIN" "$GROUNDTRUTH_BIN" $1 l2 $COMPRESSION $3 $4 $L_VALUES
}

echo "Run PipeANN on SIFT1M with $COMPRESSION_TYPE..."
# 参数说明：
# 10: top-K (recall@10)
# 32: I/O pipeline width (PipeANN使用32)
# 2: search mode (2 = pipe search)
# 10: mem_L (10 for PipeANN)
# 1: num threads
# $INDEX_PREFIX: SIFT1M索引前缀路径
# $COMPRESSION_TYPE: 压缩类型 (pq 或 rabitq)
run_10_1M 10 32 2 0 1 "$INDEX_PREFIX" "$COMPRESSION_TYPE" | tee "./data/sift1m_${COMPRESSION_TYPE}_results.txt"

echo "Test completed. Results saved to ./data/sift1m_${COMPRESSION_TYPE}_results.txt"