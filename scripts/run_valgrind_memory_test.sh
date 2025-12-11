#!/bin/bash

#==============================================
# PipeANN 单查询内存测试脚本
#==============================================
#
# 功能: 对单个查询向量执行一次搜索，使用 Valgrind 监控内存占用
#
# 使用方法:
#   1. 修改下面的配置参数
#   2. 运行脚本: ./scripts/run_valgrind_memory_test.sh
#
# 输出:
#   - valgrind_results/memory_test_<timestamp>_massif.out
#   - valgrind_results/memory_test_<timestamp>_massif_report.txt
#   - valgrind_results/memory_test_<timestamp>_memcheck.log (可选)
#
#==============================================

#==============================================
# 配置参数 - 在这里修改你的测试参数
#==============================================

# 数据目录（可通过环境变量 SIFT1M_DATA_DIR 覆盖）
DATA_DIR="${SIFT1M_DATA_DIR:-/data/lzg/sift-pipeann/sift1m_pq}"

# 可执行文件路径（相对于项目根目录）
EXECUTABLE="./build/tests/single_query_memory_test"

# 索引类型 (float/int8/uint8)
INDEX_TYPE="uint8"

# 索引文件前缀路径
INDEX_PREFIX="${DATA_DIR}/indices/sift1m_filtered"

# 查询文件路径
QUERY_FILE="${DATA_DIR}/bigann_query.bin"

# 返回的最近邻数量
K=10

# 搜索参数 L
L=100

# 距离度量 (l2/cosine/mips)
METRIC="l2"

# 邻居类型 (pq/rabitq)
NBR_TYPE="pq"

#==============================================

# 固定参数
NUM_THREADS=1

# 输出目录
OUTPUT_DIR="./valgrind_results"
mkdir -p "$OUTPUT_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${OUTPUT_DIR}/memory_test_${TIMESTAMP}"

echo "========================================"
echo "PipeANN Valgrind 单查询内存测试"
echo "========================================"
echo "说明: 此脚本对单个查询向量执行一次搜索"
echo "      并使用 Valgrind 监控内存使用情况"
echo ""
echo "测试配置:"
echo "  可执行文件: $EXECUTABLE"
echo "  索引类型:   $INDEX_TYPE"
echo "  索引前缀:   $INDEX_PREFIX"
echo "  查询文件:   $QUERY_FILE"
echo "  K:          $K"
echo "  L:          $L"
echo "  度量:       $METRIC"
echo "  邻居类型:   $NBR_TYPE"
echo "  线程数:     $NUM_THREADS"
echo "========================================"
echo ""

# 检查可执行文件
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ 错误: 可执行文件不存在: $EXECUTABLE"
    echo "   请先编译项目: cmake -B build && cmake --build build --target single_query_memory_test"
    exit 1
fi

# 检查查询文件
if [ ! -f "$QUERY_FILE" ]; then
    echo "❌ 错误: 查询文件不存在: $QUERY_FILE"
    echo "   请检查 DATA_DIR 配置或设置环境变量: export SIFT1M_DATA_DIR=/your/data/path"
    exit 1
fi

# 检查索引文件
if [ ! -f "${INDEX_PREFIX}_disk.index" ]; then
    echo "❌ 错误: 索引文件不存在: ${INDEX_PREFIX}_disk.index"
    echo "   请先构建索引"
    exit 1
fi

echo "✓ 文件检查通过"
echo ""

# 运行 Valgrind Massif (内存分析器)
echo "[1/3] 运行 Valgrind Massif (堆内存分析)..."
echo "输出文件: ${OUTPUT_PREFIX}_massif.out"
valgrind --tool=massif \
    --massif-out-file="${OUTPUT_PREFIX}_massif.out" \
    --time-unit=ms \
    --detailed-freq=1 \
    --max-snapshots=100 \
    --threshold=0.1 \
    "$EXECUTABLE" "$INDEX_TYPE" "$INDEX_PREFIX" "$NUM_THREADS" \
    "$QUERY_FILE" "$K" "$L" "$METRIC" "$NBR_TYPE"

if [ $? -ne 0 ]; then
    echo "错误: Massif 运行失败"
    exit 1
fi

# 生成 Massif 报告
echo ""
echo "[2/3] 生成 Massif 内存报告..."
ms_print "${OUTPUT_PREFIX}_massif.out" > "${OUTPUT_PREFIX}_massif_report.txt"
echo "报告保存到: ${OUTPUT_PREFIX}_massif_report.txt"

# 显示峰值内存使用
echo ""
echo "[3/3] 峰值内存使用摘要:"
echo "========================================"
grep -A 20 "peak" "${OUTPUT_PREFIX}_massif_report.txt" | head -25
echo "========================================"

# 运行 Valgrind Memcheck (内存错误检查)
# 如需运行 memcheck，取消注释以下代码：
# echo ""
# echo "运行 Valgrind Memcheck (内存错误检查)..."
# echo "输出文件: ${OUTPUT_PREFIX}_memcheck.log"
# valgrind --tool=memcheck \
#     --leak-check=full \
#     --show-leak-kinds=all \
#     --track-origins=yes \
#     --verbose \
#     --log-file="${OUTPUT_PREFIX}_memcheck.log" \
#     "$EXECUTABLE" "$INDEX_TYPE" "$INDEX_PREFIX" "$NUM_THREADS" \
#     "$QUERY_FILE" "$K" "$L" "$METRIC" "$NBR_TYPE"
# echo "Memcheck 报告保存到: ${OUTPUT_PREFIX}_memcheck.log"

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"
echo "结果文件:"
echo "  - Massif 数据:  ${OUTPUT_PREFIX}_massif.out"
echo "  - Massif 报告:  ${OUTPUT_PREFIX}_massif_report.txt"
if [ -f "${OUTPUT_PREFIX}_memcheck.log" ]; then
    echo "  - Memcheck 日志: ${OUTPUT_PREFIX}_memcheck.log"
fi
echo ""
echo "查看详细报告:"
echo "  ms_print ${OUTPUT_PREFIX}_massif.out | less"
echo "  cat ${OUTPUT_PREFIX}_massif_report.txt"
echo "========================================"