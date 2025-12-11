#!/bin/bash
# 运行 compute_groundtruth 并监控内存（RSS）
# 用法: ./run_compute_gt_with_rss.sh [选择性百分比]

set -e

SELECTIVITY=${1:-60}

DATA_DIR=${SIFT1M_DATA_DIR:-/data/lzg/sift-pipeann/sift1m_pq}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "运行 compute_groundtruth"
echo "选择性: ${SELECTIVITY}%"
echo "========================================"
echo ""

# 检查可执行文件
if [ ! -f "${PROJECT_ROOT}/build/tests/utils/compute_groundtruth" ]; then
    echo "错误: compute_groundtruth 未编译"
    echo "请先运行: make build/tests/utils/compute_groundtruth"
    exit 1
fi

# 检查数据文件
if [ ! -f "${DATA_DIR}/bigann_1m.bin" ]; then
    echo "错误: 数据文件不存在: ${DATA_DIR}/bigann_1m.bin"
    exit 1
fi

if [ ! -f "${DATA_DIR}/data_labels.spmat" ]; then
    echo "错误: 标签文件不存在: ${DATA_DIR}/data_labels.spmat"
    exit 1
fi

# 生成查询标签
echo "[1/2] 生成查询标签 (label ${SELECTIVITY})..."
python3 "${PROJECT_ROOT}/scripts/gen_random_labels.py" query-labels \
    --output "${DATA_DIR}/query_labels_${SELECTIVITY}.spmat" \
    --num-queries 10000 \
    --num-labels 100 \
    --label-id $((SELECTIVITY - 1))

echo ""
echo "[2/2] 运行 compute_groundtruth..."
echo ""

# 运行 compute_groundtruth
"${PROJECT_ROOT}/build/tests/utils/compute_groundtruth" uint8 \
    l2 \
    "${DATA_DIR}/bigann_1m.bin" \
    "${DATA_DIR}/bigann_query.bin" \
    10 \
    "${DATA_DIR}/groundtruth_1m_filtered_label_${SELECTIVITY}.bin" \
    "null" \
    "spmat" \
    "subset" \
    "${DATA_DIR}/data_labels.spmat" \
    "${DATA_DIR}/query_labels_${SELECTIVITY}.spmat" \
    2>&1 | tee "compute_gt_rss_${SELECTIVITY}.log"

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo "日志文件: compute_gt_rss_${SELECTIVITY}.log"
echo ""
echo "查看 RSS 内存变化："
echo "  grep '\\[RSS\\]' compute_gt_rss_${SELECTIVITY}.log"
echo ""
echo "查找峰值内存："
echo "  grep '\\[RSS\\]' compute_gt_rss_${SELECTIVITY}.log | awk '{print \$NF}' | sort -n | tail -1"
