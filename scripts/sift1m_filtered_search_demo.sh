#!/bin/bash
# SIFT1M 数据集 Filtered Search 完整示例
# 演示如何为 SIFT1M 数据集添加随机标签并使用 PipeANN 的 filtered search 功能
# ./sift1m_filtered_search_demo.sh  2>&1 | tee run.log

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 数据集路径
# 可以通过环境变量 SIFT1M_DATA_DIR 覆盖默认值
# 例如: export SIFT1M_DATA_DIR="/path/to/your/data"
DATA_DIR="${SIFT1M_DATA_DIR:-/data/lzg/sift-pipeann/sift1m_pq}"
DATA_FILE="${DATA_DIR}/bigann_1m.bin"           # SIFT1M 数据文件
QUERY_FILE="${DATA_DIR}/bigann_query.bin"       # 查询文件
GT_FILE="${DATA_DIR}/groundtruth_1m.bin"        # Ground truth 文件（无过滤，不使用）
FILTERED_GT_SUBSET="${DATA_DIR}/groundtruth_1m_filtered_subset.bin"      # Subset 过滤 GT
FILTERED_GT_INTERSECT="${DATA_DIR}/groundtruth_1m_filtered_intersect.bin"  # Intersect 过滤 GT
INDEX_DIR="${DATA_DIR}/indices"                 # 索引目录

# 索引参数
INDEX_PREFIX="${INDEX_DIR}/sift1m_filtered"
NUM_THREADS=16
R=64                # 最大出度
L_BUILD=96          # 构建时的候选池大小
PQ_BYTES=32         # PQ 压缩字节数
MEMORY_GB=32        # 构建索引时的内存限制（GB）
METRIC="l2"         # 距离度量：l2/cosine/mips
NBR_TYPE="pq"       # 邻居类型：pq 或 rabitq

# 搜索参数
BEAM_WIDTH=32       # I/O 宽度
K=10                # 返回 top-K 结果
L_SEARCH="20 50 100 200"  # 搜索时的 L 参数列表

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================"
echo "SIFT1M Filtered Search 演示"
echo "========================================"
echo ""
echo "配置信息："
echo "  - 数据目录: ${DATA_DIR}"
echo "  - 索引目录: ${INDEX_DIR}"
echo "  - 线程数: ${NUM_THREADS}"
echo "  - 距离度量: ${METRIC}"
echo ""

# ============================================================================
# 前置检查
# ============================================================================

echo "检查前置条件..."

# 检查数据目录是否存在
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ 错误: 数据目录不存在: ${DATA_DIR}"
    echo "   请创建目录或修改脚本中的 DATA_DIR 变量"
    exit 1
fi

# 检查必需的数据文件
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ 错误: 数据文件不存在: ${DATA_FILE}"
    echo "   请确保 SIFT1M 数据集已下载到正确位置"
    exit 1
fi

if [ ! -f "${QUERY_FILE}" ]; then
    echo "❌ 错误: 查询文件不存在: ${QUERY_FILE}"
    echo "   请确保 SIFT1M 查询文件已下载到正确位置"
    exit 1
fi

# 检查必需的可执行文件
REQUIRED_EXECUTABLES=(
    "build/tests/build_disk_index"
    "build/tests/build_memory_index"
    "build/tests/search_disk_index_filtered"
    "build/tests/utils/compute_groundtruth"
    "build/tests/utils/gen_random_slice"
)

MISSING_EXECUTABLES=()
for exe in "${REQUIRED_EXECUTABLES[@]}"; do
    if [ ! -f "${PROJECT_ROOT}/${exe}" ]; then
        MISSING_EXECUTABLES+=("${exe}")
    fi
done

if [ ${#MISSING_EXECUTABLES[@]} -gt 0 ]; then
    echo "❌ 错误: 以下可执行文件不存在:"
    for exe in "${MISSING_EXECUTABLES[@]}"; do
        echo "   - ${exe}"
    done
    echo ""
    echo "   请先编译项目:"
    echo "   cd ${PROJECT_ROOT} && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# 检查 Python 和标签生成脚本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: python3 未安装"
    echo "   请安装 Python 3: sudo apt-get install python3"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/gen_random_labels.py" ]; then
    echo "❌ 错误: 标签生成脚本不存在: ${SCRIPT_DIR}/gen_random_labels.py"
    exit 1
fi

echo "✓ 所有前置条件满足"
echo ""

# ============================================================================
# 方案 1: 范围标签（Range Labels）- 当前不支持
# ============================================================================

echo "========================================"
echo "方案 1: 范围标签 (Range Labels)"
echo "========================================"
echo ""
echo "⚠️  注意：当前 PipeANN 版本原生只支持 spmat 格式标签"
echo ""
echo "范围标签说明："
echo "  - 每个向量对应一个 uint32_t 值"
echo "  - 使用 RangeSelector 进行范围过滤，例如查询值在 [100, 200] 范围内的向量"
echo "  - 适用场景: 时间戳过滤、价格范围过滤等"
echo ""
echo "如需支持范围标签，需要："
echo "  1. 在 include/filter/label.h 中实现 RangeLabel 类"
echo "  2. 参考 SpmatLabel 的实现方式"
echo "  3. 在 get_label() 函数中添加 'range' 类型支持"
echo ""
echo "标签生成命令示例："
echo "  python ${SCRIPT_DIR}/gen_random_labels.py range \\"
echo "      --output ${DATA_DIR}/data_range_labels.bin \\"
echo "      --num-vectors 1000000 \\"
echo "      --min-value 0 \\"
echo "      --max-value 1000 \\"
echo "      --seed 42"
echo ""
echo "跳过范围标签演示..."
echo ""

# ============================================================================
# 方案 2: 稀疏矩阵标签（Spmat Labels）- 已支持
# ============================================================================

echo "========================================"
echo "方案 2: 稀疏矩阵标签 (Spmat Labels)"
echo "========================================"
echo ""

# 2.1 生成数据的 spmat 标签
echo "[步骤 1/7] 为 SIFT1M 数据集生成 spmat 标签..."
echo "  - 标签空间大小: 100"
echo "  - 每个向量标签数: 8-15 个"

if [ ! -f "${DATA_DIR}/data_labels.spmat" ]; then
    python3 "${SCRIPT_DIR}/gen_random_labels.py" spmat \
        --output "${DATA_DIR}/data_labels.spmat" \
        --num-vectors 1000000 \
        --num-labels 100 \
        --min-labels 8 \
        --max-labels 15 \
        --seed 42
    echo "  ✓ 数据标签生成完成"
else
    echo "  ✓ 数据标签已存在，跳过"
fi

echo ""

# 2.2 生成查询的 spmat 标签（用于子集过滤）
MIN_MATCHES=$((K * 3))  # 确保有足够的候选点，设置为 K * 3
echo "[步骤 2/7] 为查询生成 spmat 标签..."
echo "  - 标签空间大小: 100"
echo "  - 每个查询标签数: 2-5 个"
echo "  - 确保每个查询至少有 ${MIN_MATCHES} 个匹配结果 (K=${K} * 3)"

if [ ! -f "${DATA_DIR}/query_labels.spmat" ]; then
    python3 "${SCRIPT_DIR}/gen_random_labels.py" query-spmat \
        --output "${DATA_DIR}/query_labels.spmat" \
        --num-queries 10000 \
        --num-labels 100 \
        --min-labels 2 \
        --max-labels 5 \
        --seed 123 \
        --data-labels "${DATA_DIR}/data_labels.spmat" \
        --min-matches ${MIN_MATCHES} \
        --selector subset
    echo "  ✓ 查询标签生成完成"
else
    echo "  ✓ 查询标签已存在，跳过"
fi

echo ""

# 2.3 构建带标签的索引
echo "[步骤 3/7] 构建带 spmat 标签的索引..."
echo "  参数: R=${R}, L=${L_BUILD}, PQ=${PQ_BYTES}B, Memory=${MEMORY_GB}GB"
echo "  标签: spmat"
echo ""

mkdir -p "${INDEX_DIR}"

if [ ! -f "${INDEX_PREFIX}_disk.index" ]; then
    "${PROJECT_ROOT}/build/tests/build_disk_index" uint8 \
        "${DATA_FILE}" \
        "${INDEX_PREFIX}" \
        ${R} \
        ${L_BUILD} \
        ${PQ_BYTES} \
        ${MEMORY_GB} \
        ${NUM_THREADS} \
        ${METRIC} \
        ${NBR_TYPE} \
        spmat \
        "${DATA_DIR}/data_labels.spmat"

    echo ""
    echo "✓ 索引构建完成！"
    echo "  索引文件: ${INDEX_PREFIX}_disk.index"
else
    echo "✓ 索引文件已存在，跳过构建"
    echo "  索引文件: ${INDEX_PREFIX}_disk.index"
fi

echo ""

# 2.4 构建内存索引（可选，用于优化入口点）
echo "[步骤 4/7] 构建内存索引（可选，用于优化入口点）..."

if [ ! -f "${INDEX_PREFIX}_mem.index" ]; then
    "${PROJECT_ROOT}/build/tests/utils/gen_random_slice" uint8 \
        "${DATA_FILE}" \
        "${INDEX_PREFIX}_SAMPLE_RATE_0.01" \
        0.01

    "${PROJECT_ROOT}/build/tests/build_memory_index" uint8 \
        "${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin" \
        "${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin" \
        "${INDEX_PREFIX}_mem.index" \
        32 \
        64 \
        1.2 \
        ${NUM_THREADS} \
        ${METRIC}

    echo ""
    echo "✓ 内存索引构建完成！"
else
    echo "✓ 内存索引已存在，跳过"
fi

echo ""

# ============================================================================
# 生成 Filtered Ground Truth
# ============================================================================

echo "========================================"
echo "[步骤 5/7] 生成 Filtered Ground Truth"
echo "========================================"
echo ""
echo "💡 核心概念："
echo "  Filtered search 必须使用 filtered GT 来评估，而不是无过滤的 GT"
echo "  Filtered GT = 在满足过滤条件的数据子集中的真实最近邻"
echo ""
echo "⚠️  注意：生成 filtered GT 需要遍历所有数据点并应用过滤条件，"
echo "   可能需要几分钟时间（取决于数据集大小和过滤选择率）"
echo ""

# 5.1 生成 Subset 过滤的 GT
echo "[步骤 5.1/7] 生成 Subset 过滤的 Ground Truth..."
echo ""
echo "调用 compute_groundtruth 工具："
echo "  命令格式: compute_groundtruth <type> <metric> <base> <query> <K> <output> <tags> <label_type> <selector> <base_labels> <query_labels>"
echo ""
echo "  参数说明："
echo "    - type: uint8              # 数据类型"
echo "    - metric: ${METRIC}        # 距离度量"
echo "    - base: 数据集文件"
echo "    - query: 查询文件"
echo "    - K: ${K}                  # 返回 top-K 结果"
echo "    - output: 输出 GT 文件"
echo "    - tags: null               # 不使用 tags"
echo "    - label_type: spmat        # 标签类型"
echo "    - selector: subset         # 过滤选择器类型"
echo "    - base_labels: 数据标签文件"
echo "    - query_labels: 查询标签文件"
echo ""
echo "  工作原理："
echo "    1. 对每个查询 q，读取其查询标签 query_labels[q]"
echo "    2. 遍历所有数据点 p，检查 data_labels[p] 是否满足过滤条件："
echo "       - Subset: query_labels[q] ⊆ data_labels[p]"
echo "    3. 对满足条件的点计算距离"
echo "    4. 排序并保存 top-K 作为 filtered GT"
echo ""

if [ ! -f "${FILTERED_GT_SUBSET}" ]; then
    echo "⏳ 正在生成 Subset 过滤 GT..."

    "${PROJECT_ROOT}/build/tests/utils/compute_groundtruth" uint8 \
        ${METRIC} \
        "${DATA_FILE}" \
        "${QUERY_FILE}" \
        ${K} \
        "${FILTERED_GT_SUBSET}" \
        null \
        spmat \
        subset \
        "${DATA_DIR}/data_labels.spmat" \
        "${DATA_DIR}/query_labels.spmat"

    echo "✓ Subset 过滤 GT 生成完成: ${FILTERED_GT_SUBSET}"
else
    echo "✓ Subset 过滤 GT 已存在，跳过: ${FILTERED_GT_SUBSET}"
fi

echo ""

# 5.2 生成 Intersection 过滤的 GT
echo "[步骤 5.2/7] 生成 Intersection 过滤的 Ground Truth..."
echo ""
echo "  工作原理："
echo "    1. 对每个查询 q，读取其查询标签 query_labels[q]"
echo "    2. 遍历所有数据点 p，检查 data_labels[p] 是否满足过滤条件："
echo "       - Intersection: query_labels[q] ∩ data_labels[p] ≠ ∅"
echo "    3. 对满足条件的点计算距离"
echo "    4. 排序并保存 top-K 作为 filtered GT"
echo ""

if [ ! -f "${FILTERED_GT_INTERSECT}" ]; then
    echo "⏳ 正在生成 Intersection 过滤 GT..."

    "${PROJECT_ROOT}/build/tests/utils/compute_groundtruth" uint8 \
        ${METRIC} \
        "${DATA_FILE}" \
        "${QUERY_FILE}" \
        ${K} \
        "${FILTERED_GT_INTERSECT}" \
        null \
        spmat \
        intersect \
        "${DATA_DIR}/data_labels.spmat" \
        "${DATA_DIR}/query_labels.spmat"

    echo "✓ Intersection 过滤 GT 生成完成: ${FILTERED_GT_INTERSECT}"
else
    echo "✓ Intersection 过滤 GT 已存在，跳过: ${FILTERED_GT_INTERSECT}"
fi

echo ""
echo "✓ Filtered Ground Truth 生成完成！"
echo ""

# ============================================================================
# 执行 Filtered Search
# ============================================================================

echo "========================================"
echo "[步骤 6/7] 执行 Filtered Search（Subset）"
echo "========================================"
echo ""

# 测试 1: Subset Selector（子集过滤）
echo "----------------------------------------"
echo "测试 1: LabelSubsetSelector（子集过滤）"
echo "----------------------------------------"
echo "说明: 查询标签集合必须是数据标签集合的子集"
echo "      例如: query_labels={1,2} ⊆ data_labels={1,2,3,4} ✓"
echo "           query_labels={1,5} ⊄ data_labels={1,2,3,4} ✗"
echo ""
echo "使用 Ground Truth: ${FILTERED_GT_SUBSET}"
echo ""

"${PROJECT_ROOT}/build/tests/search_disk_index_filtered" uint8 \
    "${INDEX_PREFIX}" \
    ${NUM_THREADS} \
    ${BEAM_WIDTH} \
    "${QUERY_FILE}" \
    "${FILTERED_GT_SUBSET}" \
    ${K} \
    ${METRIC} \
    ${NBR_TYPE} \
    subset \
    "${DATA_DIR}/query_labels.spmat" \
    0 \
    10 \
    ${L_SEARCH}

echo ""
echo ""

# 测试 2: Intersection Selector（交集过滤）
echo "========================================"
echo "[步骤 7/7] 执行 Filtered Search（Intersection）"
echo "========================================"
echo ""
echo "----------------------------------------"
echo "测试 2: LabelIntersectionSelector（交集过滤）"
echo "----------------------------------------"
echo "说明: 查询标签集合与数据标签集合有交集即可"
echo "      例如: query_labels={1,2} ∩ data_labels={2,3,4} = {2} ✓"
echo "           query_labels={1,2} ∩ data_labels={3,4,5} = ∅ ✗"
echo ""
echo "使用 Ground Truth: ${FILTERED_GT_INTERSECT}"
echo ""

"${PROJECT_ROOT}/build/tests/search_disk_index_filtered" uint8 \
    "${INDEX_PREFIX}" \
    ${NUM_THREADS} \
    ${BEAM_WIDTH} \
    "${QUERY_FILE}" \
    "${FILTERED_GT_INTERSECT}" \
    ${K} \
    ${METRIC} \
    ${NBR_TYPE} \
    intersect \
    "${DATA_DIR}/query_labels.spmat" \
    0 \
    10 \
    ${L_SEARCH}

echo ""

# ============================================================================
# 结果说明
# ============================================================================

echo ""
echo "========================================"
echo "结果说明"
echo "========================================"
echo ""
echo "输出列含义："
echo "  - L            : 搜索时的候选池大小（越大召回率越高但延迟也越高）"
echo "  - I/O Width    : I/O 并行度（beam width）"
echo "  - QPS          : 每秒查询数（Queries Per Second）"
echo "  - AvgLat(us)   : 平均延迟（微秒）"
echo "  - P99 Lat      : 99 分位延迟（微秒）"
echo "  - Mean Hops    : 平均图遍历跳数"
echo "  - Mean IOs     : 平均 I/O 次数"
echo "  - Recall@10    : 召回率（百分比形式，0-100）"
echo ""
echo "💡 关键概念："
echo ""
echo "1. Recall 的计算方式："
echo "   - Recall = (找到的GT数 / (查询数 * K)) * 100"
echo "   - 范围: 0-100 (百分比)，不是 0-1"
echo "   - 例如: Recall=95.5 表示平均每个查询找到了 GT top-10 中的 9.55 个结果"
echo ""
echo "2. Filtered GT 的重要性："
echo "   - Unfiltered GT: 全部数据中的真实最近邻"
echo "   - Filtered GT: 满足过滤条件的数据子集中的真实最近邻"
echo "   - ✓ 使用 Filtered GT: Recall 应在 80-99% (取决于 L 参数)"
echo "   - ✗ 使用 Unfiltered GT: Recall 可能只有 1-6% (无意义)"
echo ""
echo "3. 过滤选择器对比："
echo "   - Subset: 更严格，query_labels ⊆ data_labels"
echo "   - Intersection: 更宽松，query_labels ∩ data_labels ≠ ∅"
echo "   - Subset 通常有更少的匹配点，因此 QPS 可能更低但结果更精确"
echo ""
echo "重要提示："
echo "  1. Filtered search 使用后过滤（post-filtering）策略"
echo "  2. 过滤选择率越低（匹配的向量越少），需要越大的 L 参数来保证召回率"
echo "  3. 可以通过调整查询标签的数量和分布来控制过滤选择率"
echo ""
echo "性能优化建议："
echo "  - 如果召回率太低（<80%），增大 L 参数"
echo "  - 如果延迟太高，减小 L 参数或增加线程数"
echo "  - 如果过滤选择率太低（<1%），考虑调整标签分布或使用更宽松的过滤器"
echo ""

# ============================================================================
# 生成的文件清单
# ============================================================================

echo "生成的文件："
echo "  - 数据标签: ${DATA_DIR}/data_labels.spmat"
echo "  - 查询标签: ${DATA_DIR}/query_labels.spmat"
echo "  - Subset Filtered GT: ${FILTERED_GT_SUBSET}"
echo "  - Intersect Filtered GT: ${FILTERED_GT_INTERSECT}"
echo "  - 磁盘索引: ${INDEX_PREFIX}_disk.index (及相关文件)"
echo "  - 内存索引: ${INDEX_PREFIX}_mem.index (及相关文件)"
echo ""
echo "如需清理这些文件，请运行："
echo "  # 清理标签文件"
echo "  rm -f ${DATA_DIR}/*.spmat"
echo ""
echo "  # 清理 Filtered GT"
echo "  rm -f ${DATA_DIR}/groundtruth_1m_filtered_*.bin"
echo ""
echo "  # 清理索引目录（包含所有索引文件）"
echo "  rm -rf ${INDEX_DIR}"
echo ""
echo "  # 或者一键清理所有生成的文件（使用分号确保所有命令都执行）"
echo "  rm -f ${DATA_DIR}/*.spmat ; rm -f ${DATA_DIR}/groundtruth_1m_filtered_*.bin ; rm -rf ${INDEX_DIR}"
echo ""
