#!/bin/bash

# Usage: ./process_bigann.sh <size>
# Example: ./process_bigann.sh 10m
# Supported units: k (thousand), m (million), b (billion)
# Maximum size: 1b (1 billion)
# 从原始的bigann数据集生成pipeann需要的所有文件（包括索引文件、查询文件、 groundtruth文件）

set -e

# Check if size argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <size>"
    echo "Example: $0 10m"
    echo "Supported formats: 10k, 10m, 1b (max 1b)"
    exit 1
fi

SIZE_ARG="$1"

# Parse the size argument
parse_size() {
    local input="$1"
    # Convert to lowercase
    input=$(echo "$input" | tr '[:upper:]' '[:lower:]')

    # Extract number and unit
    if [[ $input =~ ^([0-9]+)([kmb])$ ]]; then
        local num="${BASH_REMATCH[1]}"
        local unit="${BASH_REMATCH[2]}"

        case $unit in
            k)
                echo $((num * 1000))
                ;;
            m)
                echo $((num * 1000000))
                ;;
            b)
                echo $((num * 1000000000))
                ;;
            *)
                echo "Invalid unit: $unit" >&2
                return 1
                ;;
        esac
    else
        echo "Invalid size format: $input" >&2
        echo "Expected format: <number><unit> (e.g., 10m, 1b)" >&2
        return 1
    fi
}

# Parse the size
NUM_VECS=$(parse_size "$SIZE_ARG")
if [ $? -ne 0 ]; then
    exit 1
fi

# Check maximum limit (1 billion)
MAX_VECS=1000000000
if [ $NUM_VECS -gt $MAX_VECS ]; then
    echo "Error: Size $SIZE_ARG exceeds maximum limit of 1b (1 billion)"
    exit 1
fi

echo "Processing BigANN dataset with $NUM_VECS vectors ($SIZE_ARG)"

# Define paths
DATA_DIR="/data"
BIGANN_DIR="${DATA_DIR}/bigann"
BIGANN_QUERY_VECS="${BIGANN_DIR}/bigann_query.bvecs"
BIGANN_BASE_BIN="${BIGANN_DIR}/bigann.bin"

# Create output directory name
OUTPUT_DIR="${DATA_DIR}/sift-pipeann/sift${SIZE_ARG}"
INDICES_DIR="${OUTPUT_DIR}/indices"

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Directory $OUTPUT_DIR already exists."
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo "Removing existing directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDICES_DIR"

# Define output files
QUERY_BIN="${OUTPUT_DIR}/bigann_query.bin"
DATA_SUBSET="${OUTPUT_DIR}/bigann_${SIZE_ARG}.bin"
GROUNDTRUTH="${OUTPUT_DIR}/groundtruth_${SIZE_ARG}.bin"
INDEX_PREFIX="${INDICES_DIR}/${SIZE_ARG}"

# Build directory
BUILD_DIR="/home/dell/PipeANN/build"

# Step 1: Convert query vecs to bin
echo ""
echo "============================================"
echo "Step 1: Converting query .bvecs to .bin"
echo "============================================"
if [ ! -f "$BIGANN_QUERY_VECS" ]; then
    echo "Error: Query file not found at $BIGANN_QUERY_VECS"
    exit 1
fi
${BUILD_DIR}/tests/utils/vecs_to_bin int8 "$BIGANN_QUERY_VECS" "$QUERY_BIN"
echo "Query conversion completed: $QUERY_BIN"

# Step 2: Generate data subset
echo ""
echo "============================================"
echo "Step 2: Generating data subset ($SIZE_ARG)"
echo "============================================"
if [ ! -f "$BIGANN_BASE_BIN" ]; then
    echo "Error: Base data file not found at $BIGANN_BASE_BIN"
    exit 1
fi
${BUILD_DIR}/tests/change_pts uint8 "$BIGANN_BASE_BIN" $NUM_VECS
# change_pts creates a file with suffix of the number
TEMP_SUBSET="${BIGANN_BASE_BIN}${NUM_VECS}"
if [ ! -f "$TEMP_SUBSET" ]; then
    echo "Error: Subset file not created at $TEMP_SUBSET"
    exit 1
fi
mv "$TEMP_SUBSET" "$DATA_SUBSET"
echo "Data subset created: $DATA_SUBSET"

# Step 3: Compute groundtruth
echo ""
echo "============================================"
echo "Step 3: Computing groundtruth"
echo "============================================"
${BUILD_DIR}/tests/utils/compute_groundtruth uint8 "$DATA_SUBSET" "$QUERY_BIN" 100 "$GROUNDTRUTH"
echo "Groundtruth computed: $GROUNDTRUTH"

# Step 4: Build disk index
echo ""
echo "============================================"
echo "Step 4: Building disk index"
echo "============================================"
# Parameters based on README:
# R=96, L=128, bytes_per_nbr=32, M=64 (GB), T=32, metric=l2, nbr_type=pq
# For different scales, we may adjust R and L
R=96
L=128
if [ $NUM_VECS -ge 1000000000 ]; then
    # For 1B scale
    R=128
    L=200
fi

MAX_MEM_GB=64
NUM_THREADS=32
BYTES_PER_NBR=32

echo "Building disk index with R=$R, L=$L, bytes_per_nbr=$BYTES_PER_NBR, max_mem=${MAX_MEM_GB}GB, threads=$NUM_THREADS"
${BUILD_DIR}/tests/build_disk_index uint8 "$DATA_SUBSET" "$INDEX_PREFIX" $R $L $BYTES_PER_NBR $MAX_MEM_GB $NUM_THREADS l2 pq
echo "Disk index built with prefix: $INDEX_PREFIX"

# Step 5: Build memory index
echo ""
echo "============================================"
echo "Step 5: Building in-memory index"
echo "============================================"
# Generate random sample (1% of data)
SAMPLE_PREFIX="${INDEX_PREFIX}_SAMPLE_RATE_0.01"
${BUILD_DIR}/tests/utils/gen_random_slice uint8 "$DATA_SUBSET" "$SAMPLE_PREFIX" 0.01
echo "Random sample generated"

# Build memory index with R=32, L=64
${BUILD_DIR}/tests/build_memory_index uint8 "${SAMPLE_PREFIX}_data.bin" "${SAMPLE_PREFIX}_ids.bin" "${INDEX_PREFIX}_mem.index" 0 0 32 64 1.2 24 l2
echo "Memory index built: ${INDEX_PREFIX}_mem.index"

# Summary
echo ""
echo "============================================"
echo "Processing completed successfully!"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo "Generated files:"
echo "  - Query: $QUERY_BIN"
echo "  - Data subset: $DATA_SUBSET"
echo "  - Groundtruth: $GROUNDTRUTH"
echo "  - Disk index: ${INDEX_PREFIX}_disk.index"
echo "  - Memory index: ${INDEX_PREFIX}_mem.index"
echo ""
echo "You can now search using:"
echo "${BUILD_DIR}/tests/search_disk_index uint8 ${INDEX_PREFIX} 1 32 ${QUERY_BIN} ${GROUNDTRUTH} 10 l2 pq 2 10 10 20 30 40"
