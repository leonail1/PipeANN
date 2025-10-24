#!/bin/bash

# Usage: ./generate_incremental_groundtruth.sh <data_bin> <query_bin> <index_npts> <tot_npts> <batch_npts> <target_topk> <target_dir>
# Example: ./generate_incremental_groundtruth.sh /data/sift1m/bigann_1m.bin /data/sift1m/bigann_query.bin 1000000 1000000 10000 10 /data/sift1m/indices_upd/1m_topk
#
# This script generates groundtruth files for incremental insertion testing.
# For a test starting with index_npts vectors and inserting tot_npts-index_npts vectors in batches:
# - batch 0: gt_0.bin (for vectors [0, index_npts))
# - batch 1: gt_<batch_npts>.bin (for vectors [0, index_npts+batch_npts))
# - batch i: gt_<i*batch_npts>.bin (for vectors [0, index_npts+i*batch_npts))

set -e

# Check arguments
if [ $# -ne 7 ]; then
    echo "Usage: $0 <data_bin> <query_bin> <index_npts> <tot_npts> <batch_npts> <target_topk> <target_dir>"
    echo "Example: $0 /data/sift1m/bigann_1m.bin /data/sift1m/bigann_query.bin 1000000 1000000 10000 10 /data/sift1m/indices_upd/1m_topk"
    exit 1
fi

DATA_BIN="$1"
QUERY_BIN="$2"
INDEX_NPTS="$3"
TOT_NPTS="$4"
BATCH_NPTS="$5"
TARGET_TOPK="$6"
TARGET_DIR="$7"

# Validate files exist
if [ ! -f "$DATA_BIN" ]; then
    echo "Error: Data file not found: $DATA_BIN"
    exit 1
fi

if [ ! -f "$QUERY_BIN" ]; then
    echo "Error: Query file not found: $QUERY_BIN"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Build directory
BUILD_DIR="/home/dell/PipeANN/build"
COMPUTE_GT="${BUILD_DIR}/tests/utils/compute_groundtruth"

if [ ! -f "$COMPUTE_GT" ]; then
    echo "Error: compute_groundtruth not found at $COMPUTE_GT"
    exit 1
fi

# Determine data type from file extension and size
# Read first 8 bytes to get npts and ndims
read -r NPTS NDIMS < <(od -An -t u4 -N 8 "$DATA_BIN" | xargs)
FILE_SIZE=$(stat -c%s "$DATA_BIN")
HEADER_SIZE=8
DATA_SIZE=$((FILE_SIZE - HEADER_SIZE))
ELEMENT_SIZE=$((DATA_SIZE / NPTS / NDIMS))

if [ $ELEMENT_SIZE -eq 1 ]; then
    DATA_TYPE="uint8"
elif [ $ELEMENT_SIZE -eq 4 ]; then
    DATA_TYPE="float"
else
    echo "Error: Cannot determine data type (element size = $ELEMENT_SIZE)"
    exit 1
fi

echo "============================================"
echo "Generating incremental groundtruth files"
echo "============================================"
echo "Data file: $DATA_BIN"
echo "Query file: $QUERY_BIN"
echo "Data type: $DATA_TYPE"
echo "Data points: $NPTS, Dimensions: $NDIMS"
echo "Index size: $INDEX_NPTS"
echo "Total vectors: $TOT_NPTS"
echo "Batch size: $BATCH_NPTS"
echo "Target K: $TARGET_TOPK"
echo "Output directory: $TARGET_DIR"
echo ""

# Calculate number of batches
NUM_BATCHES=$(( (TOT_NPTS - INDEX_NPTS + BATCH_NPTS - 1) / BATCH_NPTS ))

if [ $INDEX_NPTS -gt $TOT_NPTS ]; then
    echo "Error: index_npts ($INDEX_NPTS) > tot_npts ($TOT_NPTS)"
    exit 1
fi

if [ $TOT_NPTS -gt $NPTS ]; then
    echo "Error: tot_npts ($TOT_NPTS) > available data points ($NPTS)"
    exit 1
fi

echo "Will generate $((NUM_BATCHES + 1)) groundtruth files (batch 0 to $NUM_BATCHES)"
echo ""

# Temporary directory for intermediate files
TEMP_DIR="${TARGET_DIR}/temp"
mkdir -p "$TEMP_DIR"

# Function to create a subset of the data file
create_subset() {
    local npts=$1
    local output=$2

    echo "Creating subset with $npts points..."
    ${BUILD_DIR}/tests/change_pts $DATA_TYPE "$DATA_BIN" $npts

    # change_pts creates a file with suffix of the number
    local temp_file="${DATA_BIN}${npts}"
    if [ ! -f "$temp_file" ]; then
        echo "Error: Subset file not created at $temp_file"
        return 1
    fi

    mv "$temp_file" "$output"
    echo "Created: $output"
}

# Generate groundtruth for each batch
for ((i=0; i<=NUM_BATCHES; i++)); do
    # Calculate current number of vectors in the index
    if [ $i -eq 0 ]; then
        CURRENT_NPTS=$INDEX_NPTS
        OFFSET=0
    else
        CURRENT_NPTS=$((INDEX_NPTS + i * BATCH_NPTS))
        OFFSET=$((i * BATCH_NPTS))

        # Don't exceed tot_npts
        if [ $CURRENT_NPTS -gt $TOT_NPTS ]; then
            CURRENT_NPTS=$TOT_NPTS
        fi
    fi

    GT_FILE="${TARGET_DIR}/gt_${OFFSET}.bin"

    echo "============================================"
    echo "Batch $i: Generating groundtruth for $CURRENT_NPTS vectors"
    echo "Output: $GT_FILE"
    echo "============================================"

    # Create subset file if not the full dataset
    if [ $CURRENT_NPTS -eq $NPTS ]; then
        SUBSET_FILE="$DATA_BIN"
    else
        SUBSET_FILE="${TEMP_DIR}/subset_${CURRENT_NPTS}.bin"
        create_subset $CURRENT_NPTS "$SUBSET_FILE"
    fi

    # Compute groundtruth
    echo "Computing groundtruth (K=$TARGET_TOPK)..."
    $COMPUTE_GT $DATA_TYPE "$SUBSET_FILE" "$QUERY_BIN" $TARGET_TOPK "$GT_FILE"

    echo "Generated: $GT_FILE"
    echo ""

    # Clean up subset file if created
    if [ "$SUBSET_FILE" != "$DATA_BIN" ]; then
        rm -f "$SUBSET_FILE"
    fi

    # Stop if we've reached tot_npts
    if [ $CURRENT_NPTS -ge $TOT_NPTS ]; then
        break
    fi
done

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo "============================================"
echo "Groundtruth generation completed!"
echo "============================================"
echo "Generated files in: $TARGET_DIR"
ls -lh "$TARGET_DIR"/gt_*.bin
echo ""
echo "You can now run test_insert_search with:"
echo "  truthset_prefix=$TARGET_DIR"
