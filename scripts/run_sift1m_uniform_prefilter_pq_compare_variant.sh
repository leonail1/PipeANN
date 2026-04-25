#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <prefilter-pq-bits: 8|16|32> [output-dir]" >&2
  exit 1
fi

PREFILTER_PQ_BITS="$1"
case "$PREFILTER_PQ_BITS" in
  8|16|32)
    ;;
  *)
    echo "prefilter-pq-bits must be one of: 8, 16, 32" >&2
    exit 1
    ;;
esac

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
OUTPUT_DIR="${2:-experiments/sift1m_uniform_prefilter_pq_compare}"
WORKLOAD_DIR="$OUTPUT_DIR/workloads"
SUMMARY_JSON="$WORKLOAD_DIR/uniform_exact_selectivity_summary.json"
QUERY_BIN="data/sift1m/sift_query.bin"
SOURCE_PREFIX="data/sift1m/sift1m_pipeann_uniform_pq${PREFILTER_PQ_BITS}"
DEST_PREFIX="$OUTPUT_DIR/runtime_indexes/sift1m_uniform_prefilter_compare_pq${PREFILTER_PQ_BITS}"
RUN_DIR="$OUTPUT_DIR/results/pq${PREFILTER_PQ_BITS}"

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "missing workload summary: $SUMMARY_JSON" >&2
  exit 1
fi

mapfile -t BUCKET_NAMES < <(
  "$PYTHON_BIN" - <<'PY' "$SUMMARY_JSON"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as reader:
    payload = json.load(reader)

for workload in payload["workloads"]:
    print(workload["bucket_name"])
PY
)

MAX_SELECTIVITY="$(
  "$PYTHON_BIN" - <<'PY' "$SUMMARY_JSON"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as reader:
    payload = json.load(reader)

print(max(float(workload["selectivity"]) for workload in payload["workloads"]))
PY
)"

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py prepare-index-prefix-for-labels \
  --source-prefix "$SOURCE_PREFIX" \
  --dest-prefix "$DEST_PREFIX" \
  --label-file "$WORKLOAD_DIR/base.uniform_exact_selectivity.spmat" \
  --summary-json "$RUN_DIR/index_runtime.json" \
  > "$RUN_DIR/prepare_index.log" 2>&1

THRESHOLD_CMD=(
  "$ROOT_DIR/build/tests/calibrate_hybrid_threshold"
  float
  "$DEST_PREFIX"
  1
  4
  10
  l2
  pq
  0
  100
)
for bucket_name in "${BUCKET_NAMES[@]}"; do
  THRESHOLD_CMD+=(intersect "$QUERY_BIN" "$WORKLOAD_DIR/$bucket_name/queries.spmat" 200)
done
"${THRESHOLD_CMD[@]}" > "$RUN_DIR/calibrate_threshold.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$DEST_PREFIX" \
  --index-type float \
  --selector-type intersect \
  --manifest "$RUN_DIR/manifest.json" \
  > "$RUN_DIR/build_manifest.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py calibrate-rerank \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$DEST_PREFIX" \
  --out-dir "$RUN_DIR/calibration" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --search-l 100 \
  --target-recall 98 \
  --max-selectivity "$MAX_SELECTIVITY" \
  > "$RUN_DIR/calibrate_rerank.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
  --manifest "$RUN_DIR/manifest.json" \
  --out-dir "$RUN_DIR/run" \
  --dataset-name "sift1m_uniform_prefilter_pq${PREFILTER_PQ_BITS}_compare" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes prefilter \
  --l-values 100 \
  --prefilter-rerank-json "$RUN_DIR/calibration/prefilter_rerank_calibration.json" \
  > "$RUN_DIR/run.log" 2>&1

echo "[ok] finished PQ${PREFILTER_PQ_BITS} compare variant"
