#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
OUTPUT_DIR="${1:-experiments/sift1m_uniform_prefilter_pq_compare}"
WORKLOAD_DIR="$OUTPUT_DIR/workloads"
RESULTS_DIR="$OUTPUT_DIR/results"
RUNTIME_INDEX_DIR="$OUTPUT_DIR/runtime_indexes"
SUMMARY_JSON="$WORKLOAD_DIR/uniform_exact_selectivity_summary.json"
QUERY_BIN="data/sift1m/sift_query.bin"
LEGACY_SINGLE_PLOTS=(
  "experiments/sift1m_uniform_final_mixed_pq/sift1m_uniform_final_mixed_pq_l100.png"
  "experiments/sift1m_uniform_final_mixed_prefilter_pq8/sift1m_uniform_final_mixed_prefilter_pq8_l100.png"
  "experiments/sift1m_uniform_final_mixed_prefilter_pq16/sift1m_uniform_final_mixed_prefilter_pq16_l100.png"
)
LEGACY_EXPERIMENT_DIRS=(
  "experiments/sift1m_uniform_final_mixed_pq"
  "experiments/sift1m_uniform_final_mixed_prefilter_pq8"
  "experiments/sift1m_uniform_final_mixed_prefilter_pq16"
)
SELECTIVITY_SPECS=(
  u1e-03:0.001
  u3e-03:0.003
  u1e-02:0.01
  u5e-02:0.05
  u1e-01:0.1
  u25:0.25
  u30:0.3
  u50:0.5
  u75:0.75
  u100:1.0
)
PQ_BITS=(8 16 32)

rm -rf "$OUTPUT_DIR"
rm -rf "${LEGACY_EXPERIMENT_DIRS[@]}"
mkdir -p "$WORKLOAD_DIR" "$RESULTS_DIR" "$RUNTIME_INDEX_DIR"
rm -f "${LEGACY_SINGLE_PLOTS[@]}"

SELECTIVITY_ARGS=()
for spec in "${SELECTIVITY_SPECS[@]}"; do
  SELECTIVITY_ARGS+=(--selectivity-spec "$spec")
done

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py generate-uniform-exact-selectivity-workloads \
  --base-bin data/sift1m/sift_base.bin \
  --query-bin data/sift1m/sift_query.bin \
  --index-type float \
  --selector-type intersect \
  --out-dir "$WORKLOAD_DIR" \
  "${SELECTIVITY_ARGS[@]}" \
  > "$OUTPUT_DIR/generate_workloads.log" 2>&1

for pq_bits in "${PQ_BITS[@]}"; do
  "$ROOT_DIR/scripts/run_sift1m_uniform_prefilter_pq_compare_variant.sh" "$pq_bits" "$OUTPUT_DIR"
done

"$PYTHON_BIN" scripts/plot_prefilter_pq_compare.py \
  --series PQ8 "$RESULTS_DIR/pq8/run/results.jsonl" \
  --series PQ16 "$RESULTS_DIR/pq16/run/results.jsonl" \
  --series PQ32 "$RESULTS_DIR/pq32/run/results.jsonl" \
  --output "$OUTPUT_DIR/sift1m_uniform_prefilter_pq_compare_l100.png" \
  --plot-l 100 \
  --title "PipeANN sift1m prefilter selectivity comparison (PQ8 vs PQ16 vs PQ32, 1 thread, L=100)" \
  > "$OUTPUT_DIR/plot.log" 2>&1

echo "[ok] finished sift1m prefilter PQ comparison experiment"
echo "[ok] figure: $OUTPUT_DIR/sift1m_uniform_prefilter_pq_compare_l100.png"