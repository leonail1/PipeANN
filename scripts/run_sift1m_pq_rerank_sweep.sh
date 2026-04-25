#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
LABEL_FILE="data/sift1m/uniform_exact_selectivity/base.uniform_exact_selectivity.spmat"
QUERY_BIN="data/sift1m/sift_query.bin"
SUMMARY_JSON="data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json"
BUCKETS=(u1e-05 u3e-05 u1e-04 u3e-04 u1e-03 u3e-03 u1e-02 u1e-01 u50 u75 u100)
PQ_VARIANTS=(32 16 8)

for pq in "${PQ_VARIANTS[@]}"; do
  prefix="data/sift1m/sift1m_pipeann_uniform_pq${pq}"
  outdir="experiments/sift1m_uniform_pq${pq}"

  mkdir -p "$outdir"
  rm -f \
    "${prefix}_disk.index" \
    "${prefix}_disk.index.tags" \
    "${prefix}_labels.densebit" \
    "${prefix}_hybrid.meta" \
    "${prefix}_mem.index.tags" \
    "${prefix}_pq_compressed.bin" \
    "${prefix}_pq_pivots.bin" \
    "${prefix}_partition.bin.aligned"
  rm -rf "$outdir/calibration" "$outdir/run"

  ./build/tests/build_disk_index \
    float data/sift1m/sift_base.bin "$prefix" \
    64 96 "$pq" 64 52 l2 pq spmat "$LABEL_FILE" \
    > "$outdir/build_disk_index.log" 2>&1

  threshold_cmd=(./build/tests/calibrate_hybrid_threshold float "$prefix" 52 4 10 l2 pq 0 100)
  for bucket in "${BUCKETS[@]}"; do
    threshold_cmd+=(intersect "$QUERY_BIN" "data/sift1m/uniform_exact_selectivity/${bucket}/queries.spmat" 200)
  done
  "${threshold_cmd[@]}" > "$outdir/calibrate_threshold.log" 2>&1

  "$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
    --summary-json "$SUMMARY_JSON" \
    --index-prefix "$prefix" \
    --index-type float \
    --selector-type intersect \
    --manifest "$outdir/manifest.json" \
    > "$outdir/build_manifest.log" 2>&1

  "$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py calibrate-rerank \
    --summary-json "$SUMMARY_JSON" \
    --index-prefix "$prefix" \
    --out-dir "$outdir/calibration" \
    --threads 52 \
    --beamwidth 4 \
    --k 10 \
    --similarity l2 \
    --nbr-type pq \
    --search-l 100 \
    --target-recall 98 \
    > "$outdir/calibrate_rerank.log" 2>&1

  "$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
    --manifest "$outdir/manifest.json" \
    --out-dir "$outdir/run" \
    --dataset-name "sift1m_pq${pq}" \
    --threads 52 \
    --beamwidth 4 \
    --k 10 \
    --similarity l2 \
    --nbr-type pq \
    --mem-l 0 \
    --routes auto \
    --l-values 100 \
    --prefilter-rerank-json "$outdir/calibration/prefilter_rerank_calibration.json" \
    > "$outdir/run_experiment.log" 2>&1

  "$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py plot \
    --results-jsonl "$outdir/run/results.jsonl" \
    --output "$outdir/sift1m_uniform_pq${pq}_auto_l100.png" \
    --plot-l 100 \
    --title "PipeANN sift1m uniform selectivity (PQ${pq}, calibrated rerank, auto, L=100)" \
    > "$outdir/plot.log" 2>&1
done

echo "[ok] finished SIFT1M PQ rerank sweep"