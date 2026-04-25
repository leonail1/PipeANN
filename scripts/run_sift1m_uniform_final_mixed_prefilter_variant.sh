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
SUMMARY_JSON="data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json"
PREFILTER_PREFIX="data/sift1m/sift1m_pipeann_uniform_pq${PREFILTER_PQ_BITS}"
GRAPH_PREFIX="data/sift1m/sift1m_pipeann_uniform_pq16"
FINAL_DIR="${2:-experiments/sift1m_uniform_final_mixed_prefilter_pq${PREFILTER_PQ_BITS}}"
PREFILTER_BUCKETS=(u1e-05 u3e-05 u1e-04 u3e-04 u1e-03 u3e-03 u1e-02 u1e-01)
GRAPH_BUCKETS=(u50 u75 u100)
DATASET_NAME="sift1m_uniform_final_mixed_prefilter_pq${PREFILTER_PQ_BITS}"

rm -rf "$FINAL_DIR"
mkdir -p "$FINAL_DIR"

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$PREFILTER_PREFIX" \
  --index-type float \
  --selector-type intersect \
  --manifest "$FINAL_DIR/manifest_prefilter.json" \
  > "$FINAL_DIR/build_manifest_prefilter.log" 2>&1

FINAL_DIR_ABS="$ROOT_DIR/$FINAL_DIR" PREFILTER_BUCKETS_CSV="$(IFS=,; echo "${PREFILTER_BUCKETS[*]}")" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FINAL_DIR_ABS"])
source = root / "manifest_prefilter.json"
target = root / "manifest_prefilter_selected.json"
wanted = set(os.environ["PREFILTER_BUCKETS_CSV"].split(","))
payload = json.loads(source.read_text())
payload["buckets"] = [bucket for bucket in payload["buckets"] if bucket["name"] in wanted]
payload["bucket_selection"] = sorted(wanted)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py calibrate-rerank \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$PREFILTER_PREFIX" \
  --out-dir "$FINAL_DIR/calibration_prefilter" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --search-l 100 \
  --target-recall 98 \
  > "$FINAL_DIR/calibrate_rerank_prefilter.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
  --manifest "$FINAL_DIR/manifest_prefilter_selected.json" \
  --out-dir "$FINAL_DIR/prefilter_run" \
  --dataset-name "$DATASET_NAME" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes prefilter \
  --l-values 100 \
  --prefilter-rerank-json "$FINAL_DIR/calibration_prefilter/prefilter_rerank_calibration.json" \
  > "$FINAL_DIR/run_prefilter.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$GRAPH_PREFIX" \
  --index-type float \
  --selector-type intersect \
  --manifest "$FINAL_DIR/manifest_graph.json" \
  > "$FINAL_DIR/build_manifest_graph.log" 2>&1

FINAL_DIR_ABS="$ROOT_DIR/$FINAL_DIR" GRAPH_BUCKETS_CSV="$(IFS=,; echo "${GRAPH_BUCKETS[*]}")" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FINAL_DIR_ABS"])
source = root / "manifest_graph.json"
target = root / "manifest_graph_selected.json"
wanted = set(os.environ["GRAPH_BUCKETS_CSV"].split(","))
payload = json.loads(source.read_text())
payload["buckets"] = [bucket for bucket in payload["buckets"] if bucket["name"] in wanted]
payload["bucket_selection"] = sorted(wanted)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
  --manifest "$FINAL_DIR/manifest_graph_selected.json" \
  --out-dir "$FINAL_DIR/graph_run" \
  --dataset-name "$DATASET_NAME" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes graph \
  --l-values 100 \
  > "$FINAL_DIR/run_graph.log" 2>&1

FINAL_DIR_ABS="$ROOT_DIR/$FINAL_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FINAL_DIR_ABS"])
prefilter_path = root / "prefilter_run" / "results.jsonl"
graph_path = root / "graph_run" / "results.jsonl"
output_path = root / "results.jsonl"

records = []
for source_path in (prefilter_path, graph_path):
  for line in source_path.read_text().splitlines():
    if not line.strip():
      continue
    record = json.loads(line)
    record["source_route"] = record["route"]
    record["source_index_prefix"] = record["index_prefix"]
    record["route"] = "mixed"
    records.append(record)

records.sort(key=lambda record: (record["selectivity_midpoint"], record["bucket_name"]))
with output_path.open("w", encoding="utf-8") as writer:
  for record in records:
    writer.write(json.dumps(record, sort_keys=True))
    writer.write("\n")
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py plot \
  --results-jsonl "$FINAL_DIR/results.jsonl" \
  --output "$FINAL_DIR/${DATASET_NAME}_l100.png" \
  --plot-l 100 \
  --title "PipeANN sift1m uniform selectivity (mixed: prefilter PQ${PREFILTER_PQ_BITS}, graph PQ16, 1 thread, L=100)" \
  > "$FINAL_DIR/plot.log" 2>&1

echo "[ok] finished final mixed-PQ SIFT1M uniform experiment for prefilter PQ${PREFILTER_PQ_BITS}" 