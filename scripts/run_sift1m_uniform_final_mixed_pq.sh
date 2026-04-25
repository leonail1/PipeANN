#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
SUMMARY_JSON="data/sift1m/uniform_exact_selectivity/uniform_exact_selectivity_summary.json"
FINAL_DIR="experiments/sift1m_uniform_final_mixed_pq"
PQ32_PREFIX="data/sift1m/sift1m_pipeann_uniform_pq32"
PQ16_PREFIX="data/sift1m/sift1m_pipeann_uniform_pq16"
PREFILTER_BUCKETS=(u1e-05 u3e-05 u1e-04 u3e-04 u1e-03 u3e-03 u1e-02 u1e-01)
GRAPH_BUCKETS=(u50 u75 u100)

rm -rf \
  "experiments/sift1m_uniform_pq32" \
  "experiments/sift1m_uniform_pq16" \
  "experiments/sift1m_uniform_pq8" \
  "$FINAL_DIR"

mkdir -p "$FINAL_DIR"

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$PQ32_PREFIX" \
  --index-type float \
  --selector-type intersect \
  --manifest "$FINAL_DIR/manifest_prefilter_pq32.json" \
  > "$FINAL_DIR/build_manifest_prefilter_pq32.log" 2>&1

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/data/lzg/PipeANN/experiments/sift1m_uniform_final_mixed_pq')
source = root / 'manifest_prefilter_pq32.json'
target = root / 'manifest_prefilter_pq32_selected.json'
wanted = {'u1e-05', 'u3e-05', 'u1e-04', 'u3e-04', 'u1e-03', 'u3e-03', 'u1e-02', 'u1e-01'}
payload = json.loads(source.read_text())
payload['buckets'] = [bucket for bucket in payload['buckets'] if bucket['name'] in wanted]
payload['bucket_selection'] = sorted(wanted)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py calibrate-rerank \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$PQ32_PREFIX" \
  --out-dir "$FINAL_DIR/calibration_prefilter_pq32" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --search-l 100 \
  --target-recall 98 \
  > "$FINAL_DIR/calibrate_rerank_prefilter_pq32.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
  --manifest "$FINAL_DIR/manifest_prefilter_pq32_selected.json" \
  --out-dir "$FINAL_DIR/prefilter_pq32_run" \
  --dataset-name "sift1m_uniform_final_mixed_pq" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes prefilter \
  --l-values 100 \
  --prefilter-rerank-json "$FINAL_DIR/calibration_prefilter_pq32/prefilter_rerank_calibration.json" \
  > "$FINAL_DIR/run_prefilter_pq32.log" 2>&1

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py build-manifest-from-summary \
  --summary-json "$SUMMARY_JSON" \
  --index-prefix "$PQ16_PREFIX" \
  --index-type float \
  --selector-type intersect \
  --manifest "$FINAL_DIR/manifest_graph_pq16.json" \
  > "$FINAL_DIR/build_manifest_graph_pq16.log" 2>&1

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/data/lzg/PipeANN/experiments/sift1m_uniform_final_mixed_pq')
source = root / 'manifest_graph_pq16.json'
target = root / 'manifest_graph_pq16_selected.json'
wanted = {'u50', 'u75', 'u100'}
payload = json.loads(source.read_text())
payload['buckets'] = [bucket for bucket in payload['buckets'] if bucket['name'] in wanted]
payload['bucket_selection'] = sorted(wanted)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py run \
  --manifest "$FINAL_DIR/manifest_graph_pq16_selected.json" \
  --out-dir "$FINAL_DIR/graph_pq16_run" \
  --dataset-name "sift1m_uniform_final_mixed_pq" \
  --threads 1 \
  --beamwidth 4 \
  --k 10 \
  --similarity l2 \
  --nbr-type pq \
  --mem-l 0 \
  --routes graph \
  --l-values 100 \
  > "$FINAL_DIR/run_graph_pq16.log" 2>&1

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path('/mnt/data/lzg/PipeANN/experiments/sift1m_uniform_final_mixed_pq')
prefilter_path = root / 'prefilter_pq32_run' / 'results.jsonl'
graph_path = root / 'graph_pq16_run' / 'results.jsonl'
output_path = root / 'results.jsonl'

records = []
for source_path in (prefilter_path, graph_path):
  for line in source_path.read_text().splitlines():
    if not line.strip():
      continue
    record = json.loads(line)
    record['source_route'] = record['route']
    record['source_index_prefix'] = record['index_prefix']
    record['route'] = 'mixed'
    records.append(record)

records.sort(key=lambda record: (record['selectivity_midpoint'], record['bucket_name']))
with output_path.open('w', encoding='utf-8') as writer:
  for record in records:
    writer.write(json.dumps(record, sort_keys=True))
    writer.write('\n')
PY

"$PYTHON_BIN" scripts/pipeann_hybrid_experiment.py plot \
  --results-jsonl "$FINAL_DIR/results.jsonl" \
  --output "$FINAL_DIR/sift1m_uniform_final_mixed_pq_l100.png" \
  --plot-l 100 \
  --title "PipeANN sift1m uniform selectivity (mixed: prefilter PQ32, graph PQ16, 1 thread, L=100)" \
  > "$FINAL_DIR/plot.log" 2>&1

echo "[ok] finished final mixed-PQ SIFT1M uniform experiment"