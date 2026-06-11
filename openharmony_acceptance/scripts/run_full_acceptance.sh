#!/usr/bin/env bash
set -euo pipefail

PIPEANN_ROOT=${PIPEANN_ROOT:-$(pwd)}
BUILD_DIR=${BUILD_DIR:-"${PIPEANN_ROOT}/build"}
OH_BIN_DIR=${OH_BIN_DIR:-"${BUILD_DIR}/openharmony_acceptance"}
TEST_BIN_DIR=${TEST_BIN_DIR:-"${BUILD_DIR}/tests"}
UTIL_BIN_DIR=${UTIL_BIN_DIR:-"${BUILD_DIR}/tests/utils"}
WORK_DIR=${WORK_DIR:-"${PIPEANN_ROOT}/acceptance_work/full"}
RESULTS_DIR=${RESULTS_DIR:-"${PIPEANN_ROOT}/acceptance_results/full"}
THREADS=${THREADS:-$(nproc)}
SEARCH_THREADS=${SEARCH_THREADS:-1}
FOREGROUND_UPDATE_THREADS=${FOREGROUND_UPDATE_THREADS:-4}
BATCH_UPDATE_THREADS=${BATCH_UPDATE_THREADS:-${UPDATE_THREADS:-32}}
GT_THREADS=${GT_THREADS:-16}
PQ_RETRAIN_AFTER_INSERT=${PQ_RETRAIN_AFTER_INSERT:-1}
PQ_RETRAIN_THREADS=${PQ_RETRAIN_THREADS:-8}
: "${PIPEANN_ATTR_DELTA_MERGE_BYTES:=${ATTR_DELTA_MERGE_BYTES:-67108864}}"
export PIPEANN_ATTR_DELTA_MERGE_BYTES
: "${PIPEANN_ATTR_TIMING:=0}"
export PIPEANN_ATTR_TIMING

: "${BASE_BIN:?Set BASE_BIN to the 1M base .bin file}"
: "${UPDATES_BIN:?Set UPDATES_BIN to at least 3M update vectors in .bin format}"
: "${QUERY_BIN:?Set QUERY_BIN to the query .bin file}"

TYPE=${TYPE:-float}
METRIC=${METRIC:-l2}
NPOINTS=${NPOINTS:-1000000}
NQUERIES=${NQUERIES:-1000}
FOREGROUND_CYCLES=${FOREGROUND_CYCLES:-1}
BATCH_CYCLES=${BATCH_CYCLES:-${CYCLES:-5}}
R=${R:-96}
R_DENSE=${R_DENSE:-0}
BUILD_L=${BUILD_L:-128}
PQ_BYTES=${PQ_BYTES:-32}
MEM_GB=${MEM_GB:-64}
SEARCH_L=${SEARCH_L:-100}
L_CANDIDATES=${L_CANDIDATES:-20,40,60,80,100,150,200,300,400,600,800}
K=${K:-10}
INDEX_PREFIX=${INDEX_PREFIX:-"${WORK_DIR}/index/sift1m"}
ZERO_BOOTSTRAP_NPOINTS=${ZERO_BOOTSTRAP_NPOINTS:-10000}
DYNAMIC_FG_INDEX_PREFIX=${DYNAMIC_FG_INDEX_PREFIX:-"${WORK_DIR}/dynamic_foreground_index/sift1m"}
DYNAMIC_BATCH_INDEX_PREFIX=${DYNAMIC_BATCH_INDEX_PREFIX:-"${WORK_DIR}/dynamic_batch_index/sift1m"}

safe_reset_dir() {
  local dir="$1"
  if [[ -z "${dir}" || "${dir}" == "/" ]]; then
    echo "Refusing to remove unsafe directory: ${dir}" >&2
    exit 1
  fi
  case "${dir}" in
    *acceptance_work|*acceptance_work/*|*acceptance_results|*acceptance_results/*)
      rm -rf "${dir}"
      ;;
    *)
      if [[ "${ALLOW_ACCEPTANCE_RM:-0}" == "1" ]]; then
        rm -rf "${dir}"
      else
        echo "Refusing to remove ${dir}; set ALLOW_ACCEPTANCE_RM=1 to override" >&2
        exit 1
      fi
      ;;
  esac
}

safe_reset_dir "${WORK_DIR}"
safe_reset_dir "${RESULTS_DIR}"
mkdir -p "${WORK_DIR}/gt" "${RESULTS_DIR}"
mkdir -p "$(dirname "${INDEX_PREFIX}")"
mkdir -p "$(dirname "${DYNAMIC_FG_INDEX_PREFIX}")" "$(dirname "${DYNAMIC_BATCH_INDEX_PREFIX}")"

: "${PIPEANN_ATTR_TIMING_PATH:=${RESULTS_DIR}/attr_timing.csv}"
export PIPEANN_ATTR_TIMING_PATH

GT_CACHE_SCHEMA=${GT_CACHE_SCHEMA:-oh_cpp_acceptance_gt_v4_split_dynamic_uniform_labels}
GT_CACHE_ROOT=${GT_CACHE_ROOT:-"${PIPEANN_ROOT}/acceptance_gt_cache"}
GT_CACHE_KEY=$(python3 - "${GT_CACHE_SCHEMA}" "${BASE_BIN}" "${UPDATES_BIN}" "${QUERY_BIN}" "${TYPE}" "${METRIC}" "${NPOINTS}" "${NQUERIES}" "${BATCH_CYCLES}" "${K}" <<'PYGTKEY'
import hashlib, os, sys
parts = list(sys.argv[1:])
for path in sys.argv[2:5]:
    st = os.stat(path)
    parts.extend([os.path.realpath(path), str(st.st_size), str(st.st_mtime_ns)])
print(hashlib.sha1("\0".join(parts).encode()).hexdigest()[:20])
PYGTKEY
)
GT_CACHE_DIR=${GT_CACHE_DIR:-"${GT_CACHE_ROOT}/${GT_CACHE_KEY}"}
mkdir -p "${GT_CACHE_DIR}"
echo "GT cache dir: ${GT_CACHE_DIR}"
echo "PIPEANN_ATTR_DELTA_MERGE_BYTES=${PIPEANN_ATTR_DELTA_MERGE_BYTES}"
echo "PIPEANN_ATTR_TIMING=${PIPEANN_ATTR_TIMING}"
echo "PIPEANN_ATTR_TIMING_PATH=${PIPEANN_ATTR_TIMING_PATH}"
echo "PQ_RETRAIN_AFTER_INSERT=${PQ_RETRAIN_AFTER_INSERT}"
echo "PQ_RETRAIN_THREADS=${PQ_RETRAIN_THREADS}"

compute_gt_cached() {
  local cycle="$1"
  local selector_id="$2"
  local selector_type="$3"
  local data_bin="$4"
  local label_config="$5"
  local out_gt="$6"
  local cache_gt="${GT_CACHE_DIR}/cycle${cycle}_${selector_id}.bin"
  mkdir -p "$(dirname "${out_gt}")"
  if [[ -s "${cache_gt}" ]]; then
    cp -f "${cache_gt}" "${out_gt}"
    echo "GT cache hit: cycle${cycle} ${selector_id}"
    return 0
  fi
  echo "GT cache miss: cycle${cycle} ${selector_id}"
  local tmp_gt="${out_gt}.tmp.$$"
  rm -f "${tmp_gt}"
  if [[ "${selector_type}" == "match_all" ]]; then
    env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
      "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${data_bin}" "${QUERY_ACTIVE}" "${K}" "${tmp_gt}" null null
  else
    env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
      "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${data_bin}" "${QUERY_ACTIVE}" "${K}" "${tmp_gt}" null "${label_config}"
  fi
  mv -f "${tmp_gt}" "${out_gt}"
  cp -f "${out_gt}" "${cache_gt}"
}

"${OH_BIN_DIR}/oh_generate_labels" \
  --npoints "${NPOINTS}" \
  --nqueries "${NQUERIES}" \
  --index-prefix "${INDEX_PREFIX}" \
  --out-dir "${WORK_DIR}/labels"

QUERY_ACTIVE="${WORK_DIR}/query_${NQUERIES}.bin"
"${OH_BIN_DIR}/oh_materialize_cycle_vectors" \
  --type "${TYPE}" \
  --base "${QUERY_BIN}" \
  --updates "${QUERY_BIN}" \
  --cycle 0 \
  --npoints "${NQUERIES}" \
  --out "${QUERY_ACTIVE}"

"${TEST_BIN_DIR}/build_disk_index_filtered" \
  "${TYPE}" "${BASE_BIN}" "${INDEX_PREFIX}" \
  "${R}" "${R_DENSE}" "${BUILD_L}" "${PQ_BYTES}" "${MEM_GB}" "${THREADS}" "${METRIC}" pq \
  label_spmat "${WORK_DIR}/labels/base_labels.spmat" \
  range "${WORK_DIR}/labels/base_range.bin"

"${OH_BIN_DIR}/oh_build_space" \
  --raw "${BASE_BIN}" \
  --index-prefix "${INDEX_PREFIX}" \
  --out-json "${RESULTS_DIR}/space_audit.json" \
  --out-csv "${RESULTS_DIR}/space_audit.csv"

while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
  [[ "${selector_id}" == "selector_id" ]] && continue
  GT="${WORK_DIR}/gt/cycle0_${selector_id}.bin"
  if [[ "${selector_type}" == "match_all" ]]; then
    LABEL_ARG="null"
  else
    LABEL_ARG="${label_config}"
  fi
  compute_gt_cached 0 "${selector_id}" "${selector_type}" "${BASE_BIN}" "${label_config}" "${GT}"
  "${OH_BIN_DIR}/oh_static_filtered" \
    --type "${TYPE}" \
    --metric "${METRIC}" \
    --index-prefix "${INDEX_PREFIX}" \
    --query "${QUERY_ACTIVE}" \
    --gt "${GT}" \
    --label-config "${LABEL_ARG}" \
    --selector-id "${selector_id}" \
    --threads "${SEARCH_THREADS}" \
    --L "${SEARCH_L}" \
    --L-candidates "${L_CANDIDATES}" \
    --recall-min "${RECALL_MIN:-98.0}" \
    --k "${K}" \
    --out-jsonl "${RESULTS_DIR}/static_filtered.jsonl"
done < "${WORK_DIR}/labels/selector_manifest.csv"

UPDATE_ROWS_PER_CYCLE=$((NPOINTS * 6 / 10))
for cycle in $(seq 1 "${BATCH_CYCLES}"); do
  CYCLE_BIN="${WORK_DIR}/cycle${cycle}.bin"
  "${OH_BIN_DIR}/oh_materialize_cycle_vectors" \
    --type "${TYPE}" \
    --base "${BASE_BIN}" \
    --updates "${UPDATES_BIN}" \
    --cycle "${cycle}" \
    --npoints "${NPOINTS}" \
    --update-rows-per-cycle "${UPDATE_ROWS_PER_CYCLE}" \
    --out "${CYCLE_BIN}"
  while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
    [[ "${selector_id}" == "selector_id" ]] && continue
    GT="${WORK_DIR}/gt/cycle${cycle}_${selector_id}.bin"
    compute_gt_cached "${cycle}" "${selector_id}" "${selector_type}" "${CYCLE_BIN}" "${label_config}" "${GT}"
  done < "${WORK_DIR}/labels/selector_manifest.csv"
done

FOREGROUND_INFO=$(python3 - "${RESULTS_DIR}/static_filtered.jsonl" "${WORK_DIR}/labels/selector_manifest.csv" <<'PYFG'
import csv, json, sys
static_path, manifest_path = sys.argv[1], sys.argv[2]
rows=[]
with open(static_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
if not rows:
    raise SystemExit('static_filtered.jsonl is empty; cannot calibrate foreground selector')
# Use the slowest static selector as foreground stress, but use its selected_L,
# i.e. the minimal L observed to satisfy the recall target in static search.
best=max(rows, key=lambda r: (float(r.get('avg_latency_ms', 0.0)), float(r.get('p99_latency_ms', 0.0)), str(r.get('selector_id', ''))))
manifest={}
with open(manifest_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        manifest[row['selector_id']]=row
selector_id=best['selector_id']
if selector_id not in manifest:
    raise SystemExit(f'selector {selector_id} not found in manifest')
label_config=manifest[selector_id].get('label_config') or 'null'
selected_l=int(best.get('selected_L') or best.get('L'))
print(f'{selector_id}\t{selected_l}\t{label_config}\t{best.get("avg_latency_ms")}\t{best.get("p99_latency_ms")}')
PYFG
)
IFS=$'\t' read -r FOREGROUND_SELECTOR_ID FOREGROUND_L FOREGROUND_CONFIG FOREGROUND_STATIC_AVG FOREGROUND_STATIC_P99 <<< "${FOREGROUND_INFO}"
echo "Calibrated foreground selector: ${FOREGROUND_SELECTOR_ID} L=${FOREGROUND_L} static_avg_ms=${FOREGROUND_STATIC_AVG} static_p99_ms=${FOREGROUND_STATIC_P99} config=${FOREGROUND_CONFIG}"
echo "Running foreground interference test: cycles=${FOREGROUND_CYCLES} update_threads=${FOREGROUND_UPDATE_THREADS} search_threads=${SEARCH_THREADS}"
"${OH_BIN_DIR}/oh_dynamic_chain" \
  --type "${TYPE}" \
  --metric "${METRIC}" \
  --index-prefix "${DYNAMIC_FG_INDEX_PREFIX}" \
  --base "${BASE_BIN}" \
  --updates "${UPDATES_BIN}" \
  --query "${QUERY_ACTIVE}" \
  --label-config "${FOREGROUND_CONFIG}" \
  --label-index "${DYNAMIC_FG_INDEX_PREFIX}.label.0" \
  --range-index "${DYNAMIC_FG_INDEX_PREFIX}.label.1" \
  --npoints "${NPOINTS}" \
  --cycles "${FOREGROUND_CYCLES}" \
  --insert-threads "${FOREGROUND_UPDATE_THREADS}" \
  --search-threads "${SEARCH_THREADS}" \
  --merge-threads "${FOREGROUND_UPDATE_THREADS}" \
  --start-from-zero 1 \
  --bootstrap-npoints "${ZERO_BOOTSTRAP_NPOINTS}" \
  --zero-work-dir "${WORK_DIR}/zero_start_foreground" \
  --zero-probe-selector-id "${FOREGROUND_SELECTOR_ID}" \
  --build-binary "${TEST_BIN_DIR}/build_disk_index_filtered" \
  --build-R "${R}" \
  --build-R-dense "${R_DENSE}" \
  --build-L "${BUILD_L}" \
  --build-PQ-bytes "${PQ_BYTES}" \
  --build-mem-gb "${MEM_GB}" \
  --build-threads "${THREADS}" \
  --L "${FOREGROUND_L}" \
  --L-candidates "${L_CANDIDATES}" \
  --recall-min "${RECALL_MIN:-98.0}" \
  --selector-manifest "${WORK_DIR}/labels/selector_manifest.csv" \
  --gt-dir "${WORK_DIR}/gt" \
  --foreground-enabled 1 \
  --checkpoint-enabled 0 \
  --out-jsonl "${RESULTS_DIR}/dynamic_foreground_chain.jsonl" \
  --out-foreground-jsonl "${RESULTS_DIR}/dynamic_foreground_latency.jsonl" \
  --out-zero-jsonl "${RESULTS_DIR}/zero_start_exact.jsonl" \
  --out-progress-jsonl "${RESULTS_DIR}/dynamic_foreground_progress.jsonl" \
  --out-checkpoint-jsonl "${RESULTS_DIR}/dynamic_foreground_checkpoint_search.jsonl"

echo "Running batch dynamic quality test: cycles=${BATCH_CYCLES} update_threads=${BATCH_UPDATE_THREADS} search_threads=${SEARCH_THREADS}"
"${OH_BIN_DIR}/oh_dynamic_chain" \
  --type "${TYPE}" \
  --metric "${METRIC}" \
  --index-prefix "${DYNAMIC_BATCH_INDEX_PREFIX}" \
  --base "${BASE_BIN}" \
  --updates "${UPDATES_BIN}" \
  --query "${QUERY_ACTIVE}" \
  --label-config null \
  --label-index "${DYNAMIC_BATCH_INDEX_PREFIX}.label.0" \
  --range-index "${DYNAMIC_BATCH_INDEX_PREFIX}.label.1" \
  --npoints "${NPOINTS}" \
  --cycles "${BATCH_CYCLES}" \
  --insert-threads "${BATCH_UPDATE_THREADS}" \
  --search-threads "${SEARCH_THREADS}" \
  --merge-threads "${BATCH_UPDATE_THREADS}" \
  --start-from-zero 1 \
  --bootstrap-npoints "${ZERO_BOOTSTRAP_NPOINTS}" \
  --zero-work-dir "${WORK_DIR}/zero_start_batch" \
  --build-binary "${TEST_BIN_DIR}/build_disk_index_filtered" \
  --build-R "${R}" \
  --build-R-dense "${R_DENSE}" \
  --build-L "${BUILD_L}" \
  --build-PQ-bytes "${PQ_BYTES}" \
  --build-mem-gb "${MEM_GB}" \
  --build-threads "${THREADS}" \
  --L "${SEARCH_L}" \
  --L-candidates "${L_CANDIDATES}" \
  --recall-min "${RECALL_MIN:-98.0}" \
  --selector-manifest "${WORK_DIR}/labels/selector_manifest.csv" \
  --gt-dir "${WORK_DIR}/gt" \
  --foreground-enabled 0 \
  --checkpoint-enabled 1 \
  --save-after-insert 1 \
  --pq-retrain-after-insert "${PQ_RETRAIN_AFTER_INSERT}" \
  --pq-retrain-binary "${OH_BIN_DIR}/oh_rebuild_pq" \
  --pq-retrain-data-dir "${WORK_DIR}" \
  --pq-retrain-threads "${PQ_RETRAIN_THREADS}" \
  --pq-retrain-pq-bytes "${PQ_BYTES}" \
  --checkpoint-mode static \
  --out-jsonl "${RESULTS_DIR}/dynamic_batch_chain.jsonl" \
  --out-foreground-jsonl "${RESULTS_DIR}/dynamic_batch_foreground_latency.jsonl" \
  --out-zero-jsonl "${RESULTS_DIR}/zero_start_exact.jsonl" \
  --out-progress-jsonl "${RESULTS_DIR}/dynamic_batch_progress.jsonl" \
  --out-checkpoint-jsonl "${RESULTS_DIR}/dynamic_batch_checkpoint_search.jsonl"

SINGLE_QUERY_SELECTOR_ID=${SINGLE_QUERY_SELECTOR_ID:-range_s10}
SINGLE_QUERY_INFO=$(python3 - "${RESULTS_DIR}/static_filtered.jsonl" "${WORK_DIR}/labels/selector_manifest.csv" "${SINGLE_QUERY_SELECTOR_ID}" "${SEARCH_L}" <<'PYSINGLEINFO'
import csv
import json
import sys

static_path, manifest_path, selector_id, fallback = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
selected_l = fallback
selected_row = None
with open(static_path, encoding="utf-8") as reader:
    for line in reader:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("selector_id") == selector_id:
            selected_row = row
            selected_l = str(int(row.get("selected_L", row.get("L", fallback))))
            break
if selected_row is None:
    raise SystemExit(f"selector {selector_id} not found in {static_path}")
manifest = {}
with open(manifest_path, newline="", encoding="utf-8") as reader:
    for row in csv.DictReader(reader):
        manifest[row["selector_id"]] = row
if selector_id not in manifest:
    raise SystemExit(f"selector {selector_id} not found in {manifest_path}")
label_config = manifest[selector_id].get("label_config") or "null"
print("\t".join([
    selector_id,
    selected_l,
    label_config,
    str(selected_row.get("recall_at_10", "")),
    str(selected_row.get("avg_latency_ms", "")),
    str(selected_row.get("p99_latency_ms", "")),
]))
PYSINGLEINFO
)
IFS=$'\t' read -r SINGLE_QUERY_SELECTOR_ID SINGLE_QUERY_L SINGLE_QUERY_LABEL_CONFIG SINGLE_QUERY_STATIC_RECALL SINGLE_QUERY_STATIC_AVG SINGLE_QUERY_STATIC_P99 <<< "${SINGLE_QUERY_INFO}"
if [[ "${SINGLE_QUERY_LABEL_CONFIG}" != "null" && ! -s "${SINGLE_QUERY_LABEL_CONFIG}" ]]; then
  echo "Single-query selector config not found: ${SINGLE_QUERY_LABEL_CONFIG}" >&2
  exit 1
fi
python3 - "${SINGLE_QUERY_SELECTOR_ID}" "${SINGLE_QUERY_L}" "${SINGLE_QUERY_LABEL_CONFIG}" "${SINGLE_QUERY_STATIC_RECALL}" "${SINGLE_QUERY_STATIC_AVG}" "${SINGLE_QUERY_STATIC_P99}" "${RESULTS_DIR}/single_query_config.json" <<'PYSINGLECFG'
import json
import sys

selector_id, selected_l, label_config, recall, avg, p99, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
payload = {
    "selector_id": selector_id,
    "selected_L": selected_l,
    "label_config": label_config,
    "source": "static_filtered_selected_L",
}
for key, value in {
    "static_recall_at_10": recall,
    "static_avg_latency_ms": avg,
    "static_p99_latency_ms": p99,
}.items():
    if value != "":
        payload[key] = float(value)
with open(out_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PYSINGLECFG
echo "Running single-query resource test: selector=${SINGLE_QUERY_SELECTOR_ID} L=${SINGLE_QUERY_L} source=static_filtered_selected_L"
/usr/bin/time -v "${OH_BIN_DIR}/oh_single_query" \
  --type "${TYPE}" \
  --metric "${METRIC}" \
  --index-prefix "${INDEX_PREFIX}" \
  --query "${QUERY_ACTIVE}" \
  --label-config "${SINGLE_QUERY_LABEL_CONFIG}" \
  --selector-id "${SINGLE_QUERY_SELECTOR_ID}" \
  --L "${SINGLE_QUERY_L}" \
  --k "${K}" \
  --out-jsonl "${RESULTS_DIR}/single_query_resource.jsonl" \
  2> "${RESULTS_DIR}/single_query_time.txt"

"${OH_BIN_DIR}/oh_summarize_results" \
  --results-dir "${RESULTS_DIR}" \
  --out-json "${RESULTS_DIR}/acceptance_summary.json" \
  --space-expansion-lt "${SPACE_EXPANSION_LT:-2.0}" \
  --recall-min "${RECALL_MIN:-98.0}" \
  --latency-lt "${LATENCY_LT:-10.0}" \
  --delete-ms-per-vector-lte "${DELETE_MS_PER_VECTOR_LTE:-0.5}" \
  --single-query-max-rss-bytes-lt "${SINGLE_QUERY_MAX_RSS_BYTES_LT:-30000000}" \
  --dynamic-foreground-cycles "${FOREGROUND_CYCLES}" \
  --dynamic-batch-cycles "${BATCH_CYCLES}"

echo "Full acceptance complete: ${RESULTS_DIR}"
