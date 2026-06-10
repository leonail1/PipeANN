#!/usr/bin/env bash
set -euo pipefail

PIPEANN_ROOT=${PIPEANN_ROOT:-$(pwd)}
BUILD_DIR=${BUILD_DIR:-"${PIPEANN_ROOT}/build"}
OH_BIN_DIR=${OH_BIN_DIR:-"${BUILD_DIR}/openharmony_acceptance"}
TEST_BIN_DIR=${TEST_BIN_DIR:-"${BUILD_DIR}/tests"}
UTIL_BIN_DIR=${UTIL_BIN_DIR:-"${BUILD_DIR}/tests/utils"}
WORK_DIR=${WORK_DIR:-"${PIPEANN_ROOT}/acceptance_work/smoke"}
RESULTS_DIR=${RESULTS_DIR:-"${PIPEANN_ROOT}/acceptance_results/smoke"}
THREADS=${THREADS:-$(nproc)}
SEARCH_THREADS=${SEARCH_THREADS:-1}
SMOKE_BATCH_UPDATE_THREADS=${SMOKE_BATCH_UPDATE_THREADS:-32}
SMOKE_BOOTSTRAP_NPOINTS=${SMOKE_BOOTSTRAP_NPOINTS:-100}
GT_THREADS=${GT_THREADS:-16}

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
mkdir -p "${WORK_DIR}" "${RESULTS_DIR}" "${WORK_DIR}/gt"

"${OH_BIN_DIR}/oh_make_synthetic" \
  --out-dir "${WORK_DIR}/data" \
  --nbase 2000 \
  --nupdates 2400 \
  --nquery 50 \
  --dim 32

INDEX_PREFIX="${WORK_DIR}/index/smoke"
DYNAMIC_FG_INDEX_PREFIX="${WORK_DIR}/dynamic_foreground_index/smoke"
DYNAMIC_BATCH_INDEX_PREFIX="${WORK_DIR}/dynamic_batch_index/smoke"
mkdir -p "$(dirname "${INDEX_PREFIX}")" "$(dirname "${DYNAMIC_FG_INDEX_PREFIX}")" "$(dirname "${DYNAMIC_BATCH_INDEX_PREFIX}")"
"${OH_BIN_DIR}/oh_generate_labels" \
  --npoints 2000 \
  --nqueries 50 \
  --index-prefix "${INDEX_PREFIX}" \
  --out-dir "${WORK_DIR}/labels"

"${TEST_BIN_DIR}/build_disk_index_filtered" \
  float "${WORK_DIR}/data/base.bin" "${INDEX_PREFIX}" \
  32 100 64 16 1 "${THREADS}" l2 pq \
  label_spmat "${WORK_DIR}/labels/base_labels.spmat" \
  range "${WORK_DIR}/labels/base_range.bin"

"${OH_BIN_DIR}/oh_build_space" \
  --raw "${WORK_DIR}/data/base.bin" \
  --index-prefix "${INDEX_PREFIX}" \
  --out-json "${RESULTS_DIR}/space_audit.json" \
  --out-csv "${RESULTS_DIR}/space_audit.csv"

while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
  [[ "${selector_id}" == "selector_id" ]] && continue
  GT="${WORK_DIR}/gt/cycle0_${selector_id}.bin"
  if [[ "${selector_type}" == "match_all" ]]; then
    env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "${UTIL_BIN_DIR}/compute_groundtruth" float l2 "${WORK_DIR}/data/base.bin" "${WORK_DIR}/data/query.bin" 10 "${GT}" null null
    LABEL_ARG="null"
  else
    env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "${UTIL_BIN_DIR}/compute_groundtruth" float l2 "${WORK_DIR}/data/base.bin" "${WORK_DIR}/data/query.bin" 10 "${GT}" null "${label_config}"
    LABEL_ARG="${label_config}"
  fi
  "${OH_BIN_DIR}/oh_static_filtered" \
    --type float \
    --index-prefix "${INDEX_PREFIX}" \
    --query "${WORK_DIR}/data/query.bin" \
    --gt "${GT}" \
    --label-config "${LABEL_ARG}" \
    --selector-id "${selector_id}" \
    --threads "${SEARCH_THREADS}" \
    --L 100 \
    --k 10 \
    --out-jsonl "${RESULTS_DIR}/static_filtered.jsonl"
done < "${WORK_DIR}/labels/selector_manifest.csv"

for cycle in 1 2; do
  CYCLE_BIN="${WORK_DIR}/data/cycle${cycle}.bin"
  "${OH_BIN_DIR}/oh_materialize_cycle_vectors" \
    --type float \
    --base "${WORK_DIR}/data/base.bin" \
    --updates "${WORK_DIR}/data/updates.bin" \
    --cycle "${cycle}" \
    --npoints 2000 \
    --update-rows-per-cycle 1200 \
    --out "${CYCLE_BIN}"
  while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
    [[ "${selector_id}" == "selector_id" ]] && continue
    GT="${WORK_DIR}/gt/cycle${cycle}_${selector_id}.bin"
    if [[ "${selector_type}" == "match_all" ]]; then
      env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "${UTIL_BIN_DIR}/compute_groundtruth" float l2 "${CYCLE_BIN}" "${WORK_DIR}/data/query.bin" 10 "${GT}" null null
    else
      env OMP_NUM_THREADS="${GT_THREADS}" OMP_THREAD_LIMIT="${GT_THREADS}" OMP_MAX_ACTIVE_LEVELS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "${UTIL_BIN_DIR}/compute_groundtruth" float l2 "${CYCLE_BIN}" "${WORK_DIR}/data/query.bin" 10 "${GT}" null "${label_config}"
    fi
  done < "${WORK_DIR}/labels/selector_manifest.csv"
done

"${OH_BIN_DIR}/oh_dynamic_chain" \
  --type float \
  --index-prefix "${DYNAMIC_FG_INDEX_PREFIX}" \
  --base "${WORK_DIR}/data/base.bin" \
  --updates "${WORK_DIR}/data/updates.bin" \
  --query "${WORK_DIR}/data/query.bin" \
  --label-config "${WORK_DIR}/labels/intersect_s25.json" \
  --label-index "${DYNAMIC_FG_INDEX_PREFIX}.label.0" \
  --range-index "${DYNAMIC_FG_INDEX_PREFIX}.label.1" \
  --npoints 2000 \
  --cycles 1 \
  --insert-threads 4 \
  --search-threads "${SEARCH_THREADS}" \
  --merge-threads 4 \
  --start-from-zero 1 \
  --bootstrap-npoints "${SMOKE_BOOTSTRAP_NPOINTS}" \
  --zero-work-dir "${WORK_DIR}/zero_start_foreground" \
  --zero-probe-selector-id intersect_s25 \
  --build-binary "${TEST_BIN_DIR}/build_disk_index_filtered" \
  --build-R 32 \
  --build-R-dense 100 \
  --build-L 64 \
  --build-PQ-bytes 16 \
  --build-mem-gb 1 \
  --build-threads "${THREADS}" \
  --foreground-rounds 4 \
  --selector-manifest "${WORK_DIR}/labels/selector_manifest.csv" \
  --gt-dir "${WORK_DIR}/gt" \
  --foreground-enabled 1 \
  --checkpoint-enabled 0 \
  --out-jsonl "${RESULTS_DIR}/dynamic_foreground_chain.jsonl" \
  --out-foreground-jsonl "${RESULTS_DIR}/dynamic_foreground_latency.jsonl" \
  --out-zero-jsonl "${RESULTS_DIR}/zero_start_exact.jsonl" \
  --out-progress-jsonl "${RESULTS_DIR}/dynamic_foreground_progress.jsonl" \
  --out-checkpoint-jsonl "${RESULTS_DIR}/dynamic_foreground_checkpoint_search.jsonl"

"${OH_BIN_DIR}/oh_dynamic_chain" \
  --type float \
  --index-prefix "${DYNAMIC_BATCH_INDEX_PREFIX}" \
  --base "${WORK_DIR}/data/base.bin" \
  --updates "${WORK_DIR}/data/updates.bin" \
  --query "${WORK_DIR}/data/query.bin" \
  --label-config null \
  --label-index "${DYNAMIC_BATCH_INDEX_PREFIX}.label.0" \
  --range-index "${DYNAMIC_BATCH_INDEX_PREFIX}.label.1" \
  --npoints 2000 \
  --cycles 2 \
  --insert-threads "${SMOKE_BATCH_UPDATE_THREADS}" \
  --search-threads "${SEARCH_THREADS}" \
  --merge-threads "${SMOKE_BATCH_UPDATE_THREADS}" \
  --start-from-zero 1 \
  --bootstrap-npoints "${SMOKE_BOOTSTRAP_NPOINTS}" \
  --zero-work-dir "${WORK_DIR}/zero_start_batch" \
  --build-binary "${TEST_BIN_DIR}/build_disk_index_filtered" \
  --build-R 32 \
  --build-R-dense 100 \
  --build-L 64 \
  --build-PQ-bytes 16 \
  --build-mem-gb 1 \
  --build-threads "${THREADS}" \
  --foreground-rounds 4 \
  --selector-manifest "${WORK_DIR}/labels/selector_manifest.csv" \
  --gt-dir "${WORK_DIR}/gt" \
  --foreground-enabled 0 \
  --checkpoint-enabled 1 \
  --save-after-insert 1 \
  --checkpoint-mode static \
  --out-jsonl "${RESULTS_DIR}/dynamic_batch_chain.jsonl" \
  --out-foreground-jsonl "${RESULTS_DIR}/dynamic_batch_foreground_latency.jsonl" \
  --out-zero-jsonl "${RESULTS_DIR}/zero_start_exact.jsonl" \
  --out-progress-jsonl "${RESULTS_DIR}/dynamic_batch_progress.jsonl" \
  --out-checkpoint-jsonl "${RESULTS_DIR}/dynamic_batch_checkpoint_search.jsonl"

/usr/bin/time -v "${OH_BIN_DIR}/oh_single_query" \
  --type float \
  --index-prefix "${INDEX_PREFIX}" \
  --query "${WORK_DIR}/data/query.bin" \
  --label-config "${WORK_DIR}/labels/range_s10.json" \
  --selector-id range_s10 \
  --L 100 \
  --k 10 \
  --out-jsonl "${RESULTS_DIR}/single_query_resource.jsonl" \
  2> "${RESULTS_DIR}/single_query_time.txt"

"${OH_BIN_DIR}/oh_summarize_results" \
  --results-dir "${RESULTS_DIR}" \
  --out-json "${RESULTS_DIR}/acceptance_summary.json" \
  --space-expansion-lt "${SPACE_EXPANSION_LT:-1000.0}" \
  --recall-min "${RECALL_MIN:-0.0}" \
  --latency-lt "${LATENCY_LT:-1000.0}" \
  --delete-ms-per-vector-lte "${DELETE_MS_PER_VECTOR_LTE:-1000.0}" \
  --single-query-max-rss-bytes-lt "${SINGLE_QUERY_MAX_RSS_BYTES_LT:-100000000000}" \
  --dynamic-foreground-cycles 1 \
  --dynamic-batch-cycles 2

echo "Smoke complete: ${RESULTS_DIR}"
