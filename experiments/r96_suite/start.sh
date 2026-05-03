#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

COMMON_ARGS=(
  --out-dir "${SCRIPT_DIR}"
  --build-r 96
  --build-l 150
  --pq-bytes 32
  --exp3-pq-bytes 32
  --exp4-pq-bytes 32
  --baseline-pq-bytes 32
)

USE_DEFAULT_BIGANN=1
for arg in "$@"; do
  case "${arg}" in
    --smoke|--base-bin|--base-bin=*)
      USE_DEFAULT_BIGANN=0
      ;;
  esac
done

EXP3_ARGS=()
if [[ "${USE_DEFAULT_BIGANN}" -eq 1 ]]; then
  "${PYTHON_BIN}" scripts/prepare_bigann_sift2m.py \
    --out-dir "${REPO_ROOT}/data/bigann" \
    --base-points 2000000 \
    --query-points 10000

  EXP3_ARGS=(
    --base-bin "${REPO_ROOT}/data/bigann/sift_base_2m_float.bin"
    --query-bin "${REPO_ROOT}/data/bigann/sift_query_10000_float.bin"
    --exp3-total-n 2000000
    --exp3-start-n 1000000
    --exp3-no-insert-n 1000000
  )
fi

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-exp1 \
  "$@"

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-exp2 \
  "$@"

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-exp3 \
  "${EXP3_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-exp4 \
  "$@"

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-exp5 \
  "$@"

"${PYTHON_BIN}" scripts/run_codex_dynamic_update_suite.py \
  "${COMMON_ARGS[@]}" \
  --only-baseline \
  --rerun-baseline \
  "$@"
