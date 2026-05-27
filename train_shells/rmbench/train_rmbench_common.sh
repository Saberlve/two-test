#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export PATH="${HOME}/.local/bin:${PATH}"
export OPENPI_CODE_ROOT="${OPENPI_CODE_ROOT:-${repo_root}}"
export PYTHONPATH="${OPENPI_CODE_ROOT}/src:${PYTHONPATH:-}"
export PRINT_UNUSED_PARAMETERS="${PRINT_UNUSED_PARAMETERS:-1}"
export HF_HOME="${HF_HOME:-/tmp/openpi-hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/openpi-datasets-cache}"

required_vars=(
  OPENPI_RMBENCH_REPO_ID
  OPENPI_RMBENCH_DATASET_ROOT
  TRAIN_CONFIG
  EXP_NAME
  TRAIN_NPROC_PER_NODE
  TRAIN_OVERWRITE
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "[framesamp_train_rmbench_common.sh] ${var_name} must be set by the caller script." >&2
    exit 1
  fi
done

if [[ ! -d "${OPENPI_RMBENCH_DATASET_ROOT}" ]]; then
  echo "[framesamp_train_rmbench_common.sh] RMBench dataset root does not exist: ${OPENPI_RMBENCH_DATASET_ROOT}" >&2
  exit 1
fi

feature_dir="${OPENPI_RMBENCH_DATASET_ROOT}/vision_features/${OPENPI_VISION_FEATURE_ID:-pi05_base_pytorch}/chunk-000"
if [[ ! -d "${feature_dir}" ]]; then
  echo "[framesamp_train_rmbench_common.sh] Missing precomputed feature dir: ${feature_dir}" >&2
  exit 1
fi

cd "${OPENPI_CODE_ROOT}"
if [[ -f "${OPENPI_CODE_ROOT}/.venv/bin/activate" ]]; then
  source "${OPENPI_CODE_ROOT}/.venv/bin/activate"
fi
if [[ -x "${OPENPI_CODE_ROOT}/patch_transformers.sh" ]]; then
  "${OPENPI_CODE_ROOT}/patch_transformers.sh"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[framesamp_train_rmbench_common.sh] ERROR: torchrun not found in current environment." >&2
  exit 127
fi

log_dir="${OPENPI_CODE_ROOT}/logs"
mkdir -p "${log_dir}"
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="${LOG_FILE:-${log_dir}/train_rmbench_framesamp_${timestamp}.log}"

echo "[framesamp_train_rmbench_common.sh] Logging to: ${log_file}"
echo "[framesamp_train_rmbench_common.sh] train_config=${TRAIN_CONFIG}"
echo "[framesamp_train_rmbench_common.sh] exp_name=${EXP_NAME}"
echo "[framesamp_train_rmbench_common.sh] nproc_per_node=${TRAIN_NPROC_PER_NODE}"
echo "[framesamp_train_rmbench_common.sh] OPENPI_RMBENCH_REPO_ID=${OPENPI_RMBENCH_REPO_ID}"
echo "[framesamp_train_rmbench_common.sh] OPENPI_RMBENCH_DATASET_ROOT=${OPENPI_RMBENCH_DATASET_ROOT}"
echo "[framesamp_train_rmbench_common.sh] feature_dir=${feature_dir}"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${TRAIN_NPROC_PER_NODE}"
  scripts/train_pytorch.py
  "${TRAIN_CONFIG}"
  --exp_name
  "${EXP_NAME}"
  --project_name
  "openpi-rmbench"
)

if [[ "${TRAIN_OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
else
  cmd+=(--resume)
fi

printf '[framesamp_train_rmbench_common.sh] command=' | tee -a "${log_file}"
printf ' %q' "${cmd[@]}" | tee -a "${log_file}"
printf '\n' | tee -a "${log_file}"

set -o pipefail
"${cmd[@]}" 2>&1 | tee -a "${log_file}"
exit_code=${PIPESTATUS[0]}
echo "[framesamp_train_rmbench_common.sh] Exit code: ${exit_code}" | tee -a "${log_file}"
exit "${exit_code}"
