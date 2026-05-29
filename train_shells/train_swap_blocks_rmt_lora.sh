#!/usr/bin/env bash
set -euo pipefail

# ---- paths ----
export OPENPI_CODE_ROOT="$HOME/two-test"
export OPENPI_MODELS_ROOT="/mount/localcq1/wangshuxun/models"
export OPENPI_CHECKPOINTS_ROOT="/dev/shm/wsx/checkpoints"
export OPENPI_DATASETS_ROOT="/mount/localcq1/wangshuxun/datasets"
export OPENPI_CACHE_ROOT="/mount/localcq1/wangshuxun"

# Config hardcodes dataset paths to /dev/shm/wsx/dataset/...; keep env vars
# consistent in case anything else references them.
export OPENPI_RMBENCH_REPO_ID="rmbench_swap_blocks_single_repo"
export OPENPI_RMBENCH_DATASET_ROOT="/dev/shm/wsx/dataset/rmbench_swap_blocks_single_repo"

# ---- python env (reuse MemoryRobot venv) ----
VENV_DIR="$HOME/MemoryRobot/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[launch] MemoryRobot venv not found at ${VENV_DIR}" >&2
  exit 1
fi
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON_BIN="${VENV_DIR}/bin/python"

# ---- wandb (api.bandw.top mirror, same as MemoryRobot scripts) ----
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

# Pull WANDB_API_KEY at runtime from existing MemoryRobot script (not embedded here).
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  _wandb_src="${HOME}/MemoryRobot/train_shells/simpler/train_a100.sh"
  if [[ -r "${_wandb_src}" ]]; then
    eval "$(grep -E '^export WANDB_API_KEY=' "${_wandb_src}")"
  fi
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[launch] WARNING: WANDB_API_KEY not set; wandb.init() will fail." >&2
fi

# ---- transformers monkey-patch from two-test's transformers_replace ----
SOURCE_DIR="${OPENPI_CODE_ROOT}/src/openpi/models_pytorch/transformers_replace"
if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[launch] transformers_replace not found: ${SOURCE_DIR}" >&2
  exit 1
fi
TARGET_DIR="$("${PYTHON_BIN}" -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).resolve().parent)')"
echo "[launch] Patching transformers: ${SOURCE_DIR} -> ${TARGET_DIR}"
cp -r "${SOURCE_DIR}/." "${TARGET_DIR}/"

# ---- dataset existence sanity ----
if [[ ! -d "${OPENPI_RMBENCH_DATASET_ROOT}" ]]; then
  echo "[launch] Dataset root missing: ${OPENPI_RMBENCH_DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${OPENPI_MODELS_ROOT}/pi05_base_pytorch" ]]; then
  echo "[launch] Base weights missing: ${OPENPI_MODELS_ROOT}/pi05_base_pytorch" >&2
  exit 1
fi

# ---- train args ----
TRAIN_CONFIG="pi05_rmbench_swap_blocks_rmt_lora"
EXP_NAME="${EXP_NAME:-swap_blocks_rmt_lora_v1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PROJECT_NAME="openpi-rmbench"

cd "${OPENPI_CODE_ROOT}"
export PYTHONPATH="${OPENPI_CODE_ROOT}/src:${PYTHONPATH:-}"
export PRINT_UNUSED_PARAMETERS="${PRINT_UNUSED_PARAMETERS:-1}"

LOG_DIR="${OPENPI_CODE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_${EXP_NAME}_${TS}.log"

echo "[launch] train_config=${TRAIN_CONFIG}"
echo "[launch] exp_name=${EXP_NAME}"
echo "[launch] nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] log=${LOG_FILE}"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  scripts/train_pytorch.py
  "${TRAIN_CONFIG}"
  --exp_name "${EXP_NAME}"
  --project_name "${PROJECT_NAME}"
  --resume
  --checkpoint_base_dir "/dev/shm/wsx/checkpoints"
)

printf '[launch] cmd=' | tee -a "${LOG_FILE}"
printf ' %q' "${cmd[@]}" | tee -a "${LOG_FILE}"
printf '\n' | tee -a "${LOG_FILE}"

set -o pipefail
"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
exit_code=${PIPESTATUS[0]}
echo "[launch] exit_code=${exit_code}" | tee -a "${LOG_FILE}"
exit "${exit_code}"
