#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

export OPENPI_CODE_ROOT="${OPENPI_CODE_ROOT:-${repo_root}}"
export OPENPI_MODELS_ROOT="${OPENPI_MODELS_ROOT:-/run/determined/NAS1/public/HuggingFace/PhysicalIntelligence}"
export OPENPI_RMBENCH_REPO_ID="${OPENPI_RMBENCH_REPO_ID:-rmbench_swap_blocks_single_repo}"
export OPENPI_RMBENCH_ASSET_ID="${OPENPI_RMBENCH_ASSET_ID:-${OPENPI_RMBENCH_REPO_ID}}"
export OPENPI_RMBENCH_DATASET_ROOT="${OPENPI_RMBENCH_DATASET_ROOT:-/run/determined/NAS1/public/wangshuxun/lerobot/${OPENPI_RMBENCH_REPO_ID}}"
export OPENPI_VISION_FEATURE_ID="${OPENPI_VISION_FEATURE_ID:-pi05_base_pytorch}"

export TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_rmbench_framesamp_context_pytorch_precomputed_lora}"
export EXP_NAME="${EXP_NAME:-framesamp_context_precomputed_lora_amp80g_4gpu}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-4}"
export TRAIN_OVERWRITE="${TRAIN_OVERWRITE:-1}"

exec "${script_dir}/train_rmbench_common.sh"
