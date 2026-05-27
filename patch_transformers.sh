#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/src/openpi/models_pytorch/transformers_replace"

if [[ ! -d "${source_dir}" ]]; then
  echo "[patch_transformers.sh] Source directory not found: ${source_dir}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "[patch_transformers.sh] Python executable not found: ${python_bin}" >&2
  exit 1
fi

target_dir="$("${python_bin}" - <<'PY'
import pathlib
import transformers

print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"

if [[ -z "${target_dir}" || ! -d "${target_dir}" ]]; then
  echo "[patch_transformers.sh] Failed to resolve transformers package directory." >&2
  exit 1
fi

echo "[patch_transformers.sh] Copying custom transformers files"
echo "[patch_transformers.sh]   from: ${source_dir}"
echo "[patch_transformers.sh]   to:   ${target_dir}"
cp -r "${source_dir}/." "${target_dir}/"
