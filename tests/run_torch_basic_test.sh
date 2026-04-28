#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TORCH_DIR="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
TORCH_NVIDIA_LIB_DIRS="$(python - <<'PY'
import importlib
import os

dirs = []
for module in ("nvidia.cuda_runtime", "nvidia.cuda_cupti", "nvidia.cuda_nvrtc"):
    spec = importlib.util.find_spec(module)
    if spec is None:
        continue
    package = importlib.import_module(module)
    dirs.append(os.path.join(os.path.dirname(package.__file__), "lib"))
print(":".join(dirs))
PY
)"

"${ROOT_DIR}/build.sh" \
  --target memsaver_torch_basic_test \
  --build-dir "${BUILD_DIR}"

LD_LIBRARY_PATH="${BUILD_DIR}:${TORCH_DIR}/lib${TORCH_NVIDIA_LIB_DIRS:+:${TORCH_NVIDIA_LIB_DIRS}}:${LD_LIBRARY_PATH:-}" \
  "${BUILD_DIR}/memsaver_torch_basic_test"
