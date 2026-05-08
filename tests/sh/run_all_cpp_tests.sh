#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

"${ROOT_DIR}/build.sh" \
  --target memsaver \
  --target memsaver_basic_test \
  --target memsaver_arena_test \
  --target memsaver_arena_virtual_test \
  --target memsaver_model_load_test \
  --build-dir "${BUILD_DIR}"

echo "[test] memsaver targets built successfully"
