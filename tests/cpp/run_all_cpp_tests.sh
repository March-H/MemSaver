#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

"${ROOT_DIR}/build.sh" \
  --target memsaver_core_shared \
  --target memsaver_core_static \
  --build-dir "${BUILD_DIR}"

echo "[test] preload-related C++ tests removed; core libraries built successfully"
