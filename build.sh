#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build.sh [--target <name>]... [--build-type <Release|Debug>] [--build-dir <path>]

Examples:
  ./build.sh
  ./build.sh --target memsaver
  ./build.sh --target memsaver_torch_basic_test
  ./build.sh --target memsaver_torch_arena_test
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
declare -a TARGETS=()

while (($# > 0)); do
  case "$1" in
    --target)
      shift
      if (($# == 0)); then
        echo "[build] missing value for --target" >&2
        usage
        exit 1
      fi
      TARGETS+=("$1")
      ;;
    --build-type)
      shift
      if (($# == 0)); then
        echo "[build] missing value for --build-type" >&2
        usage
        exit 1
      fi
      BUILD_TYPE="$1"
      ;;
    --build-dir)
      shift
      if (($# == 0)); then
        echo "[build] missing value for --build-dir" >&2
        usage
        exit 1
      fi
      BUILD_DIR="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[build] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${BUILD_DIR}"
LOG_DIR="${BUILD_DIR}/.build_logs"
mkdir -p "${LOG_DIR}"

CONFIG_LOG="${LOG_DIR}/configure.log"
BUILD_LOG="${LOG_DIR}/build.log"

if ! cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  >"${CONFIG_LOG}" 2>&1; then
  echo "[build] cmake configure failed. log: ${CONFIG_LOG}" >&2
  cat "${CONFIG_LOG}" >&2
  exit 1
fi

BUILD_CMD=(cmake --build "${BUILD_DIR}" -j)
if ((${#TARGETS[@]} > 0)); then
  for target in "${TARGETS[@]}"; do
    BUILD_CMD+=(--target "${target}")
  done
fi

if ! "${BUILD_CMD[@]}" >"${BUILD_LOG}" 2>&1; then
  echo "[build] compile failed. log: ${BUILD_LOG}" >&2
  cat "${BUILD_LOG}" >&2
  exit 1
fi

echo "[build] success (build_dir=${BUILD_DIR}, type=${BUILD_TYPE})"
