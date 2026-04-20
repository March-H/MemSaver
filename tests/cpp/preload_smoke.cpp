#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "memsaver/memsaver_c.h"

namespace {

void CheckCu(const CUresult status, const char* expr) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* error_string = nullptr;
  (void)cuGetErrorString(status, &error_string);
  std::cerr << "[preload_smoke] " << expr << " failed: "
            << (error_string == nullptr ? "Unknown error" : error_string)
            << " (" << static_cast<int>(status) << ")" << std::endl;
  std::exit(1);
}

void CheckCuda(const cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "[preload_smoke] " << expr << " failed: "
            << cudaGetErrorString(status) << " (" << static_cast<int>(status)
            << ")" << std::endl;
  std::exit(1);
}

void CheckTrue(const bool value, const char* message) {
  if (value) {
    return;
  }
  std::cerr << "[preload_smoke] assertion failed: " << message << std::endl;
  std::exit(1);
}

template <typename Fn>
Fn LoadSymbol(const char* name) {
  void* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol == nullptr) {
    std::cerr << "[preload_smoke] missing symbol: " << name
              << " (is libmemsaver_preload.so preloaded?)" << std::endl;
    std::exit(1);
  }
  return reinterpret_cast<Fn>(symbol);
}

bool MaybeSkipNoGpu() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    std::cout << "[preload_smoke] skipped (no CUDA device available)" << std::endl;
    return true;
  }
  return false;
}

uint64_t GetCurrentDeviceMinimumGranularityBytes() {
  constexpr uint64_t kDefaultGranularityBytes = 2ULL * 1024ULL * 1024ULL;
  return kDefaultGranularityBytes;
}

}  // namespace

int main() {
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  using SetInterestingFn = cudaError_t (*)(bool);
  using SetTagFn = cudaError_t (*)(const char*);
  using SetBackupFn = cudaError_t (*)(bool);
  using PauseResumeFn = cudaError_t (*)(const char*);
  using ConfigureArenaFn =
      cudaError_t (*)(const char*, uint64_t, memsaver_arena_create_mode_t);
  using ActivateArenaOffsetsFn =
      cudaError_t (*)(const char*, const uint64_t*, uint64_t, uint64_t);
  using DeactivateArenaOffsetsFn =
      cudaError_t (*)(const char*, const uint64_t*, uint64_t, uint64_t);

  const auto set_interesting =
      LoadSymbol<SetInterestingFn>("memsaver_preload_set_interesting_region");
  const auto set_tag =
      LoadSymbol<SetTagFn>("memsaver_preload_set_current_tag");
  const auto set_backup =
      LoadSymbol<SetBackupFn>("memsaver_preload_set_enable_cpu_backup");
  const auto pause = LoadSymbol<PauseResumeFn>("memsaver_preload_pause");
  const auto resume = LoadSymbol<PauseResumeFn>("memsaver_preload_resume");
  const auto configure_arena =
      LoadSymbol<ConfigureArenaFn>("memsaver_preload_configure_arena");
  const auto activate_arena_offsets =
      LoadSymbol<ActivateArenaOffsetsFn>("memsaver_preload_activate_arena_offsets");
  const auto deactivate_arena_offsets =
      LoadSymbol<DeactivateArenaOffsetsFn>("memsaver_preload_deactivate_arena_offsets");

  constexpr size_t kBytes = 1U << 20;
  const uint64_t granularity = GetCurrentDeviceMinimumGranularityBytes();
  const uint64_t arena_capacity = granularity * 16;
  const uint64_t offset0 = 0;

  CheckCuda(
      configure_arena(
          "preload_dynamic", arena_capacity, MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY),
      "preload configure arena_dynamic");
  CheckCuda(
      activate_arena_offsets("preload_dynamic", &offset0, 1, granularity),
      "preload activate arena_dynamic");
  CheckCuda(
      deactivate_arena_offsets("preload_dynamic", &offset0, 1, granularity),
      "preload deactivate arena_dynamic");

  CheckCuda(
      configure_arena(
          "preload_dynamic_full",
          arena_capacity,
          MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED),
      "preload configure arena_dynamic_full");
  const cudaError_t full_mapped_activate_status = activate_arena_offsets(
      "preload_dynamic_full", &offset0, 1, granularity);
  CheckTrue(full_mapped_activate_status == cudaErrorNotSupported,
            "preload activate should return NotSupported on fully-mapped arena");

  CheckCuda(set_interesting(true), "set_interesting(true)");
  CheckCuda(set_tag("preload"), "set_tag(preload)");
  CheckCuda(set_backup(true), "set_backup(true)");

  void* managed_ptr = nullptr;
  CheckCuda(cudaMalloc(&managed_ptr, kBytes), "cudaMalloc managed_ptr");

  std::vector<uint8_t> host(kBytes, 0x5A);
  CheckCuda(cudaMemcpy(managed_ptr, host.data(), host.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D managed_ptr");

  CheckCuda(pause("preload"), "preload pause");
  CheckCuda(resume("preload"), "preload resume");

  std::vector<uint8_t> after(kBytes);
  CheckCuda(cudaMemcpy(after.data(), managed_ptr, after.size(), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H managed_ptr");
  CheckTrue(std::all_of(after.begin(), after.end(),
                        [](uint8_t v) { return v == 0x5A; }),
            "managed allocation content should be preserved with cpu backup");

  CheckCuda(cudaFree(managed_ptr), "cudaFree managed_ptr");

  CheckCuda(set_interesting(false), "set_interesting(false)");

  void* unmanaged_ptr = nullptr;
  CheckCuda(cudaMalloc(&unmanaged_ptr, kBytes), "cudaMalloc unmanaged_ptr");
  CheckCuda(cudaFree(unmanaged_ptr), "cudaFree unmanaged_ptr");

  std::cout << "[preload_smoke] all tests passed" << std::endl;
  return 0;
}
