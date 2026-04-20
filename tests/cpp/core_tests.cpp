#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "memsaver/memsaver_c.h"

namespace {

void CheckCu(const CUresult status, const char* expr) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* error_string = nullptr;
  (void)cuGetErrorString(status, &error_string);
  std::cerr << "[core_tests] " << expr << " failed: "
            << (error_string == nullptr ? "Unknown error" : error_string)
            << " (" << static_cast<int>(status) << ")" << std::endl;
  std::exit(1);
}

void CheckCuda(const cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "[core_tests] " << expr << " failed: "
            << cudaGetErrorString(status) << " (" << static_cast<int>(status)
            << ")" << std::endl;
  std::exit(1);
}

void CheckTrue(const bool value, const char* message) {
  if (value) {
    return;
  }
  std::cerr << "[core_tests] assertion failed: " << message << std::endl;
  std::exit(1);
}

bool MaybeSkipNoGpu() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    std::cout << "[core_tests] skipped (no CUDA device available)" << std::endl;
    return true;
  }
  return false;
}

uint64_t GetCurrentDeviceMinimumGranularityBytes() {
  constexpr uint64_t kDefaultGranularityBytes = 2ULL * 1024ULL * 1024ULL;
  return kDefaultGranularityBytes;
}

void FillDeviceBytes(void* ptr, const size_t size, const uint8_t value) {
  std::vector<uint8_t> host(size, value);
  CheckCuda(cudaMemcpy(ptr, host.data(), size, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
}

std::vector<uint8_t> ReadDeviceBytes(const void* ptr, const size_t size) {
  std::vector<uint8_t> host(size);
  CheckCuda(cudaMemcpy(host.data(), ptr, size, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  return host;
}

void TestBasicPauseResume(memsaver_ctx_t* ctx) {
  CheckCuda(memsaver_set_interesting_region(ctx, true),
            "memsaver_set_interesting_region");
  CheckCuda(memsaver_set_current_tag(ctx, "basic"), "memsaver_set_current_tag");
  CheckCuda(memsaver_set_enable_cpu_backup(ctx, false),
            "memsaver_set_enable_cpu_backup");
  CheckCuda(memsaver_set_allocation_mode(ctx, MEMSAVER_ALLOCATION_MODE_NORMAL),
            "memsaver_set_allocation_mode");

  constexpr size_t kAllocBytes = 1U << 20;
  void* ptr = nullptr;
  CheckCuda(memsaver_malloc(ctx, &ptr, kAllocBytes), "memsaver_malloc");
  CheckTrue(ptr != nullptr, "ptr should not be null");

  const uintptr_t original_address = reinterpret_cast<uintptr_t>(ptr);

  CheckCuda(memsaver_pause(ctx, "basic"), "memsaver_pause");
  CheckCuda(memsaver_resume(ctx, "basic"), "memsaver_resume");

  const uintptr_t resumed_address = reinterpret_cast<uintptr_t>(ptr);
  CheckTrue(original_address == resumed_address,
            "virtual address should remain unchanged after pause/resume");

  FillDeviceBytes(ptr, kAllocBytes, 0x11);
  const auto after_resume = ReadDeviceBytes(ptr, kAllocBytes);
  CheckTrue(std::all_of(after_resume.begin(), after_resume.end(),
                        [](uint8_t value) { return value == 0x11; }),
            "device content write/read failed after resume");

  CheckCuda(memsaver_free(ctx, ptr), "memsaver_free basic");
}

void TestTagAndCpuBackup(memsaver_ctx_t* ctx) {
  constexpr size_t kAllocBytes = 1U << 18;

  CheckCuda(memsaver_set_interesting_region(ctx, true),
            "memsaver_set_interesting_region");
  CheckCuda(memsaver_set_allocation_mode(ctx, MEMSAVER_ALLOCATION_MODE_NORMAL),
            "memsaver_set_allocation_mode");

  CheckCuda(memsaver_set_current_tag(ctx, "with_backup"),
            "memsaver_set_current_tag with_backup");
  CheckCuda(memsaver_set_enable_cpu_backup(ctx, true),
            "memsaver_set_enable_cpu_backup true");
  void* ptr_backup = nullptr;
  CheckCuda(memsaver_malloc(ctx, &ptr_backup, kAllocBytes),
            "memsaver_malloc with_backup");
  FillDeviceBytes(ptr_backup, kAllocBytes, 0x2A);

  CheckCuda(memsaver_set_current_tag(ctx, "without_backup"),
            "memsaver_set_current_tag without_backup");
  CheckCuda(memsaver_set_enable_cpu_backup(ctx, false),
            "memsaver_set_enable_cpu_backup false");
  void* ptr_without_backup = nullptr;
  CheckCuda(memsaver_malloc(ctx, &ptr_without_backup, kAllocBytes),
            "memsaver_malloc without_backup");

  CheckCuda(memsaver_pause(ctx, "with_backup"), "memsaver_pause with_backup");

  uint8_t* cpu_backup_ptr = nullptr;
  CheckCuda(
      memsaver_get_cpu_backup_pointer(
          ctx,
          static_cast<const uint8_t*>(ptr_backup),
          16,
          &cpu_backup_ptr),
      "memsaver_get_cpu_backup_pointer backup");
  CheckTrue(cpu_backup_ptr != nullptr,
            "cpu backup pointer should exist for paused backed allocation");

  uint8_t* cpu_backup_ptr_other = reinterpret_cast<uint8_t*>(0x1);
  CheckCuda(
      memsaver_get_cpu_backup_pointer(
          ctx,
          static_cast<const uint8_t*>(ptr_without_backup),
          16,
          &cpu_backup_ptr_other),
      "memsaver_get_cpu_backup_pointer active");
  CheckTrue(cpu_backup_ptr_other == nullptr,
            "active allocation should not expose cpu backup pointer");

  CheckCuda(memsaver_resume(ctx, "with_backup"), "memsaver_resume with_backup");

  const auto content = ReadDeviceBytes(ptr_backup, kAllocBytes);
  CheckTrue(std::all_of(content.begin(), content.end(),
                        [](uint8_t value) { return value == 0x2A; }),
            "backed allocation content should be restored after resume");

  CheckCuda(memsaver_free(ctx, ptr_backup), "memsaver_free ptr_backup");
  CheckCuda(memsaver_free(ctx, ptr_without_backup),
            "memsaver_free ptr_without_backup");
}

void TestArena(memsaver_ctx_t* ctx) {
  CheckCuda(memsaver_set_interesting_region(ctx, true),
            "memsaver_set_interesting_region");

  CheckCuda(
      memsaver_configure_arena(
          ctx, "arena", 4U << 20, MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED),
      "memsaver_configure_arena");

  CheckCuda(memsaver_set_current_tag(ctx, "arena"),
            "memsaver_set_current_tag arena");
  CheckCuda(memsaver_set_allocation_mode(
                ctx, MEMSAVER_ALLOCATION_MODE_ARENA),
            "memsaver_set_allocation_mode arena");

  void* ptr_round1 = nullptr;
  CheckCuda(memsaver_malloc(ctx, &ptr_round1, 1024),
            "memsaver_malloc arena round1");
  CheckCuda(memsaver_free(ctx, ptr_round1), "memsaver_free arena round1");
  CheckCuda(memsaver_reset_arena(ctx, "arena"),
            "memsaver_reset_arena round1");

  void* ptr_round2 = nullptr;
  CheckCuda(memsaver_malloc(ctx, &ptr_round2, 1024),
            "memsaver_malloc arena round2");
  CheckTrue(ptr_round1 == ptr_round2,
            "arena reset should reuse addresses");

  const cudaError_t reset_with_live_allocs =
      memsaver_reset_arena(ctx, "arena");
  CheckTrue(reset_with_live_allocs == cudaErrorInvalidValue,
            "reset must fail when live allocations exist");

  CheckCuda(memsaver_free(ctx, ptr_round2), "memsaver_free arena round2");
  CheckCuda(memsaver_reset_arena(ctx, "arena"),
            "memsaver_reset_arena final");

  CheckCuda(memsaver_set_current_tag(ctx, "arena_oom"),
            "memsaver_set_current_tag arena_oom");
  CheckCuda(
      memsaver_configure_arena(
          ctx, "arena_oom", 1024, MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED),
      "memsaver_configure_arena arena_oom");

  void* ptr_oom = nullptr;
  const cudaError_t oom_status = memsaver_malloc(ctx, &ptr_oom, 1U << 30);
  CheckTrue(oom_status == cudaErrorMemoryAllocation,
            "arena should report OOM when capacity is exceeded");

  CheckCuda(memsaver_set_current_tag(ctx, "arena_virtual"),
            "memsaver_set_current_tag arena_virtual");
  CheckCuda(
      memsaver_configure_arena(
          ctx,
          "arena_virtual",
          4U << 20,
          MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY),
      "memsaver_configure_arena arena_virtual");

  void* virtual_ptr = nullptr;
  const cudaError_t virtual_malloc_status = memsaver_malloc(ctx, &virtual_ptr, 1024);
  CheckTrue(virtual_malloc_status == cudaErrorNotSupported,
            "virtual-only arena should return NotSupported on arena malloc");

  const cudaError_t virtual_reconfigure_status = memsaver_configure_arena(
      ctx, "arena_virtual", 4U << 20, MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY);
  CheckTrue(virtual_reconfigure_status == cudaErrorNotSupported,
            "virtual-only arena should reject reconfigure");

  CheckCuda(memsaver_reset_arena(ctx, "arena_virtual"),
            "memsaver_reset_arena virtual-only should be no-op");

  CheckCuda(memsaver_set_allocation_mode(ctx, MEMSAVER_ALLOCATION_MODE_NORMAL),
            "memsaver_set_allocation_mode normal");
}

void TestArenaDynamicOffsets(memsaver_ctx_t* ctx) {
  const uint64_t granularity = GetCurrentDeviceMinimumGranularityBytes();
  CheckTrue(granularity > 0, "granularity should be > 0");
  const uint64_t arena_capacity = granularity * 64;
  const uint64_t one_granularity = granularity;

  CheckCuda(
      memsaver_configure_arena(
          ctx,
          "arena_dynamic",
          arena_capacity,
          MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY),
      "memsaver_configure_arena arena_dynamic");
  CheckCuda(
      memsaver_configure_arena(
          ctx,
          "arena_dynamic_full",
          arena_capacity,
          MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED),
      "memsaver_configure_arena arena_dynamic_full");

  const uint64_t zero_offset = 0;
  const cudaError_t full_mapped_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic_full", &zero_offset, 1, one_granularity);
  CheckTrue(full_mapped_status == cudaErrorNotSupported,
            "activate_arena_offsets should return NotSupported on fully-mapped arena");

  const cudaError_t null_tag_status = memsaver_activate_arena_offsets(
      ctx, nullptr, &zero_offset, 1, one_granularity);
  CheckTrue(null_tag_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject null tag");

  const cudaError_t null_offsets_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", nullptr, 1, one_granularity);
  CheckTrue(null_offsets_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject null offsets when num_offsets > 0");

  const cudaError_t zero_size_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", &zero_offset, 1, 0);
  CheckTrue(zero_size_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject size_bytes=0");

  const uint64_t unaligned_offset = 1;
  const cudaError_t unaligned_offset_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", &unaligned_offset, 1, one_granularity);
  CheckTrue(unaligned_offset_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject unaligned offset");

  const cudaError_t unaligned_size_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", &zero_offset, 1, one_granularity + 1);
  CheckTrue(unaligned_size_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject unaligned size_bytes");

  const uint64_t out_of_bounds_offset = arena_capacity;
  const cudaError_t out_of_bounds_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", &out_of_bounds_offset, 1, one_granularity);
  CheckTrue(out_of_bounds_status == cudaErrorInvalidValue,
            "activate_arena_offsets should reject out-of-bounds range");

  uint64_t success_offsets[2] = {0, one_granularity * 4};
  CheckCuda(
      memsaver_activate_arena_offsets(
          ctx, "arena_dynamic", success_offsets, 2, one_granularity),
      "memsaver_activate_arena_offsets success");
  CheckCuda(
      memsaver_deactivate_arena_offsets(
          ctx, "arena_dynamic", success_offsets, 2, one_granularity),
      "memsaver_deactivate_arena_offsets success");

  const cudaError_t deactivate_inactive_status = memsaver_deactivate_arena_offsets(
      ctx, "arena_dynamic", &zero_offset, 1, one_granularity);
  CheckTrue(deactivate_inactive_status == cudaErrorInvalidValue,
            "deactivate_arena_offsets should reject inactive range");

  CheckCuda(
      memsaver_activate_arena_offsets(
          ctx, "arena_dynamic", &zero_offset, 1, one_granularity),
      "memsaver_activate_arena_offsets single");
  CheckCuda(
      memsaver_deactivate_arena_offsets(
          ctx, "arena_dynamic", &zero_offset, 1, one_granularity),
      "memsaver_deactivate_arena_offsets single");
  const cudaError_t deactivate_twice_status = memsaver_deactivate_arena_offsets(
      ctx, "arena_dynamic", &zero_offset, 1, one_granularity);
  CheckTrue(deactivate_twice_status == cudaErrorInvalidValue,
            "deactivate_arena_offsets should reject repeated deactivate");

  uint64_t mixed_offsets[2] = {
      one_granularity * 30,
      out_of_bounds_offset,
  };
  const cudaError_t mixed_batch_status = memsaver_activate_arena_offsets(
      ctx, "arena_dynamic", mixed_offsets, 2, one_granularity);
  CheckTrue(mixed_batch_status == cudaErrorInvalidValue,
            "activate_arena_offsets should fail for mixed valid/invalid batch");

  const uint64_t first_mixed_offset = mixed_offsets[0];
  const cudaError_t mixed_partial_effect_status = memsaver_deactivate_arena_offsets(
      ctx, "arena_dynamic", &first_mixed_offset, 1, one_granularity);
  CheckTrue(mixed_partial_effect_status == cudaSuccess,
            "failed mixed batch should allow partial activation before first error");

  const cudaError_t deactivate_full_mapped_status = memsaver_deactivate_arena_offsets(
      ctx, "arena_dynamic_full", &zero_offset, 1, one_granularity);
  CheckTrue(deactivate_full_mapped_status == cudaErrorNotSupported,
            "deactivate_arena_offsets should return NotSupported on fully-mapped arena");
}

void TestAllocationModeEnvParsing() {
  CheckTrue(setenv("MEMSAVER_INIT_ALLOCATION_MODE", "arena", 1) == 0,
            "setenv arena should succeed");
  memsaver_ctx_t* arena_ctx = nullptr;
  CheckCuda(memsaver_ctx_create(&arena_ctx), "memsaver_ctx_create arena env");
  memsaver_allocation_mode_t arena_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
  CheckCuda(memsaver_get_allocation_mode(arena_ctx, &arena_mode),
            "memsaver_get_allocation_mode arena env");
  CheckTrue(arena_mode == MEMSAVER_ALLOCATION_MODE_ARENA,
            "MEMSAVER_INIT_ALLOCATION_MODE=arena should parse to ARENA");
  CheckCuda(memsaver_ctx_destroy(arena_ctx), "memsaver_ctx_destroy arena env");
  CheckTrue(unsetenv("MEMSAVER_INIT_ALLOCATION_MODE") == 0,
            "unsetenv arena should succeed");

  const pid_t child = fork();
  CheckTrue(child >= 0, "fork should succeed");
  if (child == 0) {
    if (setenv("MEMSAVER_INIT_ALLOCATION_MODE", "arena_legacy", 1) != 0) {
      _exit(2);
    }

    memsaver_ctx_t* legacy_ctx = nullptr;
    const cudaError_t create_status = memsaver_ctx_create(&legacy_ctx);
    if (create_status != cudaSuccess || legacy_ctx == nullptr) {
      _exit(3);
    }

    memsaver_allocation_mode_t legacy_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
    const cudaError_t legacy_status =
        memsaver_get_allocation_mode(legacy_ctx, &legacy_mode);
    (void)memsaver_ctx_destroy(legacy_ctx);

    // With fail-fast enabled, the process aborts before reaching here.
    // With fail-fast disabled, old value should still be rejected explicitly.
    if (legacy_status == cudaErrorInvalidValue) {
      _exit(4);
    }
    if (legacy_status == cudaSuccess) {
      _exit(0);
    }
    _exit(5);
  }

  int wait_status = 0;
  CheckTrue(waitpid(child, &wait_status, 0) == child, "waitpid should succeed");
  const bool child_failed =
      (WIFSIGNALED(wait_status) != 0) ||
      (WIFEXITED(wait_status) != 0 && WEXITSTATUS(wait_status) != 0);
  CheckTrue(child_failed, "legacy arena alias string should be rejected");
}

void TestMultiGpu(memsaver_ctx_t* ctx) {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count < 2) {
    std::cout << "[core_tests] multi-gpu test skipped (<2 GPUs)" << std::endl;
    return;
  }

  CheckCuda(memsaver_set_interesting_region(ctx, true),
            "memsaver_set_interesting_region");
  CheckCuda(memsaver_set_allocation_mode(ctx, MEMSAVER_ALLOCATION_MODE_NORMAL),
            "memsaver_set_allocation_mode");
  CheckCuda(memsaver_set_enable_cpu_backup(ctx, false),
            "memsaver_set_enable_cpu_backup");

  void* ptr0 = nullptr;
  void* ptr1 = nullptr;

  CheckCuda(cudaSetDevice(0), "cudaSetDevice 0");
  CheckCuda(memsaver_set_current_tag(ctx, "gpu0"), "memsaver_set_current_tag gpu0");
  CheckCuda(memsaver_malloc(ctx, &ptr0, 1U << 20), "memsaver_malloc gpu0");

  CheckCuda(cudaSetDevice(1), "cudaSetDevice 1");
  CheckCuda(memsaver_set_current_tag(ctx, "gpu1"), "memsaver_set_current_tag gpu1");
  CheckCuda(memsaver_malloc(ctx, &ptr1, 1U << 20), "memsaver_malloc gpu1");

  CheckCuda(memsaver_pause(ctx, nullptr), "memsaver_pause all");
  CheckCuda(memsaver_resume(ctx, nullptr), "memsaver_resume all");

  CheckCuda(cudaSetDevice(0), "cudaSetDevice 0 for free");
  CheckCuda(memsaver_free(ctx, ptr0), "memsaver_free gpu0");

  CheckCuda(cudaSetDevice(1), "cudaSetDevice 1 for free");
  CheckCuda(memsaver_free(ctx, ptr1), "memsaver_free gpu1");
}

void TestUnknownPointerForwarding(memsaver_ctx_t* ctx) {
  void* raw_ptr = nullptr;
  CheckCuda(cudaMalloc(&raw_ptr, 1U << 16), "cudaMalloc raw_ptr");
  CheckCuda(memsaver_free(ctx, raw_ptr),
            "memsaver_free should forward unknown pointer to cudaFree");
}

void TestConfigGetters(memsaver_ctx_t* ctx) {
  CheckCuda(memsaver_set_interesting_region(ctx, true),
            "memsaver_set_interesting_region");
  bool interesting_region = false;
  CheckCuda(memsaver_get_interesting_region(ctx, &interesting_region),
            "memsaver_get_interesting_region");
  CheckTrue(interesting_region, "interesting_region getter mismatch");

  CheckCuda(memsaver_set_current_tag(ctx, "getter_tag"),
            "memsaver_set_current_tag getter");
  const char* current_tag = nullptr;
  CheckCuda(memsaver_get_current_tag(ctx, &current_tag),
            "memsaver_get_current_tag");
  CheckTrue(current_tag != nullptr && std::string(current_tag) == "getter_tag",
            "current_tag getter mismatch");

  CheckCuda(memsaver_set_enable_cpu_backup(ctx, true),
            "memsaver_set_enable_cpu_backup getter");
  bool enable_cpu_backup = false;
  CheckCuda(memsaver_get_enable_cpu_backup(ctx, &enable_cpu_backup),
            "memsaver_get_enable_cpu_backup");
  CheckTrue(enable_cpu_backup, "enable_cpu_backup getter mismatch");

  CheckCuda(memsaver_set_allocation_mode(
                ctx, MEMSAVER_ALLOCATION_MODE_ARENA),
            "memsaver_set_allocation_mode getter");
  memsaver_allocation_mode_t mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
  CheckCuda(memsaver_get_allocation_mode(ctx, &mode),
            "memsaver_get_allocation_mode");
  CheckTrue(mode == MEMSAVER_ALLOCATION_MODE_ARENA,
            "allocation_mode getter mismatch");

  CheckCuda(memsaver_set_allocation_mode(ctx, MEMSAVER_ALLOCATION_MODE_NORMAL),
            "memsaver_set_allocation_mode reset");
}

}  // namespace

int main() {
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  CheckTrue(unsetenv("MEMSAVER_INIT_ALLOCATION_MODE") == 0,
            "unsetenv allocation mode before tests should succeed");

  memsaver_ctx_t* ctx = nullptr;
  CheckCuda(memsaver_ctx_create(&ctx), "memsaver_ctx_create");
  CheckTrue(ctx != nullptr, "ctx should not be null");

  TestConfigGetters(ctx);
  TestAllocationModeEnvParsing();
  TestBasicPauseResume(ctx);
  TestTagAndCpuBackup(ctx);
  TestArena(ctx);
  TestArenaDynamicOffsets(ctx);
  TestMultiGpu(ctx);
  TestUnknownPointerForwarding(ctx);

  CheckCuda(memsaver_ctx_destroy(ctx), "memsaver_ctx_destroy");
  std::cout << "[core_tests] all tests passed" << std::endl;
  return 0;
}
