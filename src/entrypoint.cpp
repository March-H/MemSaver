#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <optional>

#include "internal/context_impl.h"
#include "internal/utils.h"

class ThreadLocalConfig {
public:
    std::string current_tag_ = "";

    bool is_interesting_region() {
        return is_interesting_region_;
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        return enable_cpu_backup_;
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

    void set_allocation_mode(AllocationKind mode) {
        allocation_mode_ = mode;
    }

    AllocationKind get_allocation_mode() {
        return allocation_mode_;
    }

private:
    bool is_interesting_region_ = false;
    bool enable_cpu_backup_ = false;
    AllocationKind allocation_mode_ = AllocationKind::REGULAR;
};
static thread_local ThreadLocalConfig thread_local_config;

// Validate public allocation mode enum value.
cudaError_t EnsureValidMode(const AllocationKind mode) {
  const bool valid = mode == AllocationKind::REGULAR ||
                     mode == AllocationKind::ARENA;
  RETURN_IF_FALSE(valid, cudaErrorInvalidValue,
                           "EnsureValidMode: invalid allocation mode");
  return cudaSuccess;
}

cudaError_t BuildRuntimeConfig(RuntimeConfig* runtime_config) {
  RETURN_IF_FALSE(runtime_config != nullptr, cudaErrorInvalidValue,
                           "BuildRuntimeConfig: runtime_config should not be null");
  runtime_config->interesting_region = thread_local_config.is_interesting_region();
  runtime_config->enable_cpu_backup = thread_local_config.enable_cpu_backup();
  runtime_config->tag = thread_local_config.current_tag_;
  runtime_config->allocation_mode = thread_local_config.get_allocation_mode();
  RETURN_IF_CUDA_ERROR(EnsureValidMode(runtime_config->allocation_mode));
  return cudaSuccess;
}

// Allocate memory according to current thread config and device.
cudaError_t memsaver_malloc(
    void** ptr,
    const size_t size) {
  RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                           "memsaver_malloc: ptr should not be null");
  if (!thread_local_config.is_interesting_region()) {
    LOGE("it only work with custom mem_pool allocator, it should not happend");
    return cudaErrorInvalidValue;
  }

  RuntimeConfig runtime_config;
  RETURN_IF_CUDA_ERROR(BuildRuntimeConfig(&runtime_config));

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().Malloc(ptr, device, size, runtime_config);
}

// Free pointer through MemSaver metadata or fallback to cudaFree.
cudaError_t memsaver_free(void* ptr) {
  return ContextImpl::instance().Free(ptr);
}

// Pause managed regular allocations, optionally by tag filter.
cudaError_t memsaver_pause(
    const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return ContextImpl::instance().Pause(tag);
}

// Resume paused allocations, optionally by tag filter.
cudaError_t memsaver_resume(
    const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return ContextImpl::instance().Resume(tag);
}

// Activate offset ranges for a virtual-only arena on current device.
cudaError_t memsaver_activate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_activate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().ActivateArenaOffsets(
      tag, device, offsets, num_offsets, size_bytes);
}

// Deactivate offset ranges for a virtual-only arena on current device.
cudaError_t memsaver_deactivate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_deactivate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().DeactivateArenaOffsets(
      tag, device, offsets, num_offsets, size_bytes);
}

// Query count of live metadata entries that match a tag exactly.
cudaError_t memsaver_get_metadata_count_by_tag(
    const char* tag,
    uint64_t* out_count) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_metadata_count_by_tag: tag should not be null");

  return ContextImpl::instance().GetMetadataCountByTag(tag, out_count);
}

// Return CPU backup pointer for a queried GPU range when paused.
cudaError_t memsaver_get_cpu_backup_pointer(
    const uint8_t* gpu_ptr,
    const uint64_t size,
    uint8_t** out_cpu_ptr) {
  return ContextImpl::instance().GetCpuBackupPointer(gpu_ptr, size, out_cpu_ptr);
}
