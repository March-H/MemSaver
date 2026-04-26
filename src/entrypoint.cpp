#include "memsaver/memsaver_c.h"

#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

#include "internal/context_impl.h"
#include "internal/utils.h"

struct memsaver_ctx {
  std::unique_ptr<memsaver::internal::ContextImpl> impl;
};

namespace {

using memsaver::internal::RuntimeConfig;

struct ThreadConfig {
  bool initialized = false;
  bool interesting_region = false;
  bool enable_cpu_backup = false;
  memsaver_allocation_mode_t allocation_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
  std::string current_tag = "default";
};

thread_local std::unordered_map<memsaver_ctx_t*, ThreadConfig> g_thread_configs;

// Validate public allocation mode enum value.
cudaError_t EnsureValidMode(const memsaver_allocation_mode_t mode) {
  const bool valid = mode == MEMSAVER_ALLOCATION_MODE_NORMAL ||
                     mode == MEMSAVER_ALLOCATION_MODE_ARENA;
  RETURN_IF_FALSE(valid, cudaErrorInvalidValue,
                           "EnsureValidMode: invalid allocation mode");
  return cudaSuccess;
}

// Initialize per-thread configuration on first use for a context.
cudaError_t InitializeThreadConfig(ThreadConfig* config) {
  RETURN_IF_FALSE(config != nullptr, cudaErrorInvalidValue,
                           "InitializeThreadConfig: config should not be null");

  RETURN_IF_CUDA_ERROR(
      memsaver::internal::utils::ReadBoolEnvVar(
          "MEMSAVER_ENABLE", false, &config->interesting_region));
  config->enable_cpu_backup = false;
  config->allocation_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
  config->current_tag = "default";
  config->initialized = true;
  return cudaSuccess;
}

// Resolve internal implementation from opaque C handle.
cudaError_t GetImpl(
    memsaver_ctx_t* ctx,
    memsaver::internal::ContextImpl** out_impl) {
  RETURN_IF_FALSE(ctx != nullptr, cudaErrorInvalidValue,
                           "GetImpl: ctx should not be null");
  RETURN_IF_FALSE(out_impl != nullptr, cudaErrorInvalidValue,
                           "GetImpl: out_impl should not be null");
  RETURN_IF_FALSE(ctx->impl != nullptr, cudaErrorInvalidValue,
                           "GetImpl: ctx->impl should not be null");

  *out_impl = ctx->impl.get();
  return cudaSuccess;
}

// Resolve or lazily initialize thread-local config bound to a context.
cudaError_t GetThreadConfig(memsaver_ctx_t* ctx, ThreadConfig** out_config) {
  RETURN_IF_FALSE(out_config != nullptr, cudaErrorInvalidValue,
                           "GetThreadConfig: out_config should not be null");

  memsaver::internal::ContextImpl* unused = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &unused));

  auto [it, inserted] = g_thread_configs.try_emplace(ctx);
  if (inserted) {
    RETURN_IF_CUDA_ERROR(InitializeThreadConfig(&it->second));
  }

  if (!it->second.initialized) {
    RETURN_IF_CUDA_ERROR(InitializeThreadConfig(&it->second));
  }

  *out_config = &it->second;
  return cudaSuccess;
}

cudaError_t BuildRuntimeConfig(
    memsaver_ctx_t* ctx,
    RuntimeConfig* out_runtime_config) {
  RETURN_IF_FALSE(out_runtime_config != nullptr, cudaErrorInvalidValue,
                           "BuildRuntimeConfig: out_runtime_config should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));

  out_runtime_config->interesting_region = config->interesting_region;
  out_runtime_config->enable_cpu_backup = config->enable_cpu_backup;
  out_runtime_config->tag = config->current_tag;
  out_runtime_config->allocation_mode = config->allocation_mode;
  return cudaSuccess;
}

}  // namespace

// Create a MemSaver context used by C APIs.
extern "C" cudaError_t memsaver_ctx_create(memsaver_ctx_t** out_ctx) {
  RETURN_IF_FALSE(out_ctx != nullptr, cudaErrorInvalidValue,
                           "memsaver_ctx_create: out_ctx should not be null");

  auto* raw_ctx = new (std::nothrow) memsaver_ctx();
  RETURN_IF_FALSE(raw_ctx != nullptr, cudaErrorMemoryAllocation,
                           "memsaver_ctx_create: failed to allocate memsaver_ctx");

  try {
    raw_ctx->impl =
        std::make_unique<memsaver::internal::ContextImpl>(false);
  } catch (...) {
    delete raw_ctx;
    return cudaErrorMemoryAllocation;
  }

  *out_ctx = raw_ctx;
  return cudaSuccess;
}

// Destroy a context created by memsaver_ctx_create.
extern "C" cudaError_t memsaver_ctx_destroy(memsaver_ctx_t* ctx) {
  if (ctx == nullptr) {
    return cudaSuccess;
  }

  g_thread_configs.erase(ctx);
  delete ctx;
  return cudaSuccess;
}

// Set whether allocations in this thread are routed through MemSaver.
extern "C" cudaError_t memsaver_set_interesting_region(
    memsaver_ctx_t* ctx,
    const bool value) {
  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->interesting_region = value;
  return cudaSuccess;
}

// Get current interesting-region state for this thread.
extern "C" cudaError_t memsaver_get_interesting_region(
    memsaver_ctx_t* ctx,
    bool* out_value) {
  RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_interesting_region: out_value should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_value = config->interesting_region;
  return cudaSuccess;
}

// Set current tag used by subsequent allocations in this thread.
extern "C" cudaError_t memsaver_set_current_tag(
    memsaver_ctx_t* ctx,
    const char* tag) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_set_current_tag: tag should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->current_tag = tag;
  return cudaSuccess;
}

// Get current tag used by this thread.
extern "C" cudaError_t memsaver_get_current_tag(
    memsaver_ctx_t* ctx,
    const char** out_tag) {
  RETURN_IF_FALSE(out_tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_current_tag: out_tag should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_tag = config->current_tag.c_str();
  return cudaSuccess;
}

// Enable/disable CPU backup for future regular allocations.
extern "C" cudaError_t memsaver_set_enable_cpu_backup(
    memsaver_ctx_t* ctx,
    const bool value) {
  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->enable_cpu_backup = value;
  return cudaSuccess;
}

// Query CPU-backup setting for this thread.
extern "C" cudaError_t memsaver_get_enable_cpu_backup(
    memsaver_ctx_t* ctx,
    bool* out_value) {
  RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_enable_cpu_backup: out_value should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_value = config->enable_cpu_backup;
  return cudaSuccess;
}

// Set allocation mode for future allocations in this thread.
extern "C" cudaError_t memsaver_set_allocation_mode(
    memsaver_ctx_t* ctx,
    const memsaver_allocation_mode_t mode) {
  RETURN_IF_CUDA_ERROR(EnsureValidMode(mode));

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->allocation_mode = mode;
  return cudaSuccess;
}

// Get allocation mode currently selected for this thread.
extern "C" cudaError_t memsaver_get_allocation_mode(
    memsaver_ctx_t* ctx,
    memsaver_allocation_mode_t* out_mode) {
  RETURN_IF_FALSE(out_mode != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_allocation_mode: out_mode should not be null");

  ThreadConfig* config = nullptr;
  RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_mode = config->allocation_mode;
  return cudaSuccess;
}

// Allocate memory according to current thread config and device.
extern "C" cudaError_t memsaver_malloc(
    memsaver_ctx_t* ctx,
    void** ptr,
    const size_t size) {
  RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                           "memsaver_malloc: ptr should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  RuntimeConfig runtime_config;
  RETURN_IF_CUDA_ERROR(BuildRuntimeConfig(ctx, &runtime_config));

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(
      memsaver::internal::utils::GetCurrentCudaDevice(&device));

  return impl->Malloc(ptr, device, size, runtime_config);
}

// Free pointer through MemSaver metadata or fallback to cudaFree.
extern "C" cudaError_t memsaver_free(memsaver_ctx_t* ctx, void* ptr) {
  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  return impl->Free(ptr);
}

// Pause managed regular allocations, optionally by tag filter.
extern "C" cudaError_t memsaver_pause(
    memsaver_ctx_t* ctx,
    const char* tag_or_null) {
  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return impl->Pause(tag);
}

// Resume paused allocations, optionally by tag filter.
extern "C" cudaError_t memsaver_resume(
    memsaver_ctx_t* ctx,
    const char* tag_or_null) {
  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return impl->Resume(tag);
}

// Activate offset ranges for a virtual-only arena on current device.
extern "C" cudaError_t memsaver_activate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_activate_arena_offsets: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(
      memsaver::internal::utils::GetCurrentCudaDevice(&device));

  return impl->ActivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Deactivate offset ranges for a virtual-only arena on current device.
extern "C" cudaError_t memsaver_deactivate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_deactivate_arena_offsets: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(
      memsaver::internal::utils::GetCurrentCudaDevice(&device));

  return impl->DeactivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Query count of live metadata entries that match a tag exactly.
extern "C" cudaError_t memsaver_get_metadata_count_by_tag(
    memsaver_ctx_t* ctx,
    const char* tag,
    uint64_t* out_count) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_metadata_count_by_tag: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  return impl->GetMetadataCountByTag(tag, out_count);
}

// Return CPU backup pointer for a queried GPU range when paused.
extern "C" cudaError_t memsaver_get_cpu_backup_pointer(
    memsaver_ctx_t* ctx,
    const uint8_t* gpu_ptr,
    const uint64_t size,
    uint8_t** out_cpu_ptr) {
  memsaver::internal::ContextImpl* impl = nullptr;
  RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  return impl->GetCpuBackupPointer(gpu_ptr, size, out_cpu_ptr);
}
