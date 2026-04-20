#include "memsaver/memsaver_c.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "internal/common.h"
#include "internal/context_impl.h"
#include "internal/cuda_utils.h"

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

// Parse generic boolean strings used by environment and config paths.
cudaError_t ParseBoolString(const std::string& value, bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "ParseBoolString: out_value should not be null");

  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
      value == "YES") {
    *out_value = true;
    return cudaSuccess;
  }

  if (value == "0" || value == "false" || value == "FALSE" || value == "no" ||
      value == "NO") {
    *out_value = false;
    return cudaSuccess;
  }

  return cudaErrorInvalidValue;
}

// Read bool value from environment variable with fallback default.
cudaError_t ReadBoolEnv(
    const char* name,
    const bool default_value,
    bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(name != nullptr, cudaErrorInvalidValue,
                           "ReadBoolEnv: name should not be null");
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "ReadBoolEnv: out_value should not be null");

  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    *out_value = default_value;
    return cudaSuccess;
  }

  const cudaError_t status = ParseBoolString(raw, out_value);
  if (status != cudaSuccess) {
    const std::string message = std::string("Unsupported bool env value: ") + name +
                                "=" + raw;
    MEMSAVER_RETURN_IF_FALSE(false, cudaErrorInvalidValue, message.c_str());
  }

  return cudaSuccess;
}

// Parse allocation mode from textual representation.
cudaError_t ParseAllocationModeString(
    const std::string& value,
    memsaver_allocation_mode_t* out_mode) {
  MEMSAVER_RETURN_IF_FALSE(out_mode != nullptr, cudaErrorInvalidValue,
                           "ParseAllocationModeString: out_mode should not be null");

  if (value == "normal") {
    *out_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
    return cudaSuccess;
  }
  if (value == "arena") {
    *out_mode = MEMSAVER_ALLOCATION_MODE_ARENA;
    return cudaSuccess;
  }

  return cudaErrorInvalidValue;
}

// Read initial allocation mode from MEMSAVER_INIT_ALLOCATION_MODE.
cudaError_t ReadAllocationModeEnv(memsaver_allocation_mode_t* out_mode) {
  MEMSAVER_RETURN_IF_FALSE(out_mode != nullptr, cudaErrorInvalidValue,
                           "ReadAllocationModeEnv: out_mode should not be null");

  const char* raw = std::getenv("MEMSAVER_INIT_ALLOCATION_MODE");
  if (raw == nullptr) {
    *out_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
    return cudaSuccess;
  }

  const cudaError_t status = ParseAllocationModeString(raw, out_mode);
  if (status != cudaSuccess) {
    const std::string message =
        std::string("Unsupported allocation mode env value: MEMSAVER_INIT_ALLOCATION_MODE=") +
        raw;
    MEMSAVER_RETURN_IF_FALSE(false, cudaErrorInvalidValue, message.c_str());
  }

  return cudaSuccess;
}

// Validate public allocation mode enum value.
cudaError_t EnsureValidMode(const memsaver_allocation_mode_t mode) {
  const bool valid = mode == MEMSAVER_ALLOCATION_MODE_NORMAL ||
                     mode == MEMSAVER_ALLOCATION_MODE_ARENA;
  MEMSAVER_RETURN_IF_FALSE(valid, cudaErrorInvalidValue,
                           "EnsureValidMode: invalid allocation mode");
  return cudaSuccess;
}

// Validate public arena create mode enum value.
cudaError_t EnsureValidArenaCreateMode(const memsaver_arena_create_mode_t create_mode) {
  const bool valid =
      create_mode == MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED ||
      create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY;
  MEMSAVER_RETURN_IF_FALSE(valid, cudaErrorInvalidValue,
                           "EnsureValidArenaCreateMode: invalid arena create mode");
  return cudaSuccess;
}

// Initialize per-thread configuration on first use for a context.
cudaError_t InitializeThreadConfig(ThreadConfig* config) {
  MEMSAVER_RETURN_IF_FALSE(config != nullptr, cudaErrorInvalidValue,
                           "InitializeThreadConfig: config should not be null");

  MEMSAVER_RETURN_IF_CUDA_ERROR(
      ReadBoolEnv("MEMSAVER_INIT_ENABLE", false, &config->interesting_region));
  MEMSAVER_RETURN_IF_CUDA_ERROR(ReadBoolEnv(
      "MEMSAVER_INIT_ENABLE_CPU_BACKUP", false, &config->enable_cpu_backup));
  MEMSAVER_RETURN_IF_CUDA_ERROR(ReadAllocationModeEnv(&config->allocation_mode));

  config->current_tag = "default";
  config->initialized = true;
  return cudaSuccess;
}

// Resolve internal implementation from opaque C handle.
cudaError_t GetImpl(
    memsaver_ctx_t* ctx,
    memsaver::internal::ContextImpl** out_impl) {
  MEMSAVER_RETURN_IF_FALSE(ctx != nullptr, cudaErrorInvalidValue,
                           "GetImpl: ctx should not be null");
  MEMSAVER_RETURN_IF_FALSE(out_impl != nullptr, cudaErrorInvalidValue,
                           "GetImpl: out_impl should not be null");
  MEMSAVER_RETURN_IF_FALSE(ctx->impl != nullptr, cudaErrorInvalidValue,
                           "GetImpl: ctx->impl should not be null");

  *out_impl = ctx->impl.get();
  return cudaSuccess;
}

// Resolve or lazily initialize thread-local config bound to a context.
cudaError_t GetThreadConfig(memsaver_ctx_t* ctx, ThreadConfig** out_config) {
  MEMSAVER_RETURN_IF_FALSE(out_config != nullptr, cudaErrorInvalidValue,
                           "GetThreadConfig: out_config should not be null");

  memsaver::internal::ContextImpl* unused = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &unused));

  auto [it, inserted] = g_thread_configs.try_emplace(ctx);
  if (inserted) {
    MEMSAVER_RETURN_IF_CUDA_ERROR(InitializeThreadConfig(&it->second));
  }

  if (!it->second.initialized) {
    MEMSAVER_RETURN_IF_CUDA_ERROR(InitializeThreadConfig(&it->second));
  }

  *out_config = &it->second;
  return cudaSuccess;
}

}  // namespace

// Create a MemSaver context used by C APIs.
extern "C" cudaError_t memsaver_ctx_create(memsaver_ctx_t** out_ctx) {
  MEMSAVER_RETURN_IF_FALSE(out_ctx != nullptr, cudaErrorInvalidValue,
                           "memsaver_ctx_create: out_ctx should not be null");

  auto* raw_ctx = new (std::nothrow) memsaver_ctx();
  MEMSAVER_RETURN_IF_FALSE(raw_ctx != nullptr, cudaErrorMemoryAllocation,
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
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->interesting_region = value;
  return cudaSuccess;
}

// Get current interesting-region state for this thread.
extern "C" cudaError_t memsaver_get_interesting_region(
    memsaver_ctx_t* ctx,
    bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_interesting_region: out_value should not be null");

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_value = config->interesting_region;
  return cudaSuccess;
}

// Set current tag used by subsequent allocations in this thread.
extern "C" cudaError_t memsaver_set_current_tag(
    memsaver_ctx_t* ctx,
    const char* tag) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_set_current_tag: tag should not be null");

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->current_tag = tag;
  return cudaSuccess;
}

// Get current tag used by this thread.
extern "C" cudaError_t memsaver_get_current_tag(
    memsaver_ctx_t* ctx,
    const char** out_tag) {
  MEMSAVER_RETURN_IF_FALSE(out_tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_current_tag: out_tag should not be null");

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_tag = config->current_tag.c_str();
  return cudaSuccess;
}

// Enable/disable CPU backup for future regular allocations.
extern "C" cudaError_t memsaver_set_enable_cpu_backup(
    memsaver_ctx_t* ctx,
    const bool value) {
  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->enable_cpu_backup = value;
  return cudaSuccess;
}

// Query CPU-backup setting for this thread.
extern "C" cudaError_t memsaver_get_enable_cpu_backup(
    memsaver_ctx_t* ctx,
    bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_enable_cpu_backup: out_value should not be null");

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_value = config->enable_cpu_backup;
  return cudaSuccess;
}

// Set allocation mode for future allocations in this thread.
extern "C" cudaError_t memsaver_set_allocation_mode(
    memsaver_ctx_t* ctx,
    const memsaver_allocation_mode_t mode) {
  MEMSAVER_RETURN_IF_CUDA_ERROR(EnsureValidMode(mode));

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  config->allocation_mode = mode;
  return cudaSuccess;
}

// Get allocation mode currently selected for this thread.
extern "C" cudaError_t memsaver_get_allocation_mode(
    memsaver_ctx_t* ctx,
    memsaver_allocation_mode_t* out_mode) {
  MEMSAVER_RETURN_IF_FALSE(out_mode != nullptr, cudaErrorInvalidValue,
                           "memsaver_get_allocation_mode: out_mode should not be null");

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));
  *out_mode = config->allocation_mode;
  return cudaSuccess;
}

// Allocate memory according to current thread config and device.
extern "C" cudaError_t memsaver_malloc(
    memsaver_ctx_t* ctx,
    void** ptr,
    const size_t size) {
  MEMSAVER_RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                           "memsaver_malloc: ptr should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  ThreadConfig* config = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetThreadConfig(ctx, &config));

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(memsaver::internal::cuda_utils::GetCurrentDevice(&device));

  RuntimeConfig runtime_config;
  runtime_config.interesting_region = config->interesting_region;
  runtime_config.enable_cpu_backup = config->enable_cpu_backup;
  runtime_config.tag = config->current_tag;
  runtime_config.allocation_mode = config->allocation_mode;

  return impl->Malloc(ptr, device, size, runtime_config);
}

// Free pointer through MemSaver metadata or fallback to cudaFree.
extern "C" cudaError_t memsaver_free(memsaver_ctx_t* ctx, void* ptr) {
  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  return impl->Free(ptr);
}

// Pause managed regular allocations, optionally by tag filter.
extern "C" cudaError_t memsaver_pause(
    memsaver_ctx_t* ctx,
    const char* tag_or_null) {
  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return impl->Pause(tag);
}

// Resume paused allocations, optionally by tag filter.
extern "C" cudaError_t memsaver_resume(
    memsaver_ctx_t* ctx,
    const char* tag_or_null) {
  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return impl->Resume(tag);
}

// Configure arena for current device and provided tag.
extern "C" cudaError_t memsaver_configure_arena(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t capacity_bytes,
    const memsaver_arena_create_mode_t create_mode) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_configure_arena: tag should not be null");
  MEMSAVER_RETURN_IF_CUDA_ERROR(EnsureValidArenaCreateMode(create_mode));

  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(memsaver::internal::cuda_utils::GetCurrentDevice(&device));

  return impl->ConfigureArena(tag, device, capacity_bytes, create_mode);
}

// Reset arena for current device and provided tag.
extern "C" cudaError_t memsaver_reset_arena(
    memsaver_ctx_t* ctx,
    const char* tag) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_reset_arena: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(memsaver::internal::cuda_utils::GetCurrentDevice(&device));

  return impl->ResetArena(tag, device);
}

// Activate offset ranges for a virtual-only arena on current device.
extern "C" cudaError_t memsaver_activate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_activate_arena_offsets: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(memsaver::internal::cuda_utils::GetCurrentDevice(&device));

  return impl->ActivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Deactivate offset ranges for a virtual-only arena on current device.
extern "C" cudaError_t memsaver_deactivate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_deactivate_arena_offsets: tag should not be null");

  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(memsaver::internal::cuda_utils::GetCurrentDevice(&device));

  return impl->DeactivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Set free-memory safety margin checked before regular allocation.
extern "C" cudaError_t memsaver_set_memory_margin_bytes(
    memsaver_ctx_t* ctx,
    const uint64_t value) {
  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  impl->SetMemoryMarginBytes(value);
  return cudaSuccess;
}

// Return CPU backup pointer for a queried GPU range when paused.
extern "C" cudaError_t memsaver_get_cpu_backup_pointer(
    memsaver_ctx_t* ctx,
    const uint8_t* gpu_ptr,
    const uint64_t size,
    uint8_t** out_cpu_ptr) {
  memsaver::internal::ContextImpl* impl = nullptr;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetImpl(ctx, &impl));
  return impl->GetCpuBackupPointer(gpu_ptr, size, out_cpu_ptr);
}
