#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "internal/common.h"
#include "internal/context_impl.h"
#include "internal/cuda_utils.h"
#include "internal/original_cuda_api.h"
#include "memsaver/memsaver_c.h"

namespace {

using memsaver::internal::ContextImpl;
using memsaver::internal::RuntimeConfig;

// Parse bool-like env values; return default if key is missing.
bool ParseBoolOrDefault(const char* name, const bool default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }

  const std::string value(raw);
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
      value == "YES") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "no" ||
      value == "NO") {
    return false;
  }

  const std::string message =
      std::string("Unsupported bool env value: ") + name + "=" + value;
  (void)memsaver::internal::Ensure(
      false, cudaErrorInvalidValue, message.c_str(), __FILE__, __func__, __LINE__);
  return default_value;
}

// Parse initial allocation mode from environment with fallback.
memsaver_allocation_mode_t ParseAllocationModeOrDefault() {
  const char* raw = std::getenv("MEMSAVER_INIT_ALLOCATION_MODE");
  if (raw == nullptr) {
    return MEMSAVER_ALLOCATION_MODE_NORMAL;
  }

  const std::string value(raw);
  if (value == "normal") {
    return MEMSAVER_ALLOCATION_MODE_NORMAL;
  }
  if (value == "arena") {
    return MEMSAVER_ALLOCATION_MODE_ARENA;
  }

  const std::string message =
      std::string("Unsupported allocation mode env value: MEMSAVER_INIT_ALLOCATION_MODE=") +
      value;
  (void)memsaver::internal::Ensure(
      false, cudaErrorInvalidValue, message.c_str(), __FILE__, __func__, __LINE__);
  return MEMSAVER_ALLOCATION_MODE_NORMAL;
}

// Validate allocation mode enum passed by preload control APIs.
bool IsValidAllocationMode(const memsaver_allocation_mode_t mode) {
  return mode == MEMSAVER_ALLOCATION_MODE_NORMAL ||
         mode == MEMSAVER_ALLOCATION_MODE_ARENA;
}

bool IsValidArenaCreateMode(const memsaver_arena_create_mode_t create_mode) {
  return create_mode == MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED ||
         create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY;
}

class ThreadLocalConfig {
 public:
  std::string current_tag = "default";

  // Lazily read MEMSAVER_INIT_ENABLE on first access.
  bool interesting_region() {
    if (!interesting_region_.has_value()) {
      interesting_region_ = ParseBoolOrDefault("MEMSAVER_INIT_ENABLE", false);
    }
    return *interesting_region_;
  }

  void set_interesting_region(const bool value) { interesting_region_ = value; }

  // Lazily read MEMSAVER_INIT_ENABLE_CPU_BACKUP on first access.
  bool enable_cpu_backup() {
    if (!enable_cpu_backup_.has_value()) {
      enable_cpu_backup_ =
          ParseBoolOrDefault("MEMSAVER_INIT_ENABLE_CPU_BACKUP", false);
    }
    return *enable_cpu_backup_;
  }

  void set_enable_cpu_backup(const bool value) { enable_cpu_backup_ = value; }

  // Lazily read MEMSAVER_INIT_ALLOCATION_MODE on first access.
  memsaver_allocation_mode_t allocation_mode() {
    if (!allocation_mode_.has_value()) {
      allocation_mode_ = ParseAllocationModeOrDefault();
    }
    return *allocation_mode_;
  }

  // Override allocation mode for current thread.
  cudaError_t set_allocation_mode(const memsaver_allocation_mode_t mode) {
    MEMSAVER_RETURN_IF_FALSE(IsValidAllocationMode(mode), cudaErrorInvalidValue,
                             "set_allocation_mode: invalid mode");
    allocation_mode_ = mode;
    return cudaSuccess;
  }

 private:
  std::optional<bool> interesting_region_;
  std::optional<bool> enable_cpu_backup_;
  std::optional<memsaver_allocation_mode_t> allocation_mode_;
};

thread_local ThreadLocalConfig g_thread_config;

// Process-global allocator context used by preload mode.
ContextImpl& GlobalContext() {
  static ContextImpl ctx(/*use_original_cuda_symbols=*/true);
  return ctx;
}

// Build runtime config snapshot from thread-local preload settings.
RuntimeConfig BuildRuntimeConfig() {
  RuntimeConfig config;
  config.interesting_region = g_thread_config.interesting_region();
  config.enable_cpu_backup = g_thread_config.enable_cpu_backup();
  config.tag = g_thread_config.current_tag;
  config.allocation_mode = g_thread_config.allocation_mode();
  return config;
}

// Resolve current CUDA device for allocation dispatch.
cudaError_t GetCurrentDevice(CUdevice* out_device) {
  return memsaver::internal::cuda_utils::GetCurrentDevice(out_device);
}

}  // namespace

// LD_PRELOAD hook for cudaMalloc.
extern "C" cudaError_t cudaMalloc(void** ptr, const size_t size) {
  if (!g_thread_config.interesting_region()) {
    return memsaver::internal::OriginalCudaApi::Malloc(ptr, size, true);
  }

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetCurrentDevice(&device));
  return GlobalContext().Malloc(ptr, device, size, BuildRuntimeConfig());
}

// LD_PRELOAD hook for cudaFree.
extern "C" cudaError_t cudaFree(void* ptr) { return GlobalContext().Free(ptr); }

// Preload control API: set interesting-region flag for current thread.
extern "C" cudaError_t memsaver_preload_set_interesting_region(const bool value) {
  g_thread_config.set_interesting_region(value);
  return cudaSuccess;
}

// Preload control API: get interesting-region flag for current thread.
extern "C" cudaError_t memsaver_preload_get_interesting_region(bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_get_interesting_region: out_value should not be null");
  *out_value = g_thread_config.interesting_region();
  return cudaSuccess;
}

// Preload control API: set current tag for current thread.
extern "C" cudaError_t memsaver_preload_set_current_tag(const char* tag) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_set_current_tag: tag should not be null");
  g_thread_config.current_tag = tag;
  return cudaSuccess;
}

// Preload control API: get current tag for current thread.
extern "C" cudaError_t memsaver_preload_get_current_tag(const char** out_tag) {
  MEMSAVER_RETURN_IF_FALSE(out_tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_get_current_tag: out_tag should not be null");
  *out_tag = g_thread_config.current_tag.c_str();
  return cudaSuccess;
}

// Preload control API: set CPU-backup behavior for current thread.
extern "C" cudaError_t memsaver_preload_set_enable_cpu_backup(const bool value) {
  g_thread_config.set_enable_cpu_backup(value);
  return cudaSuccess;
}

// Preload control API: get CPU-backup behavior for current thread.
extern "C" cudaError_t memsaver_preload_get_enable_cpu_backup(bool* out_value) {
  MEMSAVER_RETURN_IF_FALSE(out_value != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_get_enable_cpu_backup: out_value should not be null");
  *out_value = g_thread_config.enable_cpu_backup();
  return cudaSuccess;
}

// Preload control API: set allocation mode for current thread.
extern "C" cudaError_t memsaver_preload_set_allocation_mode(
    const memsaver_allocation_mode_t mode) {
  return g_thread_config.set_allocation_mode(mode);
}

// Preload control API: get allocation mode for current thread.
extern "C" cudaError_t memsaver_preload_get_allocation_mode(
    memsaver_allocation_mode_t* out_mode) {
  MEMSAVER_RETURN_IF_FALSE(out_mode != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_get_allocation_mode: out_mode should not be null");
  *out_mode = g_thread_config.allocation_mode();
  return cudaSuccess;
}

// Preload control API: pause managed allocations by optional tag.
extern "C" cudaError_t memsaver_preload_pause(const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return GlobalContext().Pause(tag);
}

// Preload control API: resume paused allocations by optional tag.
extern "C" cudaError_t memsaver_preload_resume(const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return GlobalContext().Resume(tag);
}

// Preload control API: configure arena on current device.
extern "C" cudaError_t memsaver_preload_configure_arena(
    const char* tag,
    const uint64_t capacity_bytes,
    const memsaver_arena_create_mode_t create_mode) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_configure_arena: tag should not be null");
  MEMSAVER_RETURN_IF_FALSE(IsValidArenaCreateMode(create_mode), cudaErrorInvalidValue,
                           "memsaver_preload_configure_arena: create_mode is invalid");

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetCurrentDevice(&device));
  return GlobalContext().ConfigureArena(tag, device, capacity_bytes, create_mode);
}

// Preload control API: reset arena on current device.
extern "C" cudaError_t memsaver_preload_reset_arena(const char* tag) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_reset_arena: tag should not be null");

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetCurrentDevice(&device));
  return GlobalContext().ResetArena(tag, device);
}

// Preload control API: activate offset ranges for a virtual-only arena.
extern "C" cudaError_t memsaver_preload_activate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_activate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetCurrentDevice(&device));
  return GlobalContext().ActivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Preload control API: deactivate offset ranges for a virtual-only arena.
extern "C" cudaError_t memsaver_preload_deactivate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  MEMSAVER_RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                           "memsaver_preload_deactivate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(GetCurrentDevice(&device));
  return GlobalContext().DeactivateArenaOffsets(tag, device, offsets, num_offsets, size_bytes);
}

// Preload control API: set memory margin for regular allocation checks.
extern "C" cudaError_t memsaver_preload_set_memory_margin_bytes(const uint64_t value) {
  GlobalContext().SetMemoryMarginBytes(value);
  return cudaSuccess;
}

// Preload control API: query CPU backup pointer for a paused range.
extern "C" cudaError_t memsaver_preload_get_cpu_backup_pointer(
    const uint8_t* gpu_ptr,
    const uint64_t size,
    uint8_t** out_cpu_ptr) {
  return GlobalContext().GetCpuBackupPointer(gpu_ptr, size, out_cpu_ptr);
}
