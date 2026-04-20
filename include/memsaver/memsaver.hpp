#ifndef MEMSAVER_MEMSAVER_HPP_
#define MEMSAVER_MEMSAVER_HPP_

#include <stdexcept>
#include <string>

#include "memsaver/memsaver_c.h"

namespace memsaver {

/**
 * RAII wrapper around the C MemSaver context API.
 *
 * This class owns one memsaver_ctx_t and forwards all operations to C APIs.
 */
class Context {
 public:
  /** Construct a context or throw std::runtime_error on failure. */
  Context() {
    memsaver_ctx_t* created = nullptr;
    const cudaError_t status = memsaver_ctx_create(&created);
    if (status != cudaSuccess) {
      throw std::runtime_error("memsaver_ctx_create failed");
    }
    ctx_ = created;
  }

  ~Context() {
    if (ctx_ != nullptr) {
      (void)memsaver_ctx_destroy(ctx_);
      ctx_ = nullptr;
    }
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  /** Move-construct context ownership. */
  Context(Context&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

  /** Move-assign context ownership. */
  Context& operator=(Context&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (ctx_ != nullptr) {
      (void)memsaver_ctx_destroy(ctx_);
    }
    ctx_ = other.ctx_;
    other.ctx_ = nullptr;
    return *this;
  }

  /** Enable or disable MemSaver-managed allocation for this thread. */
  cudaError_t SetInterestingRegion(bool value) noexcept {
    return memsaver_set_interesting_region(ctx_, value);
  }

  /** Read current interesting-region flag for this thread. */
  cudaError_t GetInterestingRegion(bool* out_value) const noexcept {
    return memsaver_get_interesting_region(ctx_, out_value);
  }

  /** Set allocation tag for this thread. */
  cudaError_t SetCurrentTag(const char* tag) noexcept {
    return memsaver_set_current_tag(ctx_, tag);
  }

  /** Get current allocation tag for this thread. */
  cudaError_t GetCurrentTag(const char** out_tag) const noexcept {
    return memsaver_get_current_tag(ctx_, out_tag);
  }

  /** Enable or disable CPU backup for future regular allocations. */
  cudaError_t SetEnableCpuBackup(bool value) noexcept {
    return memsaver_set_enable_cpu_backup(ctx_, value);
  }

  /** Query CPU backup setting for future regular allocations. */
  cudaError_t GetEnableCpuBackup(bool* out_value) const noexcept {
    return memsaver_get_enable_cpu_backup(ctx_, out_value);
  }

  /** Set allocation mode for future allocations in this thread. */
  cudaError_t SetAllocationMode(memsaver_allocation_mode_t mode) noexcept {
    return memsaver_set_allocation_mode(ctx_, mode);
  }

  /** Get current allocation mode for this thread. */
  cudaError_t GetAllocationMode(memsaver_allocation_mode_t* out_mode) const noexcept {
    return memsaver_get_allocation_mode(ctx_, out_mode);
  }

  /** Allocate GPU memory via MemSaver according to current thread config. */
  cudaError_t Malloc(void** ptr, size_t size) noexcept {
    return memsaver_malloc(ctx_, ptr, size);
  }

  /** Free a pointer allocated by MemSaver (or forwarded cudaMalloc path). */
  cudaError_t Free(void* ptr) noexcept { return memsaver_free(ctx_, ptr); }

  /** Pause managed regular allocations, optionally by tag. */
  cudaError_t Pause(const char* tag_or_null = nullptr) noexcept {
    return memsaver_pause(ctx_, tag_or_null);
  }

  /** Resume paused managed regular allocations, optionally by tag. */
  cudaError_t Resume(const char* tag_or_null = nullptr) noexcept {
    return memsaver_resume(ctx_, tag_or_null);
  }

  /** Configure arena for current device and tag. */
  cudaError_t ConfigureArena(
      const char* tag,
      uint64_t capacity_bytes,
      memsaver_arena_create_mode_t create_mode) noexcept {
    return memsaver_configure_arena(ctx_, tag, capacity_bytes, create_mode);
  }

  /** Reset arena offset to zero for current device and tag. */
  cudaError_t ResetArena(const char* tag) noexcept {
    return memsaver_reset_arena(ctx_, tag);
  }

  /** Activate virtual-only arena offsets with real device mappings. */
  cudaError_t ActivateArenaOffsets(
      const char* tag,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes) noexcept {
    return memsaver_activate_arena_offsets(
        ctx_, tag, offsets, num_offsets, size_bytes);
  }

  /** Deactivate virtual-only arena offsets and map empty handle back. */
  cudaError_t DeactivateArenaOffsets(
      const char* tag,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes) noexcept {
    return memsaver_deactivate_arena_offsets(
        ctx_, tag, offsets, num_offsets, size_bytes);
  }

  /** Set free-memory safety margin in bytes before regular allocations. */
  cudaError_t SetMemoryMarginBytes(uint64_t value) noexcept {
    return memsaver_set_memory_margin_bytes(ctx_, value);
  }

  /** Query CPU backup pointer for a GPU range in paused state. */
  cudaError_t GetCpuBackupPointer(
      const uint8_t* gpu_ptr,
      uint64_t size,
      uint8_t** out_cpu_ptr) noexcept {
    return memsaver_get_cpu_backup_pointer(ctx_, gpu_ptr, size, out_cpu_ptr);
  }

  /** Access underlying raw C context handle. */
  memsaver_ctx_t* Raw() noexcept { return ctx_; }
  /** Access underlying raw C context handle (const overload). */
  const memsaver_ctx_t* Raw() const noexcept { return ctx_; }

 private:
  memsaver_ctx_t* ctx_ = nullptr;
};

}  // namespace memsaver

#endif  // MEMSAVER_MEMSAVER_HPP_
