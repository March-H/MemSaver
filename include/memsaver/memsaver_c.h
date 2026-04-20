#ifndef MEMSAVER_MEMSAVER_C_H_
#define MEMSAVER_MEMSAVER_C_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque MemSaver context handle. */
typedef struct memsaver_ctx memsaver_ctx_t;

/** Allocation strategy used for allocations inside an interesting region. */
typedef enum memsaver_allocation_mode {
  /** Allocate regular VMM-backed blocks with per-allocation metadata. */
  MEMSAVER_ALLOCATION_MODE_NORMAL = 0,
  /** Allocate from a pre-configured per-(device, tag) arena. */
  MEMSAVER_ALLOCATION_MODE_ARENA = 1,
} memsaver_allocation_mode_t;

/** Arena creation strategy used by memsaver_configure_arena. */
typedef enum memsaver_arena_create_mode {
  /** Reserve VA and immediately map one full-size physical handle. */
  MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED = 0,
  /** Reserve VA only and prepare shared minimum-granularity handle lazily. */
  MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY = 1,
} memsaver_arena_create_mode_t;

/**
 * Create a MemSaver context.
 *
 * @param out_ctx Output pointer receiving the created context.
 * @return cudaSuccess on success; cudaErrorInvalidValue if out_ctx is null;
 *         cudaErrorMemoryAllocation if allocation fails.
 */
cudaError_t memsaver_ctx_create(memsaver_ctx_t** out_ctx);

/**
 * Destroy a MemSaver context.
 *
 * Passing null is allowed and treated as a no-op.
 *
 * @param ctx Context created by memsaver_ctx_create.
 * @return cudaSuccess.
 */
cudaError_t memsaver_ctx_destroy(memsaver_ctx_t* ctx);

/**
 * Enable or disable MemSaver-managed allocation for the calling thread.
 *
 * When disabled, memsaver_malloc falls back to regular cudaMalloc behavior.
 *
 * @param ctx MemSaver context.
 * @param value True to enable managed allocation for this thread.
 * @return cudaSuccess or an error if ctx is invalid.
 */
cudaError_t memsaver_set_interesting_region(memsaver_ctx_t* ctx, bool value);

/**
 * Query whether MemSaver-managed allocation is enabled for the calling thread.
 *
 * @param ctx MemSaver context.
 * @param out_value Output boolean.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_get_interesting_region(
    memsaver_ctx_t* ctx,
    bool* out_value);

/**
 * Set current allocation tag for the calling thread.
 *
 * @param ctx MemSaver context.
 * @param tag Non-null UTF-8 tag string.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_set_current_tag(memsaver_ctx_t* ctx, const char* tag);

/**
 * Get current allocation tag for the calling thread.
 *
 * Returned pointer remains valid until the tag is changed again on the same
 * thread for the same context.
 *
 * @param ctx MemSaver context.
 * @param out_tag Output pointer to internal tag string.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_get_current_tag(
    memsaver_ctx_t* ctx,
    const char** out_tag);

/**
 * Enable or disable CPU backup on pause for future regular allocations.
 *
 * @param ctx MemSaver context.
 * @param value True to copy data to pinned host memory during pause.
 * @return cudaSuccess or an error if ctx is invalid.
 */
cudaError_t memsaver_set_enable_cpu_backup(memsaver_ctx_t* ctx, bool value);

/**
 * Query whether CPU backup is enabled for future regular allocations.
 *
 * @param ctx MemSaver context.
 * @param out_value Output boolean.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_get_enable_cpu_backup(
    memsaver_ctx_t* ctx,
    bool* out_value);

/**
 * Set allocation mode for future allocations in the calling thread.
 *
 * @param ctx MemSaver context.
 * @param mode Allocation mode.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid mode/arguments.
 */
cudaError_t memsaver_set_allocation_mode(
    memsaver_ctx_t* ctx,
    memsaver_allocation_mode_t mode);

/**
 * Get allocation mode used by future allocations in the calling thread.
 *
 * @param ctx MemSaver context.
 * @param out_mode Output allocation mode.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_get_allocation_mode(
    memsaver_ctx_t* ctx,
    memsaver_allocation_mode_t* out_mode);

/**
 * Allocate device memory using current thread config.
 *
 * If interesting_region is disabled, this delegates to regular cudaMalloc.
 *
 * @param ctx MemSaver context.
 * @param ptr Output device pointer.
 * @param size Allocation size in bytes.
 * @return cudaSuccess or CUDA allocation/validation errors.
 */
cudaError_t memsaver_malloc(memsaver_ctx_t* ctx, void** ptr, size_t size);

/**
 * Free memory allocated by memsaver_malloc or fallback-cudaMalloc.
 *
 * For pointers not managed by MemSaver metadata, this forwards to cudaFree.
 *
 * @param ctx MemSaver context.
 * @param ptr Device pointer to free. Null is accepted.
 * @return cudaSuccess or CUDA errors.
 */
cudaError_t memsaver_free(memsaver_ctx_t* ctx, void* ptr);

/**
 * Pause managed regular allocations, optionally filtered by tag.
 *
 * Passing null pauses all tags.
 *
 * @param ctx MemSaver context.
 * @param tag_or_null Optional tag filter.
 * @return cudaSuccess or CUDA/validation errors.
 */
cudaError_t memsaver_pause(memsaver_ctx_t* ctx, const char* tag_or_null);

/**
 * Resume paused managed regular allocations, optionally filtered by tag.
 *
 * Passing null resumes all tags.
 *
 * @param ctx MemSaver context.
 * @param tag_or_null Optional tag filter.
 * @return cudaSuccess or CUDA/validation errors.
 */
cudaError_t memsaver_resume(memsaver_ctx_t* ctx, const char* tag_or_null);

/**
 * Configure or resize an arena for current device and tag.
 *
 * Existing arena can be reconfigured only when it has no live allocations.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @param capacity_bytes Arena capacity in bytes.
 * @param create_mode Arena creation mode.
 * @return cudaSuccess, cudaErrorInvalidValue, or CUDA allocation errors.
 */
cudaError_t memsaver_configure_arena(
    memsaver_ctx_t* ctx,
    const char* tag,
    uint64_t capacity_bytes,
    memsaver_arena_create_mode_t create_mode);

/**
 * Reset arena offset to zero for current device and tag.
 *
 * Reset is allowed only when the arena has no live allocations.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @return cudaSuccess or cudaErrorInvalidValue.
 */
cudaError_t memsaver_reset_arena(memsaver_ctx_t* ctx, const char* tag);

/**
 * Activate real device mappings for offset ranges in a virtual-only arena.
 *
 * Each range is [offsets[i], offsets[i] + size_bytes) relative to arena base.
 * All offsets and size_bytes must be aligned to the arena minimum granularity.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @param offsets Array of byte offsets relative to arena base.
 * @param num_offsets Number of offsets in the array.
 * @param size_bytes Range size for each offset.
 * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, or CUDA errors.
 */
cudaError_t memsaver_activate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    uint64_t num_offsets,
    uint64_t size_bytes);

/**
 * Deactivate real mappings and map shared empty handle back for offset ranges.
 *
 * Each requested range must exactly match an active binding created by
 * memsaver_activate_arena_offsets.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @param offsets Array of byte offsets relative to arena base.
 * @param num_offsets Number of offsets in the array.
 * @param size_bytes Range size for each offset.
 * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, or CUDA errors.
 */
cudaError_t memsaver_deactivate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    uint64_t num_offsets,
    uint64_t size_bytes);

/**
 * Set safety margin in bytes required to remain free before regular allocate.
 *
 * If free memory would drop below margin, allocation returns OOM.
 *
 * @param ctx MemSaver context.
 * @param value Required free-memory margin in bytes.
 * @return cudaSuccess or an error if ctx is invalid.
 */
cudaError_t memsaver_set_memory_margin_bytes(memsaver_ctx_t* ctx, uint64_t value);

/**
 * Query CPU backup pointer corresponding to a managed GPU memory range.
 *
 * If allocation is active, out_cpu_ptr is set to null and cudaSuccess is
 * returned. If allocation is paused with backup, out_cpu_ptr points to host
 * backup at matching offset.
 *
 * @param ctx MemSaver context.
 * @param gpu_ptr Query GPU pointer.
 * @param size Query size in bytes.
 * @param out_cpu_ptr Output host pointer or null.
 * @return cudaSuccess or cudaErrorInvalidValue when lookup is invalid.
 */
cudaError_t memsaver_get_cpu_backup_pointer(
    memsaver_ctx_t* ctx,
    const uint8_t* gpu_ptr,
    uint64_t size,
    uint8_t** out_cpu_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEMSAVER_MEMSAVER_C_H_
