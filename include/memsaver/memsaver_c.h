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
  /** Allocate one arena-style segment per allocation request. */
  MEMSAVER_ALLOCATION_MODE_ARENA = 1,
} memsaver_allocation_mode_t;

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
 * Activate real device mappings for offset ranges in a virtual-only arena.
 *
 * Each range is [offsets[i], offsets[i] + size_bytes) relative to arena base.
 * Offsets are applied in order and the function returns on the first error.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @param offsets Array of byte offsets relative to arena base.
 * @param num_offsets Number of offsets in the array.
 * @param size_bytes Range size for each offset.
 * @return cudaSuccess, cudaErrorInvalidValue, or CUDA errors.
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
 * Offsets are processed in order and the function returns on the first error.
 *
 * @param ctx MemSaver context.
 * @param tag Non-empty arena tag.
 * @param offsets Array of byte offsets relative to arena base.
 * @param num_offsets Number of offsets in the array.
 * @param size_bytes Range size for each offset.
 * @return cudaSuccess, cudaErrorInvalidValue, or CUDA errors.
 */
cudaError_t memsaver_deactivate_arena_offsets(
    memsaver_ctx_t* ctx,
    const char* tag,
    const uint64_t* offsets,
    uint64_t num_offsets,
    uint64_t size_bytes);

/**
 * Count live allocation metadata entries matching a tag.
 *
 * This counts current entries tracked in MemSaver internal metadata map.
 *
 * @param ctx MemSaver context.
 * @param tag Tag to match exactly.
 * @param out_count Output count.
 * @return cudaSuccess or cudaErrorInvalidValue for invalid arguments.
 */
cudaError_t memsaver_get_metadata_count_by_tag(
    memsaver_ctx_t* ctx,
    const char* tag,
    uint64_t* out_count);

/**
 * Query host backup address for a GPU subrange.
 *
 * This API is for observability/integration checks (including tests), not for
 * normal allocation/free flow control.
 *
 * Query range is [gpu_ptr, gpu_ptr + size) and must be fully covered by one
 * managed REGULAR allocation.
 *
 * On success:
 * - active allocation: *out_cpu_ptr = NULL
 * - paused allocation with CPU backup: *out_cpu_ptr = backup_base + offset
 *
 * @param ctx MemSaver context.
 * @param gpu_ptr Start address of query range on GPU.
 * @param size Query size in bytes.
 * @param out_cpu_ptr Output host pointer (or NULL for active allocation).
 * @return cudaSuccess on valid lookup; cudaErrorInvalidValue otherwise.
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
