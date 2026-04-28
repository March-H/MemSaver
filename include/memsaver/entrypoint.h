#ifndef MEMSAVER_ENTRYPOINT_H_
#define MEMSAVER_ENTRYPOINT_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class AllocationKind {
  REGULAR,
  ARENA,
};

class MemSaver {
 public:
  MemSaver();
  MemSaver(const MemSaver&) = delete;
  MemSaver& operator=(const MemSaver&) = delete;
  MemSaver(MemSaver&& other) = delete;
  MemSaver& operator=(MemSaver&& other) = delete;
  ~MemSaver();

  cudaError_t enter_region(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode = AllocationKind::REGULAR);
  cudaError_t leave_region();
  cudaError_t evict_region_pool_from_cache(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode = AllocationKind::REGULAR);

 private:
  struct RegionCacheKey;
  struct RegionCacheKeyHash;
  struct CachedPool;
  struct Impl;

  std::shared_ptr<CachedPool> get_or_create_cached_pool(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode);
  std::shared_ptr<CachedPool> get_cached_pool(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode);

  std::unique_ptr<Impl> impl_;
};

cudaError_t memsaver_malloc(void** ptr, size_t size);
cudaError_t memsaver_free(void* ptr);

void* memsaver_torch_malloc(size_t size, int device, cudaStream_t stream);
void memsaver_torch_free(void* ptr, size_t size, int device, cudaStream_t stream);

cudaError_t memsaver_pause(const char* tag_or_null);
cudaError_t memsaver_resume(const char* tag_or_null);
cudaError_t memsaver_empty_cache();

cudaError_t memsaver_activate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    uint64_t num_offsets,
    uint64_t size_bytes);
cudaError_t memsaver_deactivate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    uint64_t num_offsets,
    uint64_t size_bytes);

cudaError_t memsaver_get_metadata_count_by_tag(
    const char* tag,
    uint64_t* out_count);
cudaError_t memsaver_get_cpu_backup_pointer(
    const uint8_t* gpu_ptr,
    uint64_t size,
    uint8_t** out_cpu_ptr);

#endif
