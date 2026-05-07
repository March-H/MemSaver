#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

struct AllocatorStats {
  std::size_t allocated_bytes = 0;
  std::size_t reserved_bytes = 0;
  std::size_t max_allocated_bytes = 0;
  std::size_t max_reserved_bytes = 0;
  std::size_t num_alloc_retries = 0;
  std::size_t num_ooms = 0;
};

using CudaMallocFn = std::function<cudaError_t(void**, std::size_t)>;
using CudaFreeFn = std::function<cudaError_t(void*)>;

class MemPool;
class DeviceCachingAllocator;

class CachingAllocator {
 public:
  using CudaMallocFn = ::CudaMallocFn;
  using CudaFreeFn = ::CudaFreeFn;

  static CachingAllocator& instance();

  std::shared_ptr<MemPool> create_mem_pool(
      std::size_t mem_pool_id,
      CudaMallocFn custom_malloc = nullptr,
      CudaFreeFn custom_free = nullptr,
      int block_pool_type = 0);
  void begin_allocate_to_pool(
      int device,
      std::size_t mem_pool_id,
      std::function<bool(cudaStream_t)> filter);
  void end_allocate_to_pool(int device, std::size_t mem_pool_id);
  void release_pool(int device, std::size_t mem_pool_id);
  void* malloc(std::size_t size, int device = 0, cudaStream_t stream = nullptr);
  void free(void* ptr);
  void record_stream(void* ptr, cudaStream_t stream);
  void empty_cache();
  void empty_cache(int device);

 private:
  struct AllocationContext {
    std::shared_ptr<MemPool> mem_pool;
    DeviceCachingAllocator* device_allocator;
  };

  CachingAllocator();
  ~CachingAllocator();

  CachingAllocator(const CachingAllocator&) = delete;
  CachingAllocator& operator=(const CachingAllocator&) = delete;

  std::shared_ptr<MemPool> mem_pool_for_stream(cudaStream_t stream);

  mutable std::mutex mutex_;
  std::shared_ptr<MemPool> default_mem_pool_;
  std::unordered_map<std::size_t, std::shared_ptr<MemPool>> mem_pools_;
  std::vector<std::pair<std::size_t, std::function<bool(cudaStream_t)>>> active_mem_pools_;
  std::unordered_map<void*, AllocationContext> ptr_to_allocation_context_;
};

std::shared_ptr<MemPool> createMemPool(
    std::size_t mem_pool_id,
    CudaMallocFn custom_malloc = nullptr,
    CudaFreeFn custom_free = nullptr,
    int block_pool_type = 0);
void beginAllocateToPool(
    int device,
    std::size_t mem_pool_id,
    std::function<bool(cudaStream_t)> filter);
void endAllocateToPool(int device, std::size_t mem_pool_id);
void releasePool(int device, std::size_t mem_pool_id);
