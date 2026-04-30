#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

struct AllocatorStats {
  std::size_t allocated_bytes = 0;
  std::size_t reserved_bytes = 0;
  std::size_t max_allocated_bytes = 0;
  std::size_t max_reserved_bytes = 0;
  std::size_t num_alloc_retries = 0;
  std::size_t num_ooms = 0;
};

class CachingAllocator {
 public:
  static CachingAllocator& instance();

  void create_or_incref_pool(int device, std::size_t mem_pool_id);
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
  class DeviceCachingAllocator;

  CachingAllocator();
  ~CachingAllocator();

  CachingAllocator(const CachingAllocator&) = delete;
  CachingAllocator& operator=(const CachingAllocator&) = delete;

  DeviceCachingAllocator& device_allocator(int device);

  mutable std::mutex mutex_;
  std::unordered_map<int, std::unique_ptr<DeviceCachingAllocator>> devices_;
  std::unordered_map<void*, DeviceCachingAllocator*> ptr_to_device_allocator_;
};

namespace c10::cuda::CUDACachingAllocator {

void createOrIncrefPool(int device, std::size_t mem_pool_id);
void beginAllocateToPool(
    int device,
    std::size_t mem_pool_id,
    std::function<bool(cudaStream_t)> filter);
void endAllocateToPool(int device, std::size_t mem_pool_id);
void releasePool(int device, std::size_t mem_pool_id);

}
