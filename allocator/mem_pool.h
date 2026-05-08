#pragma once

#include "device_caching_allocator.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

class MemPool {
 public:
  explicit MemPool(
      CudaMallocFn malloc_fn = nullptr,
      CudaFreeFn free_fn = nullptr,
      int block_pool_type = 0)
      : malloc_fn_(std::move(malloc_fn)),
        free_fn_(std::move(free_fn)),
        block_pool_type_(block_pool_type) {
    if (!malloc_fn_) {
      malloc_fn_ = [](void** ptr, std::size_t size) {
        return cudaMalloc(ptr, size);
      };
    }
    if (!free_fn_) {
      free_fn_ = [](void* ptr) {
        return cudaFree(ptr);
      };
    }
  }

  DeviceCachingAllocator& device_allocator(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = devices_.find(device);
    if (it == devices_.end()) {
      it = devices_
               .emplace(
                   device,
                   std::make_unique<DeviceCachingAllocator>(
                       device,
                       malloc_fn_,
                       free_fn_,
                       block_pool_type_))
               .first;
    }
    return *it->second;
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [device, allocator] : devices_) {
      allocator->empty_cache();
    }
  }

  void empty_cache(int device) {
    device_allocator(device).empty_cache();
  }

 private:
  mutable std::mutex mutex_;
  CudaMallocFn malloc_fn_;
  CudaFreeFn free_fn_;
  int block_pool_type_ = 0;
  std::unordered_map<int, std::unique_ptr<DeviceCachingAllocator>> devices_;
};
