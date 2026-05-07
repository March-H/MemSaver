#pragma once

#include "device_caching_allocator.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

class MemPool {
 public:
  explicit MemPool(
      CudaMallocFn custom_malloc = nullptr,
      CudaFreeFn custom_free = nullptr,
      bool use_custom_pool = false)
      : malloc_fn_(std::move(custom_malloc)),
        free_fn_(std::move(custom_free)),
        use_custom_pool_(use_custom_pool) {
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
                       use_custom_pool_))
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
  bool use_custom_pool_ = false;
  std::unordered_map<int, std::unique_ptr<DeviceCachingAllocator>> devices_;
};
