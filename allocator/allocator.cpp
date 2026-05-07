#include "allocator.h"
#include "mem_pool.h"

#include <iterator>
#include <stdexcept>
#include <utility>

CachingAllocator& CachingAllocator::instance() {
  static CachingAllocator allocator;
  return allocator;
}

CachingAllocator::CachingAllocator() : default_mem_pool_(std::make_shared<MemPool>()) {}

CachingAllocator::~CachingAllocator() = default;

std::shared_ptr<MemPool> CachingAllocator::create_mem_pool(
    std::size_t mem_pool_id,
    CudaMallocFn custom_malloc,
    CudaFreeFn custom_free,
    int block_pool_type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = mem_pools_.find(mem_pool_id);
  if (it != mem_pools_.end()) {
    return it->second;
  }
  auto mem_pool = std::make_shared<MemPool>(
      std::move(custom_malloc),
      std::move(custom_free),
      block_pool_type);
  return mem_pools_.emplace(mem_pool_id, std::move(mem_pool)).first->second;
}

void CachingAllocator::begin_allocate_to_pool(
    int device,
    std::size_t mem_pool_id,
    std::function<bool(cudaStream_t)> filter) {
  (void)device;
  std::lock_guard<std::mutex> lock(mutex_);
  mem_pools_.at(mem_pool_id);
  active_mem_pools_.emplace_back(mem_pool_id, std::move(filter));
}

void CachingAllocator::end_allocate_to_pool(int device, std::size_t mem_pool_id) {
  (void)device;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = active_mem_pools_.rbegin(); it != active_mem_pools_.rend(); ++it) {
    if (it->first == mem_pool_id) {
      active_mem_pools_.erase(std::next(it).base());
      return;
    }
  }
}

void CachingAllocator::release_pool(int device, std::size_t mem_pool_id) {
  (void)device;
  for (auto& entry : active_mem_pools_) {
    if (entry.first == mem_pool_id) {
      return;
    }
  }
  auto it = mem_pools_.find(mem_pool_id);
  if (it != mem_pools_.end() && it->second.use_count() == 1) {
    mem_pools_.erase(it);
  }
}

std::shared_ptr<MemPool> CachingAllocator::mem_pool_for_stream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = active_mem_pools_.rbegin(); it != active_mem_pools_.rend(); ++it) {
    if (it->second(stream)) {
      return mem_pools_.at(it->first);
    }
  }
  return default_mem_pool_;
}

void* CachingAllocator::malloc(std::size_t size, int device, cudaStream_t stream) {
  std::shared_ptr<MemPool> mem_pool = mem_pool_for_stream(stream);
  DeviceCachingAllocator& allocator = mem_pool->device_allocator(device);
  void* ptr = allocator.malloc(size, stream);
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    ptr_to_allocation_context_[ptr] = AllocationContext{std::move(mem_pool), &allocator};
  }
  return ptr;
}

void CachingAllocator::free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  AllocationContext context;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_allocation_context_.find(ptr);
    if (it == ptr_to_allocation_context_.end()) {
      throw std::invalid_argument("unknown pointer");
    }
    context = std::move(it->second);
    ptr_to_allocation_context_.erase(it);
  }

  context.device_allocator->free(ptr);
}

void CachingAllocator::record_stream(void* ptr, cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

  DeviceCachingAllocator* allocator = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_allocation_context_.find(ptr);
    if (it == ptr_to_allocation_context_.end()) {
      throw std::invalid_argument("unknown pointer");
    }
    allocator = it->second.device_allocator;
  }

  allocator->record_stream(ptr, stream);
}

void CachingAllocator::empty_cache() {
  default_mem_pool_->empty_cache();
}

void CachingAllocator::empty_cache(int device) {
  default_mem_pool_->empty_cache(device);
}

std::shared_ptr<MemPool> createMemPool(
    std::size_t mem_pool_id,
    CudaMallocFn custom_malloc,
    CudaFreeFn custom_free,
    int block_pool_type) {
  return CachingAllocator::instance().create_mem_pool(
      mem_pool_id,
      std::move(custom_malloc),
      std::move(custom_free),
      block_pool_type);
}

void beginAllocateToPool(
    int device,
    std::size_t mem_pool_id,
    std::function<bool(cudaStream_t)> filter) {
  CachingAllocator::instance().begin_allocate_to_pool(device, mem_pool_id, std::move(filter));
}

void endAllocateToPool(int device, std::size_t mem_pool_id) {
  CachingAllocator::instance().end_allocate_to_pool(device, mem_pool_id);
}

void releasePool(int device, std::size_t mem_pool_id) {
  CachingAllocator::instance().release_pool(device, mem_pool_id);
}
