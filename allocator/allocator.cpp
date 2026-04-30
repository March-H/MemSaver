#include "allocator.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

constexpr std::size_t kAlignment = 512;
constexpr std::size_t kSmallSizeThreshold = 1ULL << 20;
constexpr std::size_t kLargeSizeThreshold = 10ULL << 20;
constexpr std::size_t kSmallBuffer = 2ULL << 20;
constexpr std::size_t kLargeBuffer = 20ULL << 20;
constexpr std::size_t kRoundLarge = 2ULL << 20;

static std::atomic<int32_t> registration_counter_global{0};

static std::size_t round_up(std::size_t size, std::size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

static std::size_t round_size(std::size_t size) {
  return round_up(size, kAlignment);
}

static std::size_t get_allocation_size(std::size_t size) {
  if (size <= kSmallSizeThreshold) {
    return kSmallBuffer;
  }
  if (size < kLargeSizeThreshold) {
    return kLargeBuffer;
  }
  return round_up(size, kRoundLarge);
}

[[noreturn]] static void throw_cuda_error(cudaError_t error, const char* op) {
  throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(error));
}

struct Block;
using Comparison = bool (*)(const Block*, const Block*);

static bool block_comparator_registration_counter(const Block* a, const Block* b);

struct PrivatePool;

struct BlockPool {
  explicit BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(block_comparator_registration_counter),
        is_small(small),
        owner_private_pool(private_pool) {}

  std::set<Block*, Comparison> blocks;
  const bool is_small;
  PrivatePool* owner_private_pool;
  int64_t get_free_blocks_call_count{0};

  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(Block* block);
};

struct PrivatePool {
  explicit PrivatePool(std::size_t mem_pool_id)
      : mem_pool_id(mem_pool_id),
        small_blocks(true, this),
        large_blocks(false, this) {}

  std::size_t mem_pool_id;
  std::size_t use_count{1};
  std::size_t cuda_malloc_count{0};
  BlockPool small_blocks;
  BlockPool large_blocks;
};

struct Block {
  int device;
  cudaStream_t stream;
  int32_t registration_counter{-1};
  std::size_t size;
  BlockPool* pool{nullptr};
  void* ptr{nullptr};
  bool allocated{false};
  Block* prev{nullptr};
  Block* next{nullptr};
  int event_count{0};
  int64_t gc_count_base{0};
  std::set<cudaStream_t> stream_uses;

  Block(int device, cudaStream_t stream, std::size_t size, BlockPool* pool, void* ptr)
      : device(device), stream(stream), size(size), pool(pool), ptr(ptr) {
    registration_counter =
        registration_counter_global.fetch_add(1, std::memory_order_relaxed) + 1;
  }

  Block(int device, cudaStream_t stream, std::size_t size)
      : device(device), stream(stream), size(size) {}

  std::size_t gc_count() const {
    return static_cast<std::size_t>(pool->get_free_blocks_call_count - gc_count_base);
  }

  bool is_split() const {
    return prev != nullptr || next != nullptr;
  }

  void splice(Block* before, Block* after) {
    if (before != nullptr) {
      before->next = this;
    }
    prev = before;
    if (after != nullptr) {
      after->prev = this;
    }
    next = after;
  }

  Block* split(std::size_t first_size) {
    if (size <= first_size) {
      return nullptr;
    }
    Block* remainder = new Block(
        device,
        stream,
        size - first_size,
        pool,
        static_cast<char*>(ptr) + first_size);
    remainder->splice(this, next);
    size = first_size;
    return remainder;
  }

  bool can_merge(const Block* after) const {
    return after != nullptr && !allocated && !after->allocated && device == after->device &&
        event_count == 0 && after->event_count == 0 && stream_uses.empty() &&
        after->stream_uses.empty() && stream == after->stream && pool == after->pool &&
        static_cast<const char*>(ptr) + size == after->ptr;
  }

  void merge(Block* after) {
    size += after->size;
    next = after->next;
    if (next != nullptr) {
      next->prev = this;
    }
    delete after;
  }
};

static bool block_comparator_registration_counter(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return a->registration_counter < b->registration_counter;
}

std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::insert_into_blocks(
    Block* block) {
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

static void delete_blocks(Block* head) {
  while (head != nullptr) {
    Block* next = head->next;
    delete head;
    head = next;
  }
}

class CachingAllocator::DeviceCachingAllocator {
 public:
  explicit DeviceCachingAllocator(int device) : device_(device) {}

  ~DeviceCachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : cuda_events_) {
      cudaEventDestroy(entry.first);
    }
    for (Block* block : blocks_) {
      cudaSetDevice(device_);
      cudaFree(block->ptr);
      delete_blocks(block);
    }
  }

  void create_or_incref_pool(std::size_t mem_pool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    create_or_incref_pool_impl(mem_pool_id);
  }

  void begin_allocate_to_pool(
      std::size_t mem_pool_id,
      std::function<bool(cudaStream_t)> filter) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (get_private_pool(mem_pool_id) == nullptr) {
      throw std::invalid_argument("unknown mem pool");
    }
    active_mempools_.emplace_back(mem_pool_id, std::move(filter));
  }

  void end_allocate_to_pool(std::size_t mem_pool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = active_mempools_.rbegin(); it != active_mempools_.rend(); ++it) {
      if (it->first == mem_pool_id) {
        active_mempools_.erase(std::next(it).base());
        return;
      }
    }
  }

  void release_pool(std::size_t mem_pool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = private_pools_.find(mem_pool_id);
    if (it == private_pools_.end()) {
      return;
    }
    it->second->use_count -= 1;
    if (it->second->use_count != 0) {
      return;
    }
    freeable_private_pools_[mem_pool_id] = it->second.get();
    process_events();
    release_freeable_private_pool(mem_pool_id);
  }

  void* malloc(std::size_t size, cudaStream_t stream) {
    if (size == 0) {
      return nullptr;
    }

    const std::size_t rounded = round_size(size);
    std::lock_guard<std::mutex> lock(mutex_);
    process_events();
    BlockPool& pool = get_pool(rounded, stream);

    Block* block = find_free_block(pool, rounded, stream);
    if (block == nullptr) {
      block = allocate_from_cuda(rounded, stream, pool);
    }

    remove_free_block(block);

    Block* remainder = block->split(rounded);
    if (remainder != nullptr) {
      insert_free_block(remainder);
    }
    block->allocated = true;
    block->stream = stream;
    live_blocks_[block->ptr] = block;
    stats_.allocated_bytes += block->size;
    stats_.max_allocated_bytes = std::max(stats_.max_allocated_bytes, stats_.allocated_bytes);
    return block->ptr;
  }

  void free(void* ptr) {
    if (ptr == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_blocks_.find(ptr);
    if (it == live_blocks_.end()) {
      throw std::invalid_argument("unknown pointer");
    }

    Block* block = it->second;
    live_blocks_.erase(it);
    block->allocated = false;
    stats_.allocated_bytes -= block->size;
    insert_events(block);
  }

  void record_stream(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_blocks_.find(ptr);
    if (it == live_blocks_.end()) {
      throw std::invalid_argument("unknown pointer");
    }

    Block* block = it->second;
    if (stream == block->stream) {
      return;
    }
    block->stream_uses.insert(stream);
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    process_events();
    release_reclaimable_blocks();
  }

 private:
  void create_or_incref_pool_impl(std::size_t mem_pool_id) {
    auto it = private_pools_.find(mem_pool_id);
    if (it == private_pools_.end()) {
      private_pools_.emplace(mem_pool_id, std::make_unique<PrivatePool>(mem_pool_id));
      return;
    }
    freeable_private_pools_.erase(mem_pool_id);
    it->second->use_count += 1;
  }

  PrivatePool* get_private_pool(std::size_t mem_pool_id) {
    auto it = private_pools_.find(mem_pool_id);
    if (it == private_pools_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  BlockPool& get_pool(std::size_t size, cudaStream_t stream) {
    for (auto it = active_mempools_.rbegin(); it != active_mempools_.rend(); ++it) {
      if (it->second(stream)) {
        PrivatePool* pool = get_private_pool(it->first);
        if (size <= kSmallSizeThreshold) {
          return pool->small_blocks;
        }
        return pool->large_blocks;
      }
    }
    if (size <= kSmallSizeThreshold) {
      return small_blocks_;
    }
    return large_blocks_;
  }

  Block* find_free_block(BlockPool& pool, std::size_t size, cudaStream_t stream) {
    ++pool.get_free_blocks_call_count;
    Block search_key(device_, stream, size);
    for (auto it = pool.blocks.lower_bound(&search_key); it != pool.blocks.end(); ++it) {
      if ((*it)->stream == stream) {
        return *it;
      }
    }
    return nullptr;
  }

  Block* allocate_from_cuda(std::size_t size, cudaStream_t stream, BlockPool& pool) {
    const std::size_t allocation_size = get_allocation_size(size);
    if (stats_.reserved_bytes + allocation_size > limit_bytes_) {
      process_events();
      release_reclaimable_blocks();
    }
    if (stats_.reserved_bytes + allocation_size > limit_bytes_) {
      stats_.num_ooms += 1;
      throw std::runtime_error("memory limit exceeded");
    }

    cudaError_t error = cudaSetDevice(device_);
    if (error != cudaSuccess) {
      throw_cuda_error(error, "cudaSetDevice");
    }

    void* ptr = nullptr;
    error = cudaMalloc(&ptr, allocation_size);
    if (error != cudaSuccess) {
      stats_.num_alloc_retries += 1;
      process_events();
      release_reclaimable_blocks();
      error = cudaMalloc(&ptr, allocation_size);
    }
    if (error != cudaSuccess) {
      stats_.num_ooms += 1;
      throw_cuda_error(error, "cudaMalloc");
    }

    Block* block = new Block(device_, stream, allocation_size, &pool, ptr);
    if (pool.owner_private_pool != nullptr) {
      pool.owner_private_pool->cuda_malloc_count += 1;
    }
    blocks_.push_back(block);
    stats_.reserved_bytes += allocation_size;
    stats_.max_reserved_bytes = std::max(stats_.max_reserved_bytes, stats_.reserved_bytes);
    return block;
  }

  void insert_free_block(Block* block) {
    block->pool->insert_into_blocks(block);
  }

  void remove_free_block(Block* block) {
    auto it = block->pool->blocks.find(block);
    if (it == block->pool->blocks.end()) {
      return;
    }
    block->pool->blocks.erase(it);
  }

  void insert_events(Block* block) {
    cudaError_t error = cudaSetDevice(device_);
    if (error != cudaSuccess) {
      throw_cuda_error(error, "cudaSetDevice");
    }

    insert_event(block, block->stream);
    for (cudaStream_t stream : block->stream_uses) {
      insert_event(block, stream);
    }
  }

  void process_events() {
    for (auto it = cuda_events_.begin(); it != cuda_events_.end();) {
      cudaError_t error = cudaEventQuery(it->first);
      if (error == cudaErrorNotReady) {
        ++it;
        continue;
      }
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaEventQuery");
      }
      Block* block = it->second;
      error = cudaEventDestroy(it->first);
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaEventDestroy");
      }
      --block->event_count;
      if (block->event_count == 0) {
        block->stream_uses.clear();
        reclaim_block(block);
      }
      it = cuda_events_.erase(it);
    }
  }

  void insert_event(Block* block, cudaStream_t stream) {
    cudaEvent_t event = nullptr;
    cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
      throw_cuda_error(error, "cudaEventCreateWithFlags");
    }

    error = cudaEventRecord(event, stream);
    if (error != cudaSuccess) {
      cudaEventDestroy(event);
      throw_cuda_error(error, "cudaEventRecord");
    }

    ++block->event_count;
    cuda_events_.emplace_back(event, block);
  }

  void reclaim_block(Block* block) {
    Block* prev = block->prev;
    if (prev != nullptr && prev->can_merge(block)) {
      remove_free_block(prev);
      prev->merge(block);
      block = prev;
    }
    if (block->next != nullptr && block->can_merge(block->next)) {
      remove_free_block(block->next);
      block->merge(block->next);
    }
    insert_free_block(block);
    if (block->pool->owner_private_pool != nullptr &&
        block->pool->owner_private_pool->use_count == 0) {
      release_freeable_private_pool(block->pool->owner_private_pool->mem_pool_id);
    }
  }

  void release_reclaimable_blocks() {
    release_reclaimable_blocks(nullptr);
  }

  bool block_belongs_to_pool(Block* block, PrivatePool* pool) const {
    return block->pool == &pool->small_blocks || block->pool == &pool->large_blocks;
  }

  void release_freeable_private_pool(std::size_t mem_pool_id) {
    auto freeable_it = freeable_private_pools_.find(mem_pool_id);
    if (freeable_it == freeable_private_pools_.end()) {
      return;
    }
    PrivatePool* pool = freeable_it->second;
    release_reclaimable_blocks(pool);
    if (pool->cuda_malloc_count != 0) {
      return;
    }
    freeable_private_pools_.erase(freeable_it);
    private_pools_.erase(mem_pool_id);
  }

  void release_reclaimable_blocks(PrivatePool* private_pool) {
    for (auto it = blocks_.begin(); it != blocks_.end();) {
      Block* block = *it;
      if (private_pool != nullptr && !block_belongs_to_pool(block, private_pool)) {
        ++it;
        continue;
      }
      if (block->allocated || block->event_count != 0 || block->is_split()) {
        ++it;
        continue;
      }

      remove_free_block(block);

      cudaError_t error = cudaSetDevice(device_);
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaSetDevice");
      }

      error = cudaFree(block->ptr);
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaFree");
      }

      if (block->pool->owner_private_pool != nullptr) {
        block->pool->owner_private_pool->cuda_malloc_count -= 1;
      }
      stats_.reserved_bytes -= block->size;
      delete_blocks(block);
      it = blocks_.erase(it);
    }
  }

  int device_ = 0;
  mutable std::mutex mutex_;
  std::size_t limit_bytes_ = static_cast<std::size_t>(-1);
  AllocatorStats stats_;
  std::unordered_map<void*, Block*> live_blocks_;
  BlockPool small_blocks_{true};
  BlockPool large_blocks_{false};
  std::unordered_map<std::size_t, std::unique_ptr<PrivatePool>> private_pools_;
  std::unordered_map<std::size_t, PrivatePool*> freeable_private_pools_;
  std::vector<std::pair<std::size_t, std::function<bool(cudaStream_t)>>> active_mempools_;
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events_;
  std::vector<Block*> blocks_;
};

CachingAllocator& CachingAllocator::instance() {
  static CachingAllocator allocator;
  return allocator;
}

CachingAllocator::CachingAllocator() = default;

CachingAllocator::~CachingAllocator() = default;

CachingAllocator::DeviceCachingAllocator& CachingAllocator::device_allocator(int device) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = devices_.find(device);
  if (it == devices_.end()) {
    it = devices_.emplace(device, std::make_unique<DeviceCachingAllocator>(device)).first;
  }
  return *it->second;
}

void CachingAllocator::create_or_incref_pool(int device, std::size_t mem_pool_id) {
  device_allocator(device).create_or_incref_pool(mem_pool_id);
}

void CachingAllocator::begin_allocate_to_pool(
    int device,
    std::size_t mem_pool_id,
    std::function<bool(cudaStream_t)> filter) {
  device_allocator(device).begin_allocate_to_pool(mem_pool_id, std::move(filter));
}

void CachingAllocator::end_allocate_to_pool(int device, std::size_t mem_pool_id) {
  device_allocator(device).end_allocate_to_pool(mem_pool_id);
}

void CachingAllocator::release_pool(int device, std::size_t mem_pool_id) {
  device_allocator(device).release_pool(mem_pool_id);
}

void* CachingAllocator::malloc(std::size_t size, int device, cudaStream_t stream) {
  DeviceCachingAllocator& allocator = device_allocator(device);
  void* ptr = allocator.malloc(size, stream);
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    ptr_to_device_allocator_[ptr] = &allocator;
  }
  return ptr;
}

void CachingAllocator::free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  DeviceCachingAllocator* allocator = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_device_allocator_.find(ptr);
    if (it == ptr_to_device_allocator_.end()) {
      throw std::invalid_argument("unknown pointer");
    }
    allocator = it->second;
    ptr_to_device_allocator_.erase(it);
  }

  allocator->free(ptr);
}

void CachingAllocator::record_stream(void* ptr, cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

  DeviceCachingAllocator* allocator = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_device_allocator_.find(ptr);
    if (it == ptr_to_device_allocator_.end()) {
      throw std::invalid_argument("unknown pointer");
    }
    allocator = it->second;
  }

  allocator->record_stream(ptr, stream);
}

void CachingAllocator::empty_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [device, allocator] : devices_) {
    allocator->empty_cache();
  }
}

void CachingAllocator::empty_cache(int device) {
  device_allocator(device).empty_cache();
}

namespace c10::cuda::CUDACachingAllocator {

void createOrIncrefPool(int device, std::size_t mem_pool_id) {
  CachingAllocator::instance().create_or_incref_pool(device, mem_pool_id);
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

}
