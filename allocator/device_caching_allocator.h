#pragma once

#include "allocator.h"
#include "block.h"
#include "memsaver/entrypoint.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

inline constexpr std::size_t kAlignment = 512;
inline constexpr std::size_t kSmallSizeThreshold = 1ULL << 20;
inline constexpr std::size_t kLargeSizeThreshold = 10ULL << 20;
inline constexpr std::size_t kSmallBuffer = 2ULL << 20;
inline constexpr std::size_t kLargeBuffer = 20ULL << 20;
inline constexpr std::size_t kRoundLarge = 2ULL << 20;
inline constexpr std::size_t kArenaVirtualMinimumActivation = 20ULL << 20;

inline std::size_t round_up(std::size_t size, std::size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

inline std::size_t round_size(std::size_t size) {
  return round_up(size, kAlignment);
}

inline std::size_t get_allocation_size(std::size_t size) {
  if (size <= kSmallSizeThreshold) {
    return kSmallBuffer;
  }
  if (size < kLargeSizeThreshold) {
    return kLargeBuffer;
  }
  return round_up(size, kRoundLarge);
}

[[noreturn]] inline void throw_cuda_error(cudaError_t error, const char* op) {
  throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(error));
}

class DeviceCachingAllocator {
 public:
  DeviceCachingAllocator(
      int device,
      CudaMallocFn malloc_fn,
      CudaFreeFn free_fn,
      int block_pool_type)
      : device_(device),
        malloc_fn_(std::move(malloc_fn)),
        free_fn_(std::move(free_fn)),
        block_pool_type_(block_pool_type) {}

  ~DeviceCachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : cuda_events_) {
      cudaEventDestroy(entry.first);
    }
    for (Block* block : blocks_) {
      cudaSetDevice(device_);
      free_fn_(block->ptr);
      delete_blocks(block);
    }
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
  BlockPool& get_pool(std::size_t size, cudaStream_t stream) {
    (void)stream;
    if (block_pool_type_ == 3) {
      return arena_virtual_blocks_;
    } else if (block_pool_type_ == 2) {
      return fixed_blocks_;
    } else if (block_pool_type_ == 1) {
      return large_blocks_;
    }
    if (size <= kSmallSizeThreshold) {
      return small_blocks_;
    }
    return large_blocks_;
  }

  bool is_arena_virtual_pool(const BlockPool& pool) const {
    return &pool == &arena_virtual_blocks_;
  }

  std::size_t get_arena_virtual_activation_size(std::size_t size) const {
    if (size <= kArenaVirtualMinimumActivation) {
      return kArenaVirtualMinimumActivation;
    }
    return round_up(size, kRoundLarge);
  }

  Block* root_block(Block* block) const {
    while (block->prev != nullptr) {
      block = block->prev;
    }
    return block;
  }

  void activate_arena_virtual_block(Block* block, std::size_t size) {
    if (block->active_size >= size) {
      return;
    }
    const std::size_t activation_size =
        get_arena_virtual_activation_size(size - block->active_size);
    Block* root = root_block(block);
    const uint64_t offset =
        static_cast<uint64_t>(
            static_cast<char*>(block->ptr) - static_cast<char*>(root->ptr)) +
        static_cast<uint64_t>(block->active_size);
    const cudaError_t error = memsaver_activate_arena_offsets(
        memsaver_current_region_tag(),
        &offset,
        1ULL,
        static_cast<uint64_t>(activation_size));
    if (error != cudaSuccess) {
      throw_cuda_error(error, "memsaver_activate_arena_offsets");
    }
    block->active_size += activation_size;
  }

  Block* find_free_block(BlockPool& pool, std::size_t size, cudaStream_t stream) {
    ++pool.get_free_blocks_call_count;
    Block search_key(device_, stream, size);
    for (auto it = pool.blocks.lower_bound(&search_key); it != pool.blocks.end(); ++it) {
      if ((*it)->stream == stream) {
        if (is_arena_virtual_pool(pool)) {
          activate_arena_virtual_block(*it, size);
        }
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
    error = malloc_fn_(&ptr, allocation_size);
    if (error != cudaSuccess) {
      stats_.num_alloc_retries += 1;
      process_events();
      release_reclaimable_blocks();
      error = malloc_fn_(&ptr, allocation_size);
    }
    if (error != cudaSuccess) {
      stats_.num_ooms += 1;
      throw_cuda_error(error, "cudaMalloc");
    }

    Block* block = new Block(device_, stream, allocation_size, &pool, ptr);
    if (!is_arena_virtual_pool(pool)) {
      block->active_size = allocation_size;
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
  }

  void release_reclaimable_blocks() {
    for (auto it = blocks_.begin(); it != blocks_.end();) {
      Block* block = *it;
      if (block->allocated || block->event_count != 0 || block->is_split()) {
        ++it;
        continue;
      }

      remove_free_block(block);

      cudaError_t error = cudaSetDevice(device_);
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaSetDevice");
      }

      error = free_fn_(block->ptr);
      if (error != cudaSuccess) {
        throw_cuda_error(error, "cudaFree");
      }

      stats_.reserved_bytes -= block->size;
      delete_blocks(block);
      it = blocks_.erase(it);
    }
  }

  int device_ = 0;
  CudaMallocFn malloc_fn_;
  CudaFreeFn free_fn_;
  int block_pool_type_ = 0;
  mutable std::mutex mutex_;
  std::size_t limit_bytes_ = static_cast<std::size_t>(-1);
  AllocatorStats stats_;
  std::unordered_map<void*, Block*> live_blocks_;
  BlockPool small_blocks_{true};
  BlockPool large_blocks_{false};
  BlockPool fixed_blocks_{false};
  BlockPool arena_virtual_blocks_{false};
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events_;
  std::vector<Block*> blocks_;
};
