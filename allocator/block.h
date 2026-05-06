#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

#include <cuda_runtime.h>

struct Block;
using Comparison = bool (*)(const Block*, const Block*);

inline std::atomic<int32_t>& block_registration_counter_global() {
  static std::atomic<int32_t> counter{0};
  return counter;
}

inline bool block_comparator_registration_counter(const Block* a, const Block* b);

struct BlockPool {
  explicit BlockPool(bool small) : blocks(block_comparator_registration_counter), is_small(small) {}

  std::set<Block*, Comparison> blocks;
  const bool is_small;
  int64_t get_free_blocks_call_count{0};

  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(Block* block);
};

struct Block {
  int device;
  cudaStream_t stream;
  int32_t registration_counter{-1};
  std::size_t size;
  BlockPool* pool{nullptr}; // 记录归属于哪个BlockPool
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
        block_registration_counter_global().fetch_add(1, std::memory_order_relaxed) + 1;
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

inline bool block_comparator_registration_counter(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return a->registration_counter < b->registration_counter;
}

inline std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::insert_into_blocks(
    Block* block) {
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

inline void delete_blocks(Block* head) {
  while (head != nullptr) {
    Block* next = head->next;
    delete head;
    head = next;
  }
}
