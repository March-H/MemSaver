#ifndef MEMSAVER_INTERNAL_CONTEXT_IMPL_H_
#define MEMSAVER_INTERNAL_CONTEXT_IMPL_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "memsaver/memsaver_c.h"

namespace memsaver::internal {

/** State of a managed regular allocation. */
enum class AllocationState {
  ACTIVE,
  PAUSED,
};

/** Backing source used by an allocation record. */
enum class AllocationKind {
  REGULAR,
  ARENA,
};

/** Metadata tracked for each managed allocation pointer. */
struct AllocationMetadata {
  size_t size = 0;
  CUdevice device = 0;
  std::string tag;
  AllocationState state = AllocationState::ACTIVE;
  bool enable_cpu_backup = false;
  void* cpu_backup = nullptr;
  CUmemGenericAllocationHandle alloc_handle = 0;
  AllocationKind kind = AllocationKind::REGULAR;
};

/** Key used to identify an arena by device and tag. */
struct ArenaKey {
  CUdevice device = 0;
  std::string tag;

  bool operator==(const ArenaKey& other) const {
    return device == other.device && tag == other.tag;
  }
};

/** Hash implementation for ArenaKey. */
struct ArenaKeyHash {
  size_t operator()(const ArenaKey& key) const {
    return std::hash<int>()(static_cast<int>(key.device)) ^
           (std::hash<std::string>()(key.tag) << 1U);
  }
};

/** Active real-handle binding metadata for a virtual-only arena range. */
struct ArenaActiveBinding {
  size_t size = 0;
  CUmemGenericAllocationHandle alloc_handle = 0;
};

/** Arena bookkeeping for arena-mode allocations. */
struct Arena {
  void* base_ptr = nullptr;
  size_t capacity = 0;
  size_t offset = 0;
  uint64_t live_allocs = 0;
  CUmemGenericAllocationHandle alloc_handle = 0;
  CUmemGenericAllocationHandle empty_handle = 0;
  size_t minimum_granularity_bytes = 0;
  std::unordered_map<size_t, ArenaActiveBinding> active_bindings;
  memsaver_arena_create_mode_t create_mode = MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED;
};

/** Thread-level runtime config used when dispatching memsaver_malloc. */
struct RuntimeConfig {
  bool interesting_region = false;
  bool enable_cpu_backup = false;
  std::string tag = "default";
  memsaver_allocation_mode_t allocation_mode = MEMSAVER_ALLOCATION_MODE_NORMAL;
};

/** Internal core implementation shared by C API and preload API. */
class ContextImpl {
 public:
  /**
   * @param use_original_cuda_symbols When true, unknown pointers are freed
   *        through original CUDA symbols resolved by dynamic linker.
   */
  explicit ContextImpl(bool use_original_cuda_symbols);
  /** Releases tracked allocations and arenas on shutdown. */
  ~ContextImpl();

  ContextImpl(const ContextImpl&) = delete;
  ContextImpl& operator=(const ContextImpl&) = delete;

  /**
   * Allocate memory according to runtime config.
   *
   * NORMAL mode uses VMM-backed regular allocation; ARENA mode
   * allocates from preconfigured arena.
   */
  cudaError_t Malloc(
      void** ptr,
      CUdevice device,
      size_t size,
      const RuntimeConfig& config);
  /** Free managed allocation or forward unknown pointer to real cudaFree. */
  cudaError_t Free(void* ptr);

  /** Pause regular managed allocations filtered by tag (empty means all). */
  cudaError_t Pause(const std::string& tag_filter);
  /** Resume paused regular managed allocations filtered by tag. */
  cudaError_t Resume(const std::string& tag_filter);

  /** Create/reconfigure arena for (device, tag). */
  cudaError_t ConfigureArena(
      const std::string& tag,
      CUdevice device,
      uint64_t capacity_bytes,
      memsaver_arena_create_mode_t create_mode);
  /** Reset arena offset to zero; requires no live arena allocations. */
  cudaError_t ResetArena(const std::string& tag, CUdevice device);
  /** Activate real device memory mappings for offset ranges in a virtual-only arena. */
  cudaError_t ActivateArenaOffsets(
      const std::string& tag,
      CUdevice device,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes);
  /** Deactivate mapped ranges and map shared empty handle back for offsets. */
  cudaError_t DeactivateArenaOffsets(
      const std::string& tag,
      CUdevice device,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes);

  /** Configure free-memory margin used by regular allocation checks. */
  void SetMemoryMarginBytes(uint64_t value) { memory_margin_bytes_.store(value); }

  /** Resolve host backup pointer for a queried GPU range. */
  cudaError_t GetCpuBackupPointer(
      const uint8_t* query_gpu_ptr,
      uint64_t query_size,
      uint8_t** out_cpu_ptr);

 private:
  /** Implementation for regular VMM-backed allocations. */
  cudaError_t MallocRegular(
      void** ptr,
      CUdevice device,
      size_t size,
      const std::string& tag,
      bool enable_cpu_backup);
  /** Implementation for arena allocations. */
  cudaError_t MallocArena(
      void** ptr,
      CUdevice device,
      size_t size,
      const std::string& tag);

  /** Create a new arena mapping. */
  cudaError_t CreateArena(
      const ArenaKey& key,
      size_t capacity_bytes,
      memsaver_arena_create_mode_t create_mode);
  /** Release CUDA resources held by a fully-mapped arena. */
  cudaError_t ReleaseArena(const Arena& arena);
  /** Lazily create and cache one minimum-granularity handle per device. */
  cudaError_t GetOrCreateSharedMinimumGranularityHandle(
      CUdevice device,
      CUmemGenericAllocationHandle* out_handle);
  /** Map [address, address+size) using the shared empty handle by granularity chunks. */
  cudaError_t MapRangeToEmptyHandle(
      CUdeviceptr address,
      size_t size,
      CUmemGenericAllocationHandle empty_handle,
      size_t minimum_granularity_bytes);

  /** Best-effort cleanup used only during object destruction. */
  void ReleaseAllocationForShutdown(void* ptr, const AllocationMetadata& metadata);

  const bool use_original_cuda_symbols_;
  std::mutex mutex_;
  std::unordered_map<void*, AllocationMetadata> allocations_;
  std::unordered_map<ArenaKey, Arena, ArenaKeyHash> arenas_;
  std::unordered_map<CUdevice, CUmemGenericAllocationHandle>
      shared_minimum_granularity_handles_;
  std::atomic<uint64_t> memory_margin_bytes_ = 0;
};

}  // namespace memsaver::internal

#endif  // MEMSAVER_INTERNAL_CONTEXT_IMPL_H_
