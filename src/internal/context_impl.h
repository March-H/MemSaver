#ifndef CONTEXT_IMPL_H_
#define CONTEXT_IMPL_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "memsaver/entrypoint.h"

/** State of a managed regular allocation. */
enum class AllocationState {
  ACTIVE,
  PAUSED,
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
  CUmemGenericAllocationHandle empty_handle = 0;
  std::unordered_map<uint64_t, CUmemGenericAllocationHandle> arena_offset_handles;
};

/** Thread-level runtime config used when dispatching memsaver_malloc. */
struct RuntimeConfig {
  bool interesting_region = false;
  bool enable_cpu_backup = false;
  std::string tag = "default";
  AllocationKind allocation_mode = AllocationKind::REGULAR;
};

/** Internal core implementation shared by C API and preload API. */
class ContextImpl {
 public:
  explicit ContextImpl();
  ~ContextImpl();

  ContextImpl(const ContextImpl&) = delete;
  ContextImpl& operator=(const ContextImpl&) = delete;

  static ContextImpl& instance();

  cudaError_t Malloc(
      void** ptr,
      CUdevice device,
      size_t size,
      const RuntimeConfig& config);
  cudaError_t Free(void* ptr);

  cudaError_t Pause(const std::string& tag_filter);
  cudaError_t Resume(const std::string& tag_filter);

  cudaError_t ActivateArenaOffsets(
      const std::string& tag,
      CUdevice device,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes);
  cudaError_t DeactivateArenaOffsets(
      const std::string& tag,
      CUdevice device,
      const uint64_t* offsets,
      uint64_t num_offsets,
      uint64_t size_bytes);

  cudaError_t GetCpuBackupPointer(
      const uint8_t* query_gpu_ptr,
      uint64_t query_size,
      uint8_t** out_cpu_ptr);
  cudaError_t GetMetadataCountByTag(
      const std::string& tag,
      uint64_t* out_count);
  cudaError_t ReleaseAllocations(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind kind);
  bool HasAllocations(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind kind);

 private:
  CUresult MallocRegular(
      void** ptr,
      CUdevice device,
      size_t size,
      const std::string& tag,
      bool enable_cpu_backup);
  CUresult MallocArena(
      void** ptr,
      CUdevice device,
      size_t size,
      const std::string& tag);
  CUresult GetOrCreateSharedMinimumGranularityHandle(
      CUdevice device,
      CUmemGenericAllocationHandle* out_handle);
  CUresult MapRangeToEmptyHandle(
      CUdeviceptr address,
      size_t size,
      CUmemGenericAllocationHandle empty_handle,
      size_t minimum_granularity_bytes);

  void ReleaseAllocationForShutdown(void* ptr, const AllocationMetadata& metadata);

  std::mutex mutex_;
  std::unordered_map<void*, AllocationMetadata> allocations_;
  std::unordered_map<CUdevice, CUmemGenericAllocationHandle>
      shared_minimum_granularity_handles_;
};

#endif  // CONTEXT_IMPL_H_
