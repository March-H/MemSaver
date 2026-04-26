#include "internal/context_impl.h"

#include <cstring>
#include <limits>
#include <utility>

#include "internal/original_cuda_api.h"
#include "internal/utils.h"
#include "internal/vmm.h"

namespace memsaver::internal {

// Construct core context. Behavior of unknown-pointer free is configured by
// use_original_cuda_symbols_.
ContextImpl::ContextImpl(const bool use_original_cuda_symbols)
    : use_original_cuda_symbols_(use_original_cuda_symbols) {}

// Tear down all tracked allocation resources and virtual arenas.
ContextImpl::~ContextImpl() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& kv : allocations_) {
    ReleaseAllocationForShutdown(kv.first, kv.second);
  }
  allocations_.clear();
}

// Destructor-only cleanup path for one allocation record.
//
// Every cleanup call is still checked here. Any CUDA/CU cleanup failure logs and
// aborts immediately.
void ContextImpl::ReleaseAllocationForShutdown(
    void* ptr,
    const AllocationMetadata& metadata) {
  if (metadata.state == AllocationState::ACTIVE) {
    (void)utils::CheckCu(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size),
                        "cuMemUnmap", __FILE__, __func__, __LINE__);
    if (metadata.alloc_handle != 0) {
      (void)utils::CheckCu(cuMemRelease(metadata.alloc_handle), "cuMemRelease",
                          __FILE__, __func__, __LINE__);
    }
  }

  (void)utils::CheckCu(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), metadata.size),
                      "cuMemAddressFree", __FILE__, __func__, __LINE__);

  if (metadata.cpu_backup != nullptr) {
    (void)utils::CheckCuda(cudaFreeHost(metadata.cpu_backup), "cudaFreeHost",
                          __FILE__, __func__, __LINE__);
  }
}

// Top-level allocation entry. Dispatches to regular allocation or arena
// allocation based on runtime config.
cudaError_t ContextImpl::Malloc(
    void** ptr,
    const CUdevice device,
    const size_t size,
    const RuntimeConfig& config) {
  if (!config.interesting_region) {
    RETURN_IF_CUDA_ERROR(
        OriginalCudaApi::Malloc(ptr, size, use_original_cuda_symbols_));
    return cudaSuccess;
  }

  if (config.allocation_mode == MEMSAVER_ALLOCATION_MODE_ARENA) {
    RETURN_IF_CU_ERROR_AS_CUDA(MallocArena(ptr, device, size, config.tag));
    return cudaSuccess;
  }
  RETURN_IF_CU_ERROR_AS_CUDA(
      MallocRegular(ptr, device, size, config.tag, config.enable_cpu_backup));
  return cudaSuccess;
}

// Allocate one regular VMM-backed block and track its metadata.
CUresult ContextImpl::MallocRegular(
    void** ptr,
    const CUdevice device,
    const size_t size,
    const std::string& tag,
    const bool enable_cpu_backup) {

  CUmemGenericAllocationHandle handle = 0;
  RETURN_IF_CU_ERROR(vmm::CreateMemoryHandle(&handle, size, device));

  CUdeviceptr reserved_address = 0;
  RETURN_IF_CU_ERROR(cuMemAddressReserve(&reserved_address, size, 0, 0, 0));
  RETURN_IF_CU_ERROR(cuMemMap(reserved_address, size, 0, handle, 0));
  RETURN_IF_CU_ERROR(vmm::SetAccess(reserved_address, size, device));

  *ptr = reinterpret_cast<void*>(reserved_address);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    allocations_.emplace(
        *ptr,
        AllocationMetadata{
            size,
            device,
            tag,
            AllocationState::ACTIVE,
            enable_cpu_backup,
            nullptr,
            handle,
            AllocationKind::REGULAR,
        });
  }

  return CUDA_SUCCESS;
}

// Free pointer managed by this context.
//
// Behavior:
// - unknown pointer: forward to real cudaFree path
// - managed pointer: unmap/release/free VMM resources and optional CPU backup
cudaError_t ContextImpl::Free(void* ptr) {

  AllocationMetadata metadata;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto metadata_it = allocations_.find(ptr);
    if (metadata_it == allocations_.end()) {
      RETURN_IF_CUDA_ERROR(OriginalCudaApi::Free(ptr, use_original_cuda_symbols_));
      return cudaSuccess;
    }

    metadata = metadata_it->second;
    allocations_.erase(metadata_it);
  }

  // PAUSED allocations have already been unmapped/released in Pause(), so
  // only ACTIVE allocations should run the unmap/release path here.
  if (metadata.state == AllocationState::ACTIVE) {
    RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
    RETURN_IF_CU_ERROR_AS_CUDA(
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));
    RETURN_IF_FALSE(metadata.alloc_handle != 0, cudaErrorInvalidValue,
                             "Free: active allocation handle should not be 0");
    RETURN_IF_CU_ERROR_AS_CUDA(cuMemRelease(metadata.alloc_handle));
  }

  RETURN_IF_CU_ERROR_AS_CUDA(
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));

  if (metadata.cpu_backup != nullptr) {
    RETURN_IF_CUDA_ERROR(cudaFreeHost(metadata.cpu_backup));
  }

  return cudaSuccess;
}

// Pause regular managed allocations matching tag filter.
//
// For each target allocation:
// - optionally copy to pinned CPU backup
// - unmap and release physical allocation handle
// - keep virtual address reservation alive
cudaError_t ContextImpl::Pause(const std::string& tag_filter) {
  const std::lock_guard<std::mutex> lock(mutex_);

  for (auto& kv : allocations_) {
    void* ptr = kv.first;
    AllocationMetadata& metadata = kv.second;

    if (metadata.kind != AllocationKind::REGULAR) {
      continue;
    }

    if (!utils::MatchesTag(tag_filter, metadata.tag)) {
      continue;
    }

    RETURN_IF_FALSE(metadata.state == AllocationState::ACTIVE,
                             cudaErrorInvalidValue,
                             "Pause: allocation is not ACTIVE");

    if (metadata.enable_cpu_backup) {
      if (metadata.cpu_backup == nullptr) {
        RETURN_IF_CUDA_ERROR(
            cudaMallocHost(&metadata.cpu_backup, metadata.size));
      }
      RETURN_IF_FALSE(metadata.cpu_backup != nullptr, cudaErrorUnknown,
                               "Pause: cpu_backup should not be null");
      RETURN_IF_CUDA_ERROR(cudaMemcpy(
          metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
    }

    RETURN_IF_CU_ERROR_AS_CUDA(
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));
    RETURN_IF_FALSE(metadata.alloc_handle != 0, cudaErrorInvalidValue,
                             "Pause: active allocation handle should not be 0");
    RETURN_IF_CU_ERROR_AS_CUDA(cuMemRelease(metadata.alloc_handle));
    metadata.alloc_handle = 0;

    metadata.state = AllocationState::PAUSED;
  }

  return cudaSuccess;
}

// Resume previously paused regular allocations matching tag filter.
//
// For each target allocation:
// - create and map a new allocation handle at original virtual address
// - restore content from CPU backup when enabled
cudaError_t ContextImpl::Resume(const std::string& tag_filter) {
  const std::lock_guard<std::mutex> lock(mutex_);

  for (auto& kv : allocations_) {
    void* ptr = kv.first;
    AllocationMetadata& metadata = kv.second;

    if (metadata.kind != AllocationKind::REGULAR) {
      continue;
    }

    if (!utils::MatchesTag(tag_filter, metadata.tag)) {
      continue;
    }

    RETURN_IF_FALSE(metadata.state == AllocationState::PAUSED,
                             cudaErrorInvalidValue,
                             "Resume: allocation is not PAUSED");

    CUmemGenericAllocationHandle new_alloc_handle = 0;
    RETURN_IF_CU_ERROR_AS_CUDA(
        vmm::CreateMemoryHandle(&new_alloc_handle, metadata.size,
                                metadata.device));
    RETURN_IF_CU_ERROR_AS_CUDA(
        cuMemMap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size, 0,
                 new_alloc_handle, 0));
    RETURN_IF_CU_ERROR_AS_CUDA(
        vmm::SetAccess(reinterpret_cast<CUdeviceptr>(ptr), metadata.size,
                       metadata.device));

    if (metadata.enable_cpu_backup) {
      RETURN_IF_FALSE(metadata.cpu_backup != nullptr, cudaErrorInvalidValue,
                               "Resume: cpu_backup should not be null");
      RETURN_IF_CUDA_ERROR(cudaMemcpy(
          ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));
      RETURN_IF_CUDA_ERROR(cudaFreeHost(metadata.cpu_backup));
      metadata.cpu_backup = nullptr;
    }

    metadata.state = AllocationState::ACTIVE;
    metadata.alloc_handle = new_alloc_handle;
  }

  return cudaSuccess;
}

// Lookup host backup pointer for GPU range [query_gpu_ptr, query_gpu_ptr + query_size).
//
// Query must be fully covered by one managed REGULAR allocation.
//
// Success behavior:
// - allocation ACTIVE: out_cpu_ptr = nullptr
// - allocation PAUSED with CPU backup: out_cpu_ptr points to host backup at matching offset
//
// Invalid queries and inconsistent paused-state metadata fail fast.
cudaError_t ContextImpl::GetCpuBackupPointer(
    const uint8_t* query_gpu_ptr,
    const uint64_t query_size,
    uint8_t** out_cpu_ptr) {
  RETURN_IF_FALSE(query_gpu_ptr != nullptr, cudaErrorInvalidValue,
                           "GetCpuBackupPointer: query_gpu_ptr should not be null");
  RETURN_IF_FALSE(out_cpu_ptr != nullptr, cudaErrorInvalidValue,
                           "GetCpuBackupPointer: out_cpu_ptr should not be null");

  *out_cpu_ptr = nullptr;

  const std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& kv : allocations_) {
    const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(kv.first);
    const AllocationMetadata& metadata = kv.second;

    if (metadata.kind != AllocationKind::REGULAR) {
      continue;
    }

    const size_t total_size = metadata.size;
    if (base_ptr <= query_gpu_ptr &&
        query_gpu_ptr + query_size <= base_ptr + total_size) {
      if (metadata.state == AllocationState::ACTIVE) {
        return cudaSuccess;
      }

      RETURN_IF_FALSE(
          metadata.cpu_backup != nullptr,
          cudaErrorInvalidValue,
          "GetCpuBackupPointer: allocation is paused but cpu backup is missing");

      const size_t offset = static_cast<size_t>(query_gpu_ptr - base_ptr);
      *out_cpu_ptr = reinterpret_cast<uint8_t*>(metadata.cpu_backup) + offset;
      return cudaSuccess;
    }
  }

  RETURN_IF_FALSE(false, cudaErrorInvalidValue,
                           "GetCpuBackupPointer: matching allocation not found");
  return cudaSuccess;
}

cudaError_t ContextImpl::GetMetadataCountByTag(
    const std::string& tag,
    uint64_t* out_count) {
  RETURN_IF_FALSE(out_count != nullptr, cudaErrorInvalidValue,
                           "GetMetadataCountByTag: out_count should not be null");

  uint64_t count = 0;
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& kv : allocations_) {
    if (kv.second.tag == tag) {
      ++count;
    }
  }

  *out_count = count;
  return cudaSuccess;
}

// Lazily create one minimum-granularity shared handle for a device.
CUresult ContextImpl::GetOrCreateSharedMinimumGranularityHandle(
    const CUdevice device,
    CUmemGenericAllocationHandle* out_handle) {
  RETURN_IF_CU_FALSE(
      out_handle != nullptr, CUDA_ERROR_INVALID_VALUE,
      "GetOrCreateSharedMinimumGranularityHandle: out_handle should not be null");

  auto it = shared_minimum_granularity_handles_.find(device);
  if (it != shared_minimum_granularity_handles_.end()) {
    *out_handle = it->second;
    return CUDA_SUCCESS;
  }

  size_t minimum_granularity = 0;
  RETURN_IF_CU_ERROR(
      vmm::GetVmmMinimumGranularity(device, &minimum_granularity));

  CUmemGenericAllocationHandle handle = 0;
  RETURN_IF_CU_ERROR(
      vmm::CreateMemoryHandle(&handle, minimum_granularity, device));

  shared_minimum_granularity_handles_.emplace(device, handle);
  *out_handle = handle;
  return CUDA_SUCCESS;
}

// Map [address, address + size) to shared empty handle by granularity-sized chunks.
CUresult ContextImpl::MapRangeToEmptyHandle(
    const CUdeviceptr address,
    const size_t size,
    const CUmemGenericAllocationHandle empty_handle,
    const size_t minimum_granularity_bytes) {
  RETURN_IF_CU_FALSE(size > 0, CUDA_ERROR_INVALID_VALUE,
                     "MapRangeToEmptyHandle: size should be > 0");
  RETURN_IF_CU_FALSE(empty_handle != 0, CUDA_ERROR_INVALID_VALUE,
                     "MapRangeToEmptyHandle: empty_handle should not be 0");
  RETURN_IF_CU_FALSE(
      minimum_granularity_bytes > 0, CUDA_ERROR_INVALID_VALUE,
      "MapRangeToEmptyHandle: minimum_granularity_bytes should be > 0");
  RETURN_IF_CU_FALSE(
      size % minimum_granularity_bytes == 0, CUDA_ERROR_INVALID_VALUE,
      "MapRangeToEmptyHandle: size should be aligned to minimum granularity");

  size_t mapped_size = 0;
  while (mapped_size < size) {
    RETURN_IF_CU_ERROR(cuMemMap(address + mapped_size, minimum_granularity_bytes,
                                0, empty_handle, 0));
    mapped_size += minimum_granularity_bytes;
  }

  return CUDA_SUCCESS;
}

CUresult ContextImpl::MallocArena(
    void** ptr,
    const CUdevice device,
    const size_t size,
    const std::string& tag) {

  CUdeviceptr reserved_address = 0;
  RETURN_IF_CU_ERROR(cuMemAddressReserve(&reserved_address, size, 0, 0, 0));

  size_t minimum_granularity_bytes = 0;
  RETURN_IF_CU_ERROR(
      vmm::GetVmmMinimumGranularity(device, &minimum_granularity_bytes));
  RETURN_IF_CU_FALSE(minimum_granularity_bytes > 0, CUDA_ERROR_INVALID_VALUE,
                     "CreateArena: minimum_granularity_bytes should be > 0");

  CUmemGenericAllocationHandle shared_handle = 0;
  RETURN_IF_CU_ERROR(
      GetOrCreateSharedMinimumGranularityHandle(device, &shared_handle));
  RETURN_IF_CU_ERROR(MapRangeToEmptyHandle(
      reserved_address, size, shared_handle, minimum_granularity_bytes));

  *ptr = reinterpret_cast<void*>(reserved_address);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    AllocationMetadata metadata;
    metadata.size = size;
    metadata.device = device;
    metadata.tag = tag;
    metadata.state = AllocationState::ACTIVE;
    metadata.enable_cpu_backup = false;
    metadata.cpu_backup = nullptr;
    metadata.alloc_handle = 0;
    metadata.kind = AllocationKind::ARENA;
    metadata.empty_handle = shared_handle;
    allocations_.emplace(*ptr, std::move(metadata));
  }
  
  return CUDA_SUCCESS;
}

// Activate real mappings for [base + offset, base + offset + size_bytes) on a virtual-only arena.
cudaError_t ContextImpl::ActivateArenaOffsets(
    const std::string& tag,
    const CUdevice device,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "ActivateArenaOffsets: tag should not be empty");
  if (num_offsets == 0) {
    return cudaSuccess;
  }
  RETURN_IF_FALSE(offsets != nullptr, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: offsets should not be null when num_offsets > 0");
  RETURN_IF_FALSE(size_bytes > 0, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: size_bytes should be > 0");

  const std::lock_guard<std::mutex> lock(mutex_);
  void* arena_base_ptr = nullptr;
  AllocationMetadata* arena_metadata = nullptr;
  for (auto& kv : allocations_) {
    if (kv.second.kind != AllocationKind::ARENA) {
      continue;
    }
    if (kv.second.device != device || kv.second.tag != tag) {
      continue;
    }
    RETURN_IF_FALSE(arena_metadata == nullptr, cudaErrorInvalidValue,
                    "ActivateArenaOffsets: multiple matching arenas found");
    arena_base_ptr = kv.first;
    arena_metadata = &kv.second;
  }
  RETURN_IF_FALSE(arena_metadata != nullptr, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: unique arena not found");

  const size_t size = static_cast<size_t>(size_bytes);
  for (uint64_t i = 0; i < num_offsets; ++i) {
    const uint64_t offset = offsets[i];
    const CUdeviceptr address =
        reinterpret_cast<CUdeviceptr>(arena_base_ptr) + static_cast<CUdeviceptr>(offset);
    RETURN_IF_CU_ERROR_AS_CUDA(cuMemUnmap(address, size));

    CUmemGenericAllocationHandle handle = 0;
    RETURN_IF_CU_ERROR_AS_CUDA(vmm::CreateMemoryHandle(&handle, size, device));
    RETURN_IF_CU_ERROR_AS_CUDA(cuMemMap(address, size, 0, handle, 0));
    RETURN_IF_CU_ERROR_AS_CUDA(vmm::SetAccess(address, size, device));

    arena_metadata->arena_offset_handles[offset] = handle;
  }

  return cudaSuccess;
}

// Deactivate real mappings and map empty handle back for requested offsets on a virtual-only arena.
cudaError_t ContextImpl::DeactivateArenaOffsets(
    const std::string& tag,
    const CUdevice device,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: tag should not be empty");
  if (num_offsets == 0) {
    return cudaSuccess;
  }
  RETURN_IF_FALSE(offsets != nullptr, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: offsets should not be null when num_offsets > 0");
  RETURN_IF_FALSE(size_bytes > 0, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: size_bytes should be > 0");
  RETURN_IF_FALSE(
      size_bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      cudaErrorInvalidValue,
      "DeactivateArenaOffsets: size_bytes is too large");

  const std::lock_guard<std::mutex> lock(mutex_);
  void* arena_base_ptr = nullptr;
  AllocationMetadata* arena_metadata = nullptr;
  for (auto& kv : allocations_) {
    if (kv.second.kind != AllocationKind::ARENA) {
      continue;
    }
    if (kv.second.device != device || kv.second.tag != tag) {
      continue;
    }
    RETURN_IF_FALSE(arena_metadata == nullptr, cudaErrorInvalidValue,
                    "DeactivateArenaOffsets: multiple matching arenas found");
    arena_base_ptr = kv.first;
    arena_metadata = &kv.second;
  }
  RETURN_IF_FALSE(arena_metadata != nullptr, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: unique arena not found");

  const size_t size = static_cast<size_t>(size_bytes);
  for (uint64_t i = 0; i < num_offsets; ++i) {
    const uint64_t offset = offsets[i];
    const CUdeviceptr address =
        reinterpret_cast<CUdeviceptr>(arena_base_ptr) + static_cast<CUdeviceptr>(offset);

    RETURN_IF_CU_ERROR_AS_CUDA(cuMemUnmap(address, size));

    const CUmemGenericAllocationHandle handle =
        arena_metadata->arena_offset_handles[offset];

    RETURN_IF_CU_ERROR_AS_CUDA(cuMemRelease(handle));
    size_t minimum_granularity_bytes = 0;
    RETURN_IF_CU_ERROR_AS_CUDA(
        vmm::GetVmmMinimumGranularity(device, &minimum_granularity_bytes));
    RETURN_IF_CU_ERROR_AS_CUDA(MapRangeToEmptyHandle(
        address,
        size,
        arena_metadata->empty_handle,
        minimum_granularity_bytes));

    arena_metadata->arena_offset_handles.erase(offset);
  }

  return cudaSuccess;
}

}  // namespace memsaver::internal
