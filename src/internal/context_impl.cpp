#include "internal/context_impl.h"

#include <cstring>
#include <limits>
#include <utility>

#include "internal/common.h"
#include "internal/cuda_utils.h"
#include "internal/original_cuda_api.h"

namespace memsaver::internal {
namespace {

// All tensor should be aligned to 512 bytes
constexpr size_t kArenaAlignment = 512;

// Convert CUresult to cudaError_t without triggering fail-fast.
inline cudaError_t ToCudaErrorNoAbort(const CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : CuResultToCudaError(result);
}

// Return true when tag filter is empty or exactly matches candidate tag.
bool TagMatches(const std::string& filter, const std::string& candidate) {
  return filter.empty() || filter == candidate;
}

// Best-effort rollback for partially-created regular VMM allocation.
//
// This helper is used only on error paths in MallocRegular.
cudaError_t CleanupRegularAllocationFailure(
    const CUdeviceptr address,
    const size_t size,
    const CUmemGenericAllocationHandle handle) {
  if (address != 0) {
    (void)HandleCuError(
        cuMemAddressFree(address, size), "cuMemAddressFree", __FILE__, __func__, __LINE__);
  }
  if (handle != 0) {
    (void)HandleCuError(cuMemRelease(handle), "cuMemRelease", __FILE__, __func__, __LINE__);
  }
  return cudaErrorUnknown;
}

}  // namespace

// Construct core context. Behavior of unknown-pointer free is configured by
// use_original_cuda_symbols_.
ContextImpl::ContextImpl(const bool use_original_cuda_symbols)
    : use_original_cuda_symbols_(use_original_cuda_symbols) {}

// Tear down all tracked allocation resources and arenas.
ContextImpl::~ContextImpl() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& kv : allocations_) {
    ReleaseAllocationForShutdown(kv.first, kv.second);
  }
  allocations_.clear();

  for (const auto& kv : arenas_) {
    if (kv.second.create_mode == MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED) {
      (void)ReleaseArena(kv.second);
    }
  }
  arenas_.clear();
}

// Destructor-only cleanup path for one regular allocation record.
//
// This best-effort path:
// - unmaps and releases active VMM allocation handles
// - frees reserved virtual address range
// - releases optional pinned-host CPU backup
//
// Arena resources are released separately via ReleaseArena.
void ContextImpl::ReleaseAllocationForShutdown(
    void* ptr,
    const AllocationMetadata& metadata) {
  if (metadata.kind != AllocationKind::REGULAR) {
    return;
  }

  if (metadata.state == AllocationState::ACTIVE) {
    (void)HandleCuError(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size),
                        "cuMemUnmap", __FILE__, __func__, __LINE__);
    if (metadata.alloc_handle != 0) {
      (void)HandleCuError(cuMemRelease(metadata.alloc_handle), "cuMemRelease",
                          __FILE__, __func__, __LINE__);
    }
  }

  (void)HandleCuError(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), metadata.size),
                      "cuMemAddressFree", __FILE__, __func__, __LINE__);

  if (metadata.cpu_backup != nullptr) {
    (void)HandleCudaError(cudaFreeHost(metadata.cpu_backup), "cudaFreeHost",
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
  MEMSAVER_RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                           "ContextImpl::Malloc: ptr should not be null");
  MEMSAVER_RETURN_IF_FALSE(size > 0, cudaErrorInvalidValue,
                           "ContextImpl::Malloc: size should be > 0");

  if (!config.interesting_region) {
    return OriginalCudaApi::Malloc(ptr, size, use_original_cuda_symbols_);
  }

  if (config.allocation_mode == MEMSAVER_ALLOCATION_MODE_ARENA) {
    return MallocArena(ptr, device, size, config.tag);
  }
  return MallocRegular(ptr, device, size, config.tag, config.enable_cpu_backup);
}

// Allocate one regular VMM-backed block and track its metadata.
cudaError_t ContextImpl::MallocRegular(
    void** ptr,
    const CUdevice device,
    const size_t size,
    const std::string& tag,
    const bool enable_cpu_backup) {
  size_t aligned_size = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::GetVmmAlignedSize(size, device, &aligned_size));

  CUmemGenericAllocationHandle handle = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::CreateMemoryHandle(&handle, aligned_size, device));

  CUdeviceptr reserved_address = 0;
  CUresult cu_result =
      cuMemAddressReserve(&reserved_address, aligned_size, 0, 0, 0);
  if (cu_result != CUDA_SUCCESS) {
    (void)HandleCuError(cuMemRelease(handle), "cuMemRelease", __FILE__, __func__,
                        __LINE__);
    return HandleCuError(cu_result, "cuMemAddressReserve", __FILE__, __func__,
                         __LINE__);
  }

  cu_result = cuMemMap(reserved_address, aligned_size, 0, handle, 0);
  if (cu_result != CUDA_SUCCESS) {
    (void)CleanupRegularAllocationFailure(reserved_address, aligned_size, handle);
    return HandleCuError(cu_result, "cuMemMap", __FILE__, __func__, __LINE__);
  }

  const cudaError_t access_status =
      cuda_utils::SetAccess(reinterpret_cast<void*>(reserved_address), aligned_size,
                            device);
  if (access_status != cudaSuccess) {
    (void)HandleCuError(cuMemUnmap(reserved_address, aligned_size), "cuMemUnmap",
                        __FILE__, __func__, __LINE__);
    (void)CleanupRegularAllocationFailure(reserved_address, aligned_size, handle);
    return access_status;
  }

  *ptr = reinterpret_cast<void*>(reserved_address);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    allocations_.emplace(
        *ptr,
        AllocationMetadata{
            aligned_size,
            device,
            tag,
            AllocationState::ACTIVE,
            enable_cpu_backup,
            nullptr,
            handle,
            AllocationKind::REGULAR,
        });
  }

  return cudaSuccess;
}

// Lazily create one minimum-granularity shared handle for a device.
cudaError_t ContextImpl::GetOrCreateSharedMinimumGranularityHandle(
    const CUdevice device,
    CUmemGenericAllocationHandle* out_handle) {
  MEMSAVER_RETURN_IF_FALSE(out_handle != nullptr, cudaErrorInvalidValue,
                           "GetOrCreateSharedMinimumGranularityHandle: out_handle should not be null");

  auto it = shared_minimum_granularity_handles_.find(device);
  if (it != shared_minimum_granularity_handles_.end()) {
    *out_handle = it->second;
    return cudaSuccess;
  }

  size_t minimum_granularity = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::GetVmmMinimumGranularity(device, &minimum_granularity));

  CUmemGenericAllocationHandle handle = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::CreateMemoryHandle(&handle, minimum_granularity, device));

  shared_minimum_granularity_handles_.emplace(device, handle);
  *out_handle = handle;
  return cudaSuccess;
}

// Map [address, address + size) to shared empty handle by granularity-sized chunks.
cudaError_t ContextImpl::MapRangeToEmptyHandle(
    const CUdeviceptr address,
    const size_t size,
    const CUmemGenericAllocationHandle empty_handle,
    const size_t minimum_granularity_bytes) {
  MEMSAVER_RETURN_IF_FALSE(size > 0, cudaErrorInvalidValue,
                           "MapRangeToEmptyHandle: size should be > 0");
  MEMSAVER_RETURN_IF_FALSE(empty_handle != 0, cudaErrorInvalidValue,
                           "MapRangeToEmptyHandle: empty_handle should not be 0");
  MEMSAVER_RETURN_IF_FALSE(minimum_granularity_bytes > 0, cudaErrorInvalidValue,
                           "MapRangeToEmptyHandle: minimum_granularity_bytes should be > 0");
  MEMSAVER_RETURN_IF_FALSE(size % minimum_granularity_bytes == 0,
                           cudaErrorInvalidValue,
                           "MapRangeToEmptyHandle: size should be aligned to minimum granularity");

  size_t mapped_size = 0;
  while (mapped_size < size) {
    const CUresult map_result = cuMemMap(
        address + mapped_size, minimum_granularity_bytes, 0, empty_handle, 0);
    if (map_result != CUDA_SUCCESS) {
      size_t rollback_size = 0;
      while (rollback_size < mapped_size) {
        (void)cuMemUnmap(address + rollback_size, minimum_granularity_bytes);
        rollback_size += minimum_granularity_bytes;
      }
      return ToCudaErrorNoAbort(map_result);
    }
    mapped_size += minimum_granularity_bytes;
  }

  return cudaSuccess;
}

// Create a new arena mapping for (device, tag).
cudaError_t ContextImpl::CreateArena(
    const ArenaKey& key,
    const size_t capacity_bytes,
    const memsaver_arena_create_mode_t create_mode) {
  MEMSAVER_RETURN_IF_FALSE(
      create_mode == MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED ||
          create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY,
      cudaErrorInvalidValue,
      "CreateArena: create_mode is invalid");

  size_t aligned_capacity = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::GetVmmAlignedSize(capacity_bytes, key.device, &aligned_capacity));

  CUdeviceptr reserved_address = 0;
  CUresult cu_result =
      cuMemAddressReserve(&reserved_address, aligned_capacity, 0, 0, 0);
  if (cu_result != CUDA_SUCCESS) {
    return HandleCuError(cu_result, "cuMemAddressReserve", __FILE__, __func__,
                         __LINE__);
  }

  if (create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
    size_t minimum_granularity_bytes = 0;
    MEMSAVER_RETURN_IF_CUDA_ERROR(
        cuda_utils::GetVmmMinimumGranularity(key.device, &minimum_granularity_bytes));
    MEMSAVER_RETURN_IF_FALSE(minimum_granularity_bytes > 0, cudaErrorInvalidValue,
                             "CreateArena: minimum_granularity_bytes should be > 0");
    MEMSAVER_RETURN_IF_FALSE(
        aligned_capacity % minimum_granularity_bytes == 0,
        cudaErrorInvalidValue,
        "CreateArena: aligned_capacity should be aligned to minimum granularity");

    CUmemGenericAllocationHandle shared_handle = 0;
    const cudaError_t status =
        GetOrCreateSharedMinimumGranularityHandle(key.device, &shared_handle);
    if (status != cudaSuccess) {
      (void)HandleCuError(cuMemAddressFree(reserved_address, aligned_capacity),
                          "cuMemAddressFree", __FILE__, __func__, __LINE__);
      return status;
    }

    const cudaError_t map_empty_status = MapRangeToEmptyHandle(
        reserved_address, aligned_capacity, shared_handle, minimum_granularity_bytes);
    if (map_empty_status != cudaSuccess) {
      (void)HandleCuError(cuMemAddressFree(reserved_address, aligned_capacity),
                          "cuMemAddressFree", __FILE__, __func__, __LINE__);
      return map_empty_status;
    }

    arenas_.emplace(
        key,
        Arena{
            reinterpret_cast<void*>(reserved_address),
            aligned_capacity,
            0,
            0,
            0,
            shared_handle,
            minimum_granularity_bytes,
            {},
            MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY,
        });
    return cudaSuccess;
  }

  CUmemGenericAllocationHandle handle = 0;
  const cudaError_t handle_status =
      cuda_utils::CreateMemoryHandle(&handle, aligned_capacity, key.device);
  if (handle_status != cudaSuccess) {
    (void)HandleCuError(cuMemAddressFree(reserved_address, aligned_capacity),
                        "cuMemAddressFree", __FILE__, __func__, __LINE__);
    return handle_status;
  }

  cu_result = cuMemMap(reserved_address, aligned_capacity, 0, handle, 0);
  if (cu_result != CUDA_SUCCESS) {
    (void)HandleCuError(cuMemAddressFree(reserved_address, aligned_capacity),
                        "cuMemAddressFree", __FILE__, __func__, __LINE__);
    (void)HandleCuError(cuMemRelease(handle), "cuMemRelease", __FILE__, __func__,
                        __LINE__);
    return HandleCuError(cu_result, "cuMemMap", __FILE__, __func__, __LINE__);
  }

  const cudaError_t access_status =
      cuda_utils::SetAccess(reinterpret_cast<void*>(reserved_address),
                            aligned_capacity, key.device);
  if (access_status != cudaSuccess) {
    (void)HandleCuError(cuMemUnmap(reserved_address, aligned_capacity),
                        "cuMemUnmap", __FILE__, __func__, __LINE__);
    (void)HandleCuError(cuMemAddressFree(reserved_address, aligned_capacity),
                        "cuMemAddressFree", __FILE__, __func__, __LINE__);
    (void)HandleCuError(cuMemRelease(handle), "cuMemRelease", __FILE__, __func__,
                        __LINE__);
    return access_status;
  }

  arenas_.emplace(
      key,
      Arena{
          reinterpret_cast<void*>(reserved_address),
          aligned_capacity,
          0,
          0,
          handle,
          0,
          0,
          {},
          MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED,
      });
  return cudaSuccess;
}

// Release one fully-mapped arena mapping and its allocation handle.
cudaError_t ContextImpl::ReleaseArena(const Arena& arena) {
  if (arena.base_ptr == nullptr || arena.capacity == 0) {
    return cudaSuccess;
  }

  MEMSAVER_RETURN_IF_CU_ERROR(
      cuMemUnmap(reinterpret_cast<CUdeviceptr>(arena.base_ptr), arena.capacity));
  MEMSAVER_RETURN_IF_CU_ERROR(cuMemRelease(arena.alloc_handle));
  MEMSAVER_RETURN_IF_CU_ERROR(cuMemAddressFree(
      reinterpret_cast<CUdeviceptr>(arena.base_ptr), arena.capacity));
  return cudaSuccess;
}

// Configure or reconfigure arena capacity for (device, tag).
//
// Reconfiguration is allowed only when there are no live allocations.
cudaError_t ContextImpl::ConfigureArena(
    const std::string& tag,
    const CUdevice device,
    const uint64_t capacity_bytes,
    const memsaver_arena_create_mode_t create_mode) {
  MEMSAVER_RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "ConfigureArena: tag should not be empty");
  MEMSAVER_RETURN_IF_FALSE(capacity_bytes > 0, cudaErrorInvalidValue,
                           "ConfigureArena: capacity_bytes should be > 0");
  MEMSAVER_RETURN_IF_FALSE(
      create_mode == MEMSAVER_ARENA_CREATE_MODE_FULLY_MAPPED ||
          create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY,
      cudaErrorInvalidValue,
      "ConfigureArena: create_mode is invalid");

  const std::lock_guard<std::mutex> lock(mutex_);
  ArenaKey key{device, tag};

  size_t aligned_capacity = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(
      cuda_utils::GetVmmAlignedSize(capacity_bytes, device, &aligned_capacity));

  auto it = arenas_.find(key);
  if (it != arenas_.end()) {
    if (it->second.create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
      return cudaErrorNotSupported;
    }

    MEMSAVER_RETURN_IF_FALSE(it->second.live_allocs == 0, cudaErrorInvalidValue,
                             "ConfigureArena: arena has live allocations");

    if (it->second.capacity == aligned_capacity &&
        it->second.create_mode == create_mode) {
      return cudaSuccess;
    }

    MEMSAVER_RETURN_IF_CUDA_ERROR(ReleaseArena(it->second));
    arenas_.erase(it);
  }

  return CreateArena(key, capacity_bytes, create_mode);
}

// Allocate from an already configured arena by linear bump pointer.
cudaError_t ContextImpl::MallocArena(
    void** ptr,
    const CUdevice device,
    const size_t size,
    const std::string& tag) {
  const std::lock_guard<std::mutex> lock(mutex_);

  ArenaKey key{device, tag};
  auto arena_it = arenas_.find(key);
  if (arena_it == arenas_.end()) {
    return cudaErrorMemoryAllocation;
  }

  Arena& arena = arena_it->second;
  if (arena.create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
    return cudaErrorNotSupported;
  }

  const size_t aligned_offset =
      cuda_utils::RoundUp(arena.offset, kArenaAlignment);

  if (aligned_offset + size > arena.capacity) {
    return cudaErrorMemoryAllocation;
  }

  uint8_t* base = reinterpret_cast<uint8_t*>(arena.base_ptr);
  *ptr = base + aligned_offset;
  arena.offset = aligned_offset + size;
  arena.live_allocs += 1;

  allocations_.emplace(
      *ptr,
      AllocationMetadata{
          size,
          device,
          tag,
          AllocationState::ACTIVE,
          false,
          nullptr,
          0,
          AllocationKind::ARENA,
      });

  return cudaSuccess;
}

// Reset arena bump offset to 0 after a completed arena cycle.
//
// Reset is rejected if live allocations still exist.
cudaError_t ContextImpl::ResetArena(const std::string& tag, const CUdevice device) {
  MEMSAVER_RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "ResetArena: tag should not be empty");

  const std::lock_guard<std::mutex> lock(mutex_);

  ArenaKey key{device, tag};
  auto arena_it = arenas_.find(key);
  MEMSAVER_RETURN_IF_FALSE(arena_it != arenas_.end(), cudaErrorInvalidValue,
                           "ResetArena: arena not found");
  if (arena_it->second.create_mode == MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
    return cudaSuccess;
  }
  MEMSAVER_RETURN_IF_FALSE(arena_it->second.live_allocs == 0, cudaErrorInvalidValue,
                           "ResetArena: arena still has live allocations");

  arena_it->second.offset = 0;
  return cudaSuccess;
}

// Activate real mappings for [base + offset, base + offset + size_bytes) on a virtual-only arena.
cudaError_t ContextImpl::ActivateArenaOffsets(
    const std::string& tag,
    const CUdevice device,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  MEMSAVER_RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "ActivateArenaOffsets: tag should not be empty");
  if (num_offsets == 0) {
    return cudaSuccess;
  }
  MEMSAVER_RETURN_IF_FALSE(offsets != nullptr, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: offsets should not be null when num_offsets > 0");
  MEMSAVER_RETURN_IF_FALSE(size_bytes > 0, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: size_bytes should be > 0");
  MEMSAVER_RETURN_IF_FALSE(
      size_bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      cudaErrorInvalidValue,
      "ActivateArenaOffsets: size_bytes is too large");

  const std::lock_guard<std::mutex> lock(mutex_);

  ArenaKey key{device, tag};
  auto arena_it = arenas_.find(key);
  MEMSAVER_RETURN_IF_FALSE(arena_it != arenas_.end(), cudaErrorInvalidValue,
                           "ActivateArenaOffsets: arena not found");

  Arena& arena = arena_it->second;
  if (arena.create_mode != MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
    return cudaErrorNotSupported;
  }

  MEMSAVER_RETURN_IF_FALSE(arena.minimum_granularity_bytes > 0, cudaErrorInvalidValue,
                           "ActivateArenaOffsets: minimum_granularity_bytes should be > 0");
  const size_t size = static_cast<size_t>(size_bytes);
  for (uint64_t i = 0; i < num_offsets; ++i) {
    const size_t offset = static_cast<size_t>(offsets[i]);
    if (arena.active_bindings.find(offset) != arena.active_bindings.end()) {
      return cudaErrorInvalidValue;
    }

    const CUdeviceptr address = reinterpret_cast<CUdeviceptr>(arena.base_ptr) + offset;
    const CUresult unmap_result = cuMemUnmap(address, size);
    if (unmap_result != CUDA_SUCCESS) {
      return ToCudaErrorNoAbort(unmap_result);
    }

    CUmemGenericAllocationHandle handle = 0;
    MEMSAVER_RETURN_IF_CUDA_ERROR(
        cuda_utils::CreateMemoryHandle(&handle, size, device));

    const CUresult map_result = cuMemMap(address, size, 0, handle, 0);
    if (map_result != CUDA_SUCCESS) {
      (void)cuMemRelease(handle);
      return ToCudaErrorNoAbort(map_result);
    }

    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    const CUresult access_result =
        cuMemSetAccess(address, size, &access_desc, 1);
    if (access_result != CUDA_SUCCESS) {
      (void)cuMemUnmap(address, size);
      (void)cuMemRelease(handle);
      return ToCudaErrorNoAbort(access_result);
    }

    const auto [_, inserted] =
        arena.active_bindings.emplace(offset, ArenaActiveBinding{size, handle});
    MEMSAVER_RETURN_IF_FALSE(inserted, cudaErrorInvalidValue,
                             "ActivateArenaOffsets: duplicated active binding offset");
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
  MEMSAVER_RETURN_IF_FALSE(!tag.empty(), cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: tag should not be empty");
  if (num_offsets == 0) {
    return cudaSuccess;
  }
  MEMSAVER_RETURN_IF_FALSE(offsets != nullptr, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: offsets should not be null when num_offsets > 0");
  MEMSAVER_RETURN_IF_FALSE(size_bytes > 0, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: size_bytes should be > 0");
  MEMSAVER_RETURN_IF_FALSE(
      size_bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      cudaErrorInvalidValue,
      "DeactivateArenaOffsets: size_bytes is too large");

  const std::lock_guard<std::mutex> lock(mutex_);

  ArenaKey key{device, tag};
  auto arena_it = arenas_.find(key);
  MEMSAVER_RETURN_IF_FALSE(arena_it != arenas_.end(), cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: arena not found");

  Arena& arena = arena_it->second;
  if (arena.create_mode != MEMSAVER_ARENA_CREATE_MODE_VIRTUAL_ONLY) {
    return cudaErrorNotSupported;
  }

  MEMSAVER_RETURN_IF_FALSE(arena.empty_handle != 0, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: empty_handle should not be 0");
  MEMSAVER_RETURN_IF_FALSE(arena.minimum_granularity_bytes > 0, cudaErrorInvalidValue,
                           "DeactivateArenaOffsets: minimum_granularity_bytes should be > 0");
  const size_t size = static_cast<size_t>(size_bytes);
  for (uint64_t i = 0; i < num_offsets; ++i) {
    const size_t offset = static_cast<size_t>(offsets[i]);
    auto binding_it = arena.active_bindings.find(offset);
    MEMSAVER_RETURN_IF_FALSE(binding_it != arena.active_bindings.end(), cudaErrorInvalidValue,
                             "DeactivateArenaOffsets: active binding disappeared unexpectedly");
    MEMSAVER_RETURN_IF_FALSE(binding_it->second.size == size,
                             cudaErrorInvalidValue,
                             "DeactivateArenaOffsets: requested range size does not match active binding");

    const CUdeviceptr address = reinterpret_cast<CUdeviceptr>(arena.base_ptr) + offset;
    const CUmemGenericAllocationHandle handle = binding_it->second.alloc_handle;

    const CUresult unmap_result = cuMemUnmap(address, size);
    if (unmap_result != CUDA_SUCCESS) {
      return ToCudaErrorNoAbort(unmap_result);
    }
    const CUresult release_result = cuMemRelease(handle);
    if (release_result != CUDA_SUCCESS) {
      return ToCudaErrorNoAbort(release_result);
    }
    MEMSAVER_RETURN_IF_CUDA_ERROR(MapRangeToEmptyHandle(
        address,
        size,
        arena.empty_handle,
        arena.minimum_granularity_bytes));

    arena.active_bindings.erase(binding_it);
  }

  return cudaSuccess;
}

// Free pointer managed by this context.
//
// Behavior:
// - unknown pointer: forward to real cudaFree path
// - arena pointer: decrement live_allocs only
// - regular pointer: unmap/release/free VMM resources and optional CPU backup
cudaError_t ContextImpl::Free(void* ptr) {
  if (ptr == nullptr) {
    return cudaSuccess;
  }

  AllocationMetadata metadata;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto metadata_it = allocations_.find(ptr);
    if (metadata_it == allocations_.end()) {
      return OriginalCudaApi::Free(ptr, use_original_cuda_symbols_);
    }

    metadata = metadata_it->second;
    allocations_.erase(metadata_it);

    if (metadata.kind == AllocationKind::ARENA) {
      ArenaKey key{metadata.device, metadata.tag};
      auto arena_it = arenas_.find(key);
      MEMSAVER_RETURN_IF_FALSE(arena_it != arenas_.end(), cudaErrorInvalidValue,
                               "Free: arena not found");
      MEMSAVER_RETURN_IF_FALSE(arena_it->second.live_allocs > 0, cudaErrorInvalidValue,
                               "Free: arena live_allocs already 0");
      arena_it->second.live_allocs -= 1;
      return cudaSuccess;
    }
  }

  if (metadata.state == AllocationState::ACTIVE) {
    MEMSAVER_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
    MEMSAVER_RETURN_IF_CU_ERROR(
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));
    if (metadata.alloc_handle != 0) {
      MEMSAVER_RETURN_IF_CU_ERROR(cuMemRelease(metadata.alloc_handle));
    }
  }

  MEMSAVER_RETURN_IF_CU_ERROR(
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));

  if (metadata.cpu_backup != nullptr) {
    MEMSAVER_RETURN_IF_CUDA_ERROR(cudaFreeHost(metadata.cpu_backup));
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

    if (!TagMatches(tag_filter, metadata.tag)) {
      continue;
    }

    MEMSAVER_RETURN_IF_FALSE(metadata.state == AllocationState::ACTIVE,
                             cudaErrorInvalidValue,
                             "Pause: allocation is not ACTIVE");

    if (metadata.enable_cpu_backup) {
      if (metadata.cpu_backup == nullptr) {
        MEMSAVER_RETURN_IF_CUDA_ERROR(
            cudaMallocHost(&metadata.cpu_backup, metadata.size));
      }
      MEMSAVER_RETURN_IF_FALSE(metadata.cpu_backup != nullptr, cudaErrorUnknown,
                               "Pause: cpu_backup should not be null");
      MEMSAVER_RETURN_IF_CUDA_ERROR(cudaMemcpy(
          metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
    }

    MEMSAVER_RETURN_IF_CU_ERROR(
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size));
    if (metadata.alloc_handle != 0) {
      MEMSAVER_RETURN_IF_CU_ERROR(cuMemRelease(metadata.alloc_handle));
      metadata.alloc_handle = 0;
    }

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

    if (!TagMatches(tag_filter, metadata.tag)) {
      continue;
    }

    MEMSAVER_RETURN_IF_FALSE(metadata.state == AllocationState::PAUSED,
                             cudaErrorInvalidValue,
                             "Resume: allocation is not PAUSED");

    CUmemGenericAllocationHandle new_alloc_handle = 0;
    MEMSAVER_RETURN_IF_CUDA_ERROR(
        cuda_utils::CreateMemoryHandle(&new_alloc_handle, metadata.size,
                                       metadata.device));

    const CUresult map_result =
        cuMemMap(reinterpret_cast<CUdeviceptr>(ptr), metadata.size, 0,
                 new_alloc_handle, 0);
    if (map_result != CUDA_SUCCESS) {
      (void)HandleCuError(cuMemRelease(new_alloc_handle), "cuMemRelease", __FILE__,
                          __func__, __LINE__);
      return HandleCuError(map_result, "cuMemMap", __FILE__, __func__, __LINE__);
    }

    const cudaError_t access_status =
        cuda_utils::SetAccess(ptr, metadata.size, metadata.device);
    if (access_status != cudaSuccess) {
      (void)HandleCuError(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr),
                                     metadata.size),
                          "cuMemUnmap", __FILE__, __func__, __LINE__);
      (void)HandleCuError(cuMemRelease(new_alloc_handle), "cuMemRelease", __FILE__,
                          __func__, __LINE__);
      return access_status;
    }

    if (metadata.enable_cpu_backup) {
      MEMSAVER_RETURN_IF_FALSE(metadata.cpu_backup != nullptr, cudaErrorInvalidValue,
                               "Resume: cpu_backup should not be null");
      MEMSAVER_RETURN_IF_CUDA_ERROR(cudaMemcpy(
          ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));
      MEMSAVER_RETURN_IF_CUDA_ERROR(cudaFreeHost(metadata.cpu_backup));
      metadata.cpu_backup = nullptr;
    }

    metadata.state = AllocationState::ACTIVE;
    metadata.alloc_handle = new_alloc_handle;
  }

  return cudaSuccess;
}

// Resolve CPU backup pointer for [query_gpu_ptr, query_gpu_ptr + query_size).
//
// Returns:
// - cudaSuccess with out_cpu_ptr == nullptr when allocation is ACTIVE
// - cudaSuccess with valid host pointer when allocation is PAUSED with backup
// - cudaErrorInvalidValue when no covering allocation exists
cudaError_t ContextImpl::GetCpuBackupPointer(
    const uint8_t* query_gpu_ptr,
    const uint64_t query_size,
    uint8_t** out_cpu_ptr) {
  MEMSAVER_RETURN_IF_FALSE(query_gpu_ptr != nullptr, cudaErrorInvalidValue,
                           "GetCpuBackupPointer: query_gpu_ptr should not be null");
  MEMSAVER_RETURN_IF_FALSE(out_cpu_ptr != nullptr, cudaErrorInvalidValue,
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

      MEMSAVER_RETURN_IF_FALSE(
          metadata.cpu_backup != nullptr,
          cudaErrorInvalidValue,
          "GetCpuBackupPointer: allocation is paused but cpu backup is missing");

      const size_t offset = static_cast<size_t>(query_gpu_ptr - base_ptr);
      *out_cpu_ptr = reinterpret_cast<uint8_t*>(metadata.cpu_backup) + offset;
      return cudaSuccess;
    }
  }

  MEMSAVER_RETURN_IF_FALSE(false, cudaErrorInvalidValue,
                           "GetCpuBackupPointer: matching allocation not found");
  return cudaErrorInvalidValue;
}

}  // namespace memsaver::internal
