#ifndef INTERNAL_VMM_H_
#define INTERNAL_VMM_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>

#include "internal/utils.h"

namespace memsaver::internal::vmm {

constexpr size_t kGlobalGranularityBytes = 2ULL * 1024ULL * 1024ULL;

inline size_t AlignUp(const size_t value, const size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

inline cudaError_t CreateMemoryHandle(
    CUmemGenericAllocationHandle* out_handle,
    const size_t size,
    const CUdevice device) {
  RETURN_IF_FALSE(out_handle != nullptr, cudaErrorInvalidValue,
                  "CreateMemoryHandle: out_handle should not be null");

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  const CUresult result = cuMemCreate(out_handle, size, &prop, 0);
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    return cudaErrorMemoryAllocation;
  }
  RETURN_IF_CU_ERROR(result);
  return cudaSuccess;
}

inline cudaError_t SetAccess(
    void* ptr,
    const size_t size,
    const CUdevice device) {
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  RETURN_IF_CU_ERROR(cuMemSetAccess(
      reinterpret_cast<CUdeviceptr>(ptr), size, &access_desc, 1));
  return cudaSuccess;
}

inline cudaError_t RollbackAllocationFailure(
    const CUdeviceptr address,
    const size_t size,
    const CUmemGenericAllocationHandle handle) {
  if (address != 0) {
    (void)utils::CheckCu(cuMemAddressFree(address, size), "cuMemAddressFree",
                         __FILE__, __func__, __LINE__);
  }
  if (handle != 0) {
    (void)utils::CheckCu(cuMemRelease(handle), "cuMemRelease", __FILE__,
                         __func__, __LINE__);
  }
  return cudaErrorUnknown;
}

inline cudaError_t GetVmmAlignedSize(
    const size_t size,
    const CUdevice device,
    size_t* out_size) {
  RETURN_IF_FALSE(out_size != nullptr, cudaErrorInvalidValue,
                  "GetVmmAlignedSize: out_size should not be null");
  (void)device;
  *out_size = AlignUp(size, kGlobalGranularityBytes);
  return cudaSuccess;
}

inline cudaError_t GetVmmMinimumGranularity(
    const CUdevice device,
    size_t* out_granularity) {
  RETURN_IF_FALSE(out_granularity != nullptr, cudaErrorInvalidValue,
                  "GetVmmMinimumGranularity: out_granularity should not be null");
  (void)device;
  *out_granularity = kGlobalGranularityBytes;
  return cudaSuccess;
}

}  // namespace memsaver::internal::vmm

#endif  // INTERNAL_VMM_H_
