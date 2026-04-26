#ifndef INTERNAL_VMM_H_
#define INTERNAL_VMM_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>

#include "internal/utils.h"

namespace memsaver::internal::vmm {

constexpr size_t kGlobalGranularityBytes = 2ULL * 1024ULL * 1024ULL;

inline CUresult CreateMemoryHandle(
    CUmemGenericAllocationHandle* out_handle,
    const size_t size,
    const CUdevice device) {
  RETURN_IF_CU_FALSE(out_handle != nullptr, CUDA_ERROR_INVALID_VALUE,
                     "CreateMemoryHandle: out_handle should not be null");

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  RETURN_IF_CU_ERROR(cuMemCreate(out_handle, size, &prop, 0));
  return CUDA_SUCCESS;
}

inline CUresult SetAccess(
    const CUdeviceptr ptr,
    const size_t size,
    const CUdevice device) {
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  RETURN_IF_CU_ERROR(cuMemSetAccess(ptr, size, &access_desc, 1));
  return CUDA_SUCCESS;
}

inline CUresult GetVmmMinimumGranularity(
    const CUdevice device,
    size_t* out_granularity) {
  RETURN_IF_CU_FALSE(out_granularity != nullptr, CUDA_ERROR_INVALID_VALUE,
                     "GetVmmMinimumGranularity: out_granularity should not be null");
  (void)device;
  *out_granularity = kGlobalGranularityBytes;
  return CUDA_SUCCESS;
}

}  // namespace memsaver::internal::vmm

#endif  // INTERNAL_VMM_H_
