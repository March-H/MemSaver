#ifndef MEMSAVER_INTERNAL_CUDA_UTILS_H_
#define MEMSAVER_INTERNAL_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>

#include "internal/common.h"

namespace memsaver::internal::cuda_utils {

/** Global fixed VMM granularity (2MB). */
constexpr size_t kGlobalVmmGranularityBytes = 2ULL * 1024ULL * 1024ULL;

/** Round up value to alignment (alignment=0 keeps original value). */
inline size_t RoundUp(const size_t value, const size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

/** Resolve current CUDA device into CUdevice. */
inline cudaError_t GetCurrentDevice(CUdevice* out_device) {
  MEMSAVER_RETURN_IF_FALSE(out_device != nullptr, cudaErrorInvalidValue,
                           "GetCurrentDevice: out_device should not be null");
  int device_ordinal = 0;
  MEMSAVER_RETURN_IF_CUDA_ERROR(cudaGetDevice(&device_ordinal));
  MEMSAVER_RETURN_IF_CU_ERROR(cuDeviceGet(out_device, device_ordinal));
  return cudaSuccess;
}

/** Resolve device by ordinal into CUdevice. */
inline cudaError_t GetDeviceByOrdinal(const int device_ordinal, CUdevice* out_device) {
  MEMSAVER_RETURN_IF_FALSE(out_device != nullptr, cudaErrorInvalidValue,
                           "GetDeviceByOrdinal: out_device should not be null");
  MEMSAVER_RETURN_IF_CU_ERROR(cuDeviceGet(out_device, device_ordinal));
  return cudaSuccess;
}

/** Create VMM allocation handle for device-local memory. */
inline cudaError_t CreateMemoryHandle(
    CUmemGenericAllocationHandle* out_handle,
    const size_t size,
    const CUdevice device) {
  MEMSAVER_RETURN_IF_FALSE(out_handle != nullptr, cudaErrorInvalidValue,
                           "CreateMemoryHandle: out_handle should not be null");

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  const CUresult result = cuMemCreate(out_handle, size, &prop, 0);
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    return cudaErrorMemoryAllocation;
  }
  MEMSAVER_RETURN_IF_CU_ERROR(result);
  return cudaSuccess;
}

/** Set device read/write access on a mapped virtual address range. */
inline cudaError_t SetAccess(
    void* ptr,
    const size_t size,
    const CUdevice device) {
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MEMSAVER_RETURN_IF_CU_ERROR(cuMemSetAccess(
      reinterpret_cast<CUdeviceptr>(ptr), size, &access_desc, 1));
  return cudaSuccess;
}

/** Align requested size to VMM granularity for target device. */
inline cudaError_t GetVmmAlignedSize(
    const size_t size,
    const CUdevice device,
    size_t* out_size) {
  MEMSAVER_RETURN_IF_FALSE(out_size != nullptr, cudaErrorInvalidValue,
                           "GetVmmAlignedSize: out_size should not be null");
  (void)device;
  *out_size = RoundUp(size, kGlobalVmmGranularityBytes);
  return cudaSuccess;
}

/** Query minimum VMM allocation granularity for target device. */
inline cudaError_t GetVmmMinimumGranularity(
    const CUdevice device,
    size_t* out_granularity) {
  MEMSAVER_RETURN_IF_FALSE(out_granularity != nullptr, cudaErrorInvalidValue,
                           "GetVmmMinimumGranularity: out_granularity should not be null");
  (void)device;
  *out_granularity = kGlobalVmmGranularityBytes;
  return cudaSuccess;
}

}  // namespace memsaver::internal::cuda_utils

#endif  // MEMSAVER_INTERNAL_CUDA_UTILS_H_
