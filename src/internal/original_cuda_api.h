#ifndef MEMSAVER_INTERNAL_ORIGINAL_CUDA_API_H_
#define MEMSAVER_INTERNAL_ORIGINAL_CUDA_API_H_

#include <cuda_runtime_api.h>

namespace memsaver::internal {

/** Utility wrapper for calling CUDA allocation/free APIs. */
class OriginalCudaApi {
 public:
  /**
   * Allocate GPU memory.
   *
   * @param ptr Output pointer.
   * @param size Size in bytes.
   * @param use_original_cuda_symbols If true, resolve and call original CUDA
   *        symbol via dynamic linker.
   */
  static cudaError_t Malloc(void** ptr, size_t size, bool use_original_cuda_symbols);
  /**
   * Free GPU memory.
   *
   * @param ptr Pointer to free.
   * @param use_original_cuda_symbols If true, resolve and call original CUDA
   *        symbol via dynamic linker.
   */
  static cudaError_t Free(void* ptr, bool use_original_cuda_symbols);
};

}  // namespace memsaver::internal

#endif  // MEMSAVER_INTERNAL_ORIGINAL_CUDA_API_H_
