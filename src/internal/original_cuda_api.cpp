#include "internal/original_cuda_api.h"

#include <dlfcn.h>

#include <mutex>

#include "internal/utils.h"

namespace memsaver::internal {
namespace {

using CudaMallocFn = cudaError_t (*)(void**, size_t);
using CudaFreeFn = cudaError_t (*)(void*);

constexpr const char* kCudaMallocSymbol = "cudaMalloc";
constexpr const char* kCudaFreeSymbol = "cudaFree";

std::mutex g_symbol_mutex;
CudaMallocFn g_original_cuda_malloc = nullptr;
CudaFreeFn g_original_cuda_free = nullptr;

// Resolve original cudaMalloc symbol and cache the function pointer.
cudaError_t ResolveOriginalCudaMalloc(CudaMallocFn* out_fn) {
  RETURN_IF_FALSE(out_fn != nullptr, cudaErrorInvalidValue,
                  "ResolveOriginalCudaMalloc: out_fn should not be null");

  std::lock_guard<std::mutex> lock(g_symbol_mutex);
  if (g_original_cuda_malloc != nullptr) {
    *out_fn = g_original_cuda_malloc;
    return cudaSuccess;
  }

  (void)dlerror();
  void* symbol = dlsym(RTLD_NEXT, kCudaMallocSymbol);
  if (symbol == nullptr) {
    const char* dl_error = dlerror();
    const std::string message =
        std::string("dlsym failed for cudaMalloc: ") +
        (dl_error == nullptr ? "<null>" : dl_error);
    RETURN_IF_FALSE(false, cudaErrorUnknown, message.c_str());
  }

  g_original_cuda_malloc = reinterpret_cast<CudaMallocFn>(symbol);
  *out_fn = g_original_cuda_malloc;
  return cudaSuccess;
}

// Resolve original cudaFree symbol and cache the function pointer.
cudaError_t ResolveOriginalCudaFree(CudaFreeFn* out_fn) {
  RETURN_IF_FALSE(out_fn != nullptr, cudaErrorInvalidValue,
                  "ResolveOriginalCudaFree: out_fn should not be null");

  std::lock_guard<std::mutex> lock(g_symbol_mutex);
  if (g_original_cuda_free != nullptr) {
    *out_fn = g_original_cuda_free;
    return cudaSuccess;
  }

  (void)dlerror();
  void* symbol = dlsym(RTLD_NEXT, kCudaFreeSymbol);
  if (symbol == nullptr) {
    const char* dl_error = dlerror();
    const std::string message =
        std::string("dlsym failed for cudaFree: ") +
        (dl_error == nullptr ? "<null>" : dl_error);
    RETURN_IF_FALSE(false, cudaErrorUnknown, message.c_str());
  }

  g_original_cuda_free = reinterpret_cast<CudaFreeFn>(symbol);
  *out_fn = g_original_cuda_free;
  return cudaSuccess;
}

}  // namespace

// Allocate via direct cudaMalloc or original CUDA symbol.
cudaError_t OriginalCudaApi::Malloc(
    void** ptr,
    const size_t size,
    const bool use_original_cuda_symbols) {
  RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                  "OriginalCudaApi::Malloc: ptr should not be null");

  if (!use_original_cuda_symbols) {
    return cudaMalloc(ptr, size);
  }

  CudaMallocFn fn = nullptr;
  RETURN_IF_CUDA_ERROR(ResolveOriginalCudaMalloc(&fn));
  return fn(ptr, size);
}

// Free via direct cudaFree or original CUDA symbol.
cudaError_t OriginalCudaApi::Free(void* ptr, const bool use_original_cuda_symbols) {
  if (!use_original_cuda_symbols) {
    return cudaFree(ptr);
  }

  CudaFreeFn fn = nullptr;
  RETURN_IF_CUDA_ERROR(ResolveOriginalCudaFree(&fn));
  return fn(ptr);
}

}  // namespace memsaver::internal
