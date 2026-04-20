#ifndef MEMSAVER_INTERNAL_COMMON_H_
#define MEMSAVER_INTERNAL_COMMON_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#ifndef MEMSAVER_FATAL_ON_ERROR
#define MEMSAVER_FATAL_ON_ERROR 1
#endif

namespace memsaver::internal {

#if defined(__GNUC__) || defined(__clang__)
#define MEMSAVER_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define MEMSAVER_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define MEMSAVER_LIKELY(expr) (expr)
#define MEMSAVER_UNLIKELY(expr) (expr)
#endif

[[noreturn]] inline void FailFast(
    const std::string& message,
    const char* file,
    const char* func,
    int line) {
  std::cerr << "[memsaver] " << message << " file=" << file << " func=" << func
            << " line=" << line << std::endl;
  std::abort();
}

inline void LogError(
    const std::string& message,
    const char* file,
    const char* func,
    int line) {
  std::cerr << "[memsaver] " << message << " file=" << file << " func=" << func
            << " line=" << line << std::endl;
}

inline cudaError_t CuResultToCudaError(const CUresult result) {
  switch (result) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    case CUDA_ERROR_OUT_OF_MEMORY:
      return cudaErrorMemoryAllocation;
    case CUDA_ERROR_INVALID_VALUE:
      return cudaErrorInvalidValue;
    default:
      return cudaErrorUnknown;
  }
}

inline cudaError_t HandleCudaError(
    const cudaError_t error,
    const char* expr,
    const char* file,
    const char* func,
    const int line) {
  if (error == cudaSuccess) {
    return cudaSuccess;
  }

  std::ostringstream oss;
  oss << "CUDA error from " << expr << ": code=" << static_cast<int>(error)
      << " message=" << cudaGetErrorString(error);

#if MEMSAVER_FATAL_ON_ERROR
  FailFast(oss.str(), file, func, line);
#else
  LogError(oss.str(), file, func, line);
#endif
  return error;
}

inline cudaError_t HandleCuError(
    const CUresult result,
    const char* expr,
    const char* file,
    const char* func,
    const int line) {
  if (result == CUDA_SUCCESS) {
    return cudaSuccess;
  }

  const char* error_string = nullptr;
  (void)cuGetErrorString(result, &error_string);

  std::ostringstream oss;
  oss << "CUresult error from " << expr << ": code=" << static_cast<int>(result)
      << " message=" << (error_string == nullptr ? "Unknown error" : error_string);

#if MEMSAVER_FATAL_ON_ERROR
  FailFast(oss.str(), file, func, line);
#else
  LogError(oss.str(), file, func, line);
#endif
  return CuResultToCudaError(result);
}

inline cudaError_t Ensure(
    const bool condition,
    const cudaError_t error_code,
    const char* message,
    const char* file,
    const char* func,
    const int line) {
  if (condition) {
    return cudaSuccess;
  }

  LogError(message, file, func, line);
  return error_code;
}

}  // namespace memsaver::internal

#define MEMSAVER_RETURN_IF_FALSE(COND, ERROR_CODE, MSG)                           \
  do {                                                                             \
    const cudaError_t _status =                                                   \
        memsaver::internal::Ensure((COND), (ERROR_CODE), (MSG), __FILE__,        \
                                   __func__, __LINE__);                           \
    if (_status != cudaSuccess) {                                                 \
      return _status;                                                             \
    }                                                                             \
  } while (false)

#define MEMSAVER_RETURN_IF_CUDA_ERROR(EXPR)                                        \
  do {                                                                              \
    const cudaError_t _status =                                                   \
        memsaver::internal::HandleCudaError((EXPR), #EXPR, __FILE__, __func__,    \
                                            __LINE__);                              \
    if (_status != cudaSuccess) {                                                  \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#define MEMSAVER_RETURN_IF_CU_ERROR(EXPR)                                           \
  do {                                                                              \
    const cudaError_t _status =                                                   \
        memsaver::internal::HandleCuError((EXPR), #EXPR, __FILE__, __func__,      \
                                          __LINE__);                                \
    if (_status != cudaSuccess) {                                                  \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#endif  // MEMSAVER_INTERNAL_COMMON_H_
