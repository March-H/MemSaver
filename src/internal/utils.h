#ifndef INTERNAL_UTILS_H_
#define INTERNAL_UTILS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <string>

#ifndef FATAL_ON_ERROR
#ifdef MEMSAVER_FATAL_ON_ERROR
#define FATAL_ON_ERROR MEMSAVER_FATAL_ON_ERROR
#else
#define FATAL_ON_ERROR 1
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define LIKELY(expr) (expr)
#define UNLIKELY(expr) (expr)
#endif

namespace memsaver::internal::utils {

inline void LogFailure(
    const std::string& message,
    const char* file,
    const char* func,
    const int line,
    const bool fatal = false) {
  std::cerr << "[memsaver] " << message << " file=" << file << " func=" << func
            << " line=" << line << std::endl;
  if (fatal) {
    std::abort();
  }
}

inline cudaError_t FailCuda(
    const cudaError_t error_code,
    const std::string& message,
    const char* file,
    const char* func,
    const int line,
    const bool fatal = false) {
  LogFailure(message, file, func, line, fatal);
  return error_code;
}

inline CUresult FailCu(
    const CUresult error_code,
    const std::string& message,
    const char* file,
    const char* func,
    const int line,
    const bool fatal = false) {
  LogFailure(message, file, func, line, fatal);
  return error_code;
}

inline cudaError_t ConvertCuResult(const CUresult result) {
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

inline cudaError_t CheckCuda(
    const cudaError_t error,
    const char* expr,
    const char* file,
    const char* func,
    const int line) {
  if (error == cudaSuccess) {
    return cudaSuccess;
  }
  return FailCuda(error,
                  "CUDA error from " + std::string(expr) + ": code=" +
                      std::to_string(static_cast<int>(error)) + " message=" +
                      cudaGetErrorString(error),
                  file, func, line, FATAL_ON_ERROR);
}

inline CUresult CheckCu(
    const CUresult result,
    const char* expr,
    const char* file,
    const char* func,
    const int line) {
  if (result == CUDA_SUCCESS) {
    return CUDA_SUCCESS;
  }

  const char* error_string = nullptr;
  (void)cuGetErrorString(result, &error_string);
  return FailCu(result,
                "CUresult error from " + std::string(expr) + ": code=" +
                    std::to_string(static_cast<int>(result)) + " message=" +
                    (error_string == nullptr ? "Unknown error" : error_string),
                file, func, line, FATAL_ON_ERROR);
}

inline cudaError_t Check(
    const bool condition,
    const cudaError_t error_code,
    const char* message,
    const char* file,
    const char* func,
    const int line) {
  if (condition) {
    return cudaSuccess;
  }
  return FailCuda(error_code, message, file, func, line);
}

inline CUresult CheckCuCondition(
    const bool condition,
    const CUresult error_code,
    const char* message,
    const char* file,
    const char* func,
    const int line) {
  if (condition) {
    return CUDA_SUCCESS;
  }
  return FailCu(error_code, message, file, func, line);
}

inline cudaError_t ReadBoolEnvVar(
    const char* name,
    const bool default_value,
    bool* out_value) {
  if (name == nullptr) {
    return FailCuda(cudaErrorInvalidValue,
                    "ReadBoolEnvVar: name should not be null", __FILE__,
                    __func__, __LINE__);
  }
  if (out_value == nullptr) {
    return FailCuda(cudaErrorInvalidValue,
                    "ReadBoolEnvVar: out_value should not be null", __FILE__,
                    __func__, __LINE__);
  }

  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    *out_value = default_value;
    return cudaSuccess;
  }

  const std::string value(raw);
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
      value == "YES") {
    *out_value = true;
    return cudaSuccess;
  }

  if (value == "0" || value == "false" || value == "FALSE" || value == "no" ||
      value == "NO") {
    *out_value = false;
    return cudaSuccess;
  }

  return FailCuda(cudaErrorInvalidValue,
                  std::string("Unsupported bool env value: ") + name + "=" +
                      raw,
                  __FILE__, __func__, __LINE__);
}

inline cudaError_t GetCurrentCudaDevice(CUdevice* out_device) {
  if (out_device == nullptr) {
    return FailCuda(cudaErrorInvalidValue,
                    "GetCurrentCudaDevice: out_device should not be null",
                    __FILE__, __func__, __LINE__);
  }

  int device_ordinal = 0;
  const cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess) {
    return CheckCuda(status, "cudaGetDevice(&device_ordinal)", __FILE__,
                     __func__, __LINE__);
  }

  const CUresult cu_status = CheckCu(cuDeviceGet(out_device, device_ordinal),
                                     "cuDeviceGet(out_device, device_ordinal)",
                                     __FILE__, __func__, __LINE__);
  if (cu_status != CUDA_SUCCESS) {
    return ConvertCuResult(cu_status);
  }
  return cudaSuccess;
}

inline bool MatchesTag(const std::string& filter, const std::string& candidate) {
  return filter.empty() || filter == candidate;
}

}  // namespace memsaver::internal::utils

#define RETURN_IF_FALSE(COND, ERROR_CODE, MSG)                                     \
  do {                                                                              \
    const cudaError_t _status =                                                    \
        ::memsaver::internal::utils::Check((COND), (ERROR_CODE), (MSG), __FILE__,  \
                                           __func__, __LINE__);                    \
    if (_status != cudaSuccess) {                                                  \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#define RETURN_IF_CU_FALSE(COND, ERROR_CODE, MSG)                                  \
  do {                                                                              \
    const CUresult _status =                                                       \
        ::memsaver::internal::utils::CheckCuCondition(                             \
            (COND), (ERROR_CODE), (MSG), __FILE__, __func__, __LINE__);            \
    if (_status != CUDA_SUCCESS) {                                                 \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#define RETURN_IF_CUDA_ERROR(EXPR)                                                  \
  do {                                                                               \
    const cudaError_t _status =                                                    \
        ::memsaver::internal::utils::CheckCuda((EXPR), #EXPR, __FILE__, __func__,  \
                                               __LINE__);                          \
    if (_status != cudaSuccess) {                                                  \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#define RETURN_IF_CU_ERROR(EXPR)                                                    \
  do {                                                                               \
    const CUresult _status =                                                       \
        ::memsaver::internal::utils::CheckCu((EXPR), #EXPR, __FILE__, __func__,    \
                                             __LINE__);                            \
    if (_status != CUDA_SUCCESS) {                                                 \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#define RETURN_IF_CU_ERROR_AS_CUDA(EXPR)                                            \
  do {                                                                               \
    const CUresult _status =                                                       \
        ::memsaver::internal::utils::CheckCu((EXPR), #EXPR, __FILE__, __func__,    \
                                             __LINE__);                            \
    if (_status != CUDA_SUCCESS) {                                                 \
      return ::memsaver::internal::utils::ConvertCuResult(_status);                \
    }                                                                              \
  } while (false)

#endif  // INTERNAL_UTILS_H_
