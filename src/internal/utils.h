#ifndef INTERNAL_UTILS_H_
#define INTERNAL_UTILS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
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

[[noreturn]] inline void AbortWithLog(
    const std::string& message,
    const char* file,
    const char* func,
    const int line) {
  std::cerr << "[memsaver] " << message << " file=" << file << " func=" << func
            << " line=" << line << std::endl;
  std::abort();
}

inline void ReportError(
    const std::string& message,
    const char* file,
    const char* func,
    const int line) {
  std::cerr << "[memsaver] " << message << " file=" << file << " func=" << func
            << " line=" << line << std::endl;
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

  std::ostringstream oss;
  oss << "CUDA error from " << expr << ": code=" << static_cast<int>(error)
      << " message=" << cudaGetErrorString(error);

#if FATAL_ON_ERROR
  AbortWithLog(oss.str(), file, func, line);
#else
  ReportError(oss.str(), file, func, line);
#endif
  return error;
}

inline cudaError_t CheckCu(
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

#if FATAL_ON_ERROR
  AbortWithLog(oss.str(), file, func, line);
#else
  ReportError(oss.str(), file, func, line);
#endif
  return ConvertCuResult(result);
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

  ReportError(message, file, func, line);
  return error_code;
}

inline cudaError_t ParseBool(const std::string& value, bool* out_value) {
  if (out_value == nullptr) {
    return Check(false, cudaErrorInvalidValue,
                 "ParseBool: out_value should not be null", __FILE__, __func__,
                 __LINE__);
  }

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

  return cudaErrorInvalidValue;
}

inline cudaError_t ReadBoolEnvVar(
    const char* name,
    const bool default_value,
    bool* out_value) {
  if (name == nullptr) {
    return Check(false, cudaErrorInvalidValue,
                 "ReadBoolEnvVar: name should not be null", __FILE__, __func__,
                 __LINE__);
  }
  if (out_value == nullptr) {
    return Check(false, cudaErrorInvalidValue,
                 "ReadBoolEnvVar: out_value should not be null", __FILE__,
                 __func__, __LINE__);
  }

  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    *out_value = default_value;
    return cudaSuccess;
  }

  const cudaError_t status = ParseBool(raw, out_value);
  if (status != cudaSuccess) {
    const std::string message = std::string("Unsupported bool env value: ") + name +
                                "=" + raw;
    return Check(false, cudaErrorInvalidValue, message.c_str(), __FILE__,
                 __func__, __LINE__);
  }

  return cudaSuccess;
}

inline cudaError_t GetCurrentCudaDevice(CUdevice* out_device) {
  if (out_device == nullptr) {
    return Check(false, cudaErrorInvalidValue,
                 "GetCurrentCudaDevice: out_device should not be null", __FILE__,
                 __func__, __LINE__);
  }

  int device_ordinal = 0;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess) {
    return CheckCuda(status, "cudaGetDevice(&device_ordinal)", __FILE__,
                     __func__, __LINE__);
  }

  const CUresult cu_status = cuDeviceGet(out_device, device_ordinal);
  if (cu_status != CUDA_SUCCESS) {
    return CheckCu(cu_status, "cuDeviceGet(out_device, device_ordinal)", __FILE__,
                   __func__, __LINE__);
  }
  return cudaSuccess;
}

inline cudaError_t GetCudaDeviceByOrdinal(const int device_ordinal, CUdevice* out_device) {
  if (out_device == nullptr) {
    return Check(false, cudaErrorInvalidValue,
                 "GetCudaDeviceByOrdinal: out_device should not be null", __FILE__,
                 __func__, __LINE__);
  }
  const CUresult status = cuDeviceGet(out_device, device_ordinal);
  if (status != CUDA_SUCCESS) {
    return CheckCu(status, "cuDeviceGet(out_device, device_ordinal)", __FILE__,
                   __func__, __LINE__);
  }
  return cudaSuccess;
}

inline cudaError_t ConvertCuResultNoAbort(const CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : ConvertCuResult(result);
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
    const cudaError_t _status =                                                    \
        ::memsaver::internal::utils::CheckCu((EXPR), #EXPR, __FILE__, __func__,    \
                                             __LINE__);                            \
    if (_status != cudaSuccess) {                                                  \
      return _status;                                                              \
    }                                                                              \
  } while (false)

#endif  // INTERNAL_UTILS_H_
