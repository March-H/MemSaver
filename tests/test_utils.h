#ifndef MEMSAVER_TESTS_CPP_TEST_UTILS_H_
#define MEMSAVER_TESTS_CPP_TEST_UTILS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "memsaver/entrypoint.h"

constexpr uint64_t kMiB = 1024ULL * 1024ULL;
constexpr uint64_t kAligned2MiB = 2ULL * kMiB;

inline void CheckCuda(const cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "[basic_test] " << expr << " failed: "
            << cudaGetErrorString(status) << " (" << static_cast<int>(status)
            << ")" << std::endl;
  std::exit(1);
}

inline void CheckTrue(const bool value, const char* message) {
  if (value) {
    return;
  }
  std::cerr << "[basic_test] assertion failed: " << message << std::endl;
  std::exit(1);
}

inline bool MaybeSkipNoGpu() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    std::cout << "[basic_test] skipped (no CUDA device available)" << std::endl;
    return true;
  }
  return false;
}

inline void SyncCuda() {
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

inline uint64_t DeviceUsedBytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return static_cast<uint64_t>(total_bytes - free_bytes);
}

inline void EmptyTorchCache() {
  CheckCuda(memsaver_empty_cache(), "memsaver_empty_cache");
  SyncCuda();
}

inline void EvictPoolFromCache(
    MemSaver& memsaver,
    const char* tag,
    const bool enable_cpu_backup,
    const AllocationKind mode = AllocationKind::REGULAR) {
  CheckCuda(
      memsaver.evict_region_pool_from_cache(tag, enable_cpu_backup, mode),
      "evict_region_pool_from_cache");
  SyncCuda();
}

inline void WarmUpTorchMatmul() {
  auto fp16_cuda =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  {
    torch::NoGradGuard no_grad;
    auto a = torch::randn({32, 32}, fp16_cuda);
    auto b = torch::randn({32, 32}, fp16_cuda);
    auto c = torch::matmul(a, b);
    (void)c;
  }
  SyncCuda();
  EmptyTorchCache();
}

inline uint64_t CurrentDeltaBytes(const uint64_t baseline) {
  const uint64_t current = DeviceUsedBytes();
  const int64_t observed_delta =
      static_cast<int64_t>(current) - static_cast<int64_t>(baseline);
  return observed_delta > 0 ? static_cast<uint64_t>(observed_delta) : 0ULL;
}

inline void ExpectDeltaExact(
    const uint64_t baseline,
    const uint64_t expected_delta,
    const char* label) {
  const uint64_t observed_delta = CurrentDeltaBytes(baseline);
  std::cout << "[basic_test] " << label << " observed delta == "
            << observed_delta / 1024.0 / 1024.0 << " MB" << std::endl;
  if (observed_delta == expected_delta) {
    return;
  }
  std::cerr << "[basic_test] " << label << " expected delta == "
            << expected_delta << " but observed " << observed_delta
            << std::endl;
  std::exit(1);
}

inline void ExpectReleasedExact(
    const uint64_t before_delta,
    const uint64_t after_delta,
    const uint64_t expected_release,
    const char* label) {
  const uint64_t released =
      before_delta > after_delta ? before_delta - after_delta : 0ULL;
  std::cout << "[basic_test] " << label << " observed released == "
            << released / 1024.0 / 1024.0 << " MB" << std::endl;
  if (released == expected_release) {
    return;
  }
  std::cerr << "[basic_test] " << label << " expected released == "
            << expected_release << " but observed " << released << std::endl;
  std::exit(1);
}

inline torch::Tensor AllocBytesTensor(const uint64_t bytes) {
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  return torch::empty({static_cast<int64_t>(bytes)}, options);
}

inline uintptr_t TensorAddress(const torch::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr());
}

inline void CheckManagedMetadataExistsForTensor(
    const torch::Tensor& tensor,
    const char* label) {
  uint8_t* cpu_backup = reinterpret_cast<uint8_t*>(0x1);
  const auto status = memsaver_get_cpu_backup_pointer(
      static_cast<const uint8_t*>(tensor.data_ptr()), 16, &cpu_backup);
  CheckCuda(status, label);
  CheckTrue(cpu_backup == nullptr,
            "active managed allocation should expose null cpu backup pointer");
}

inline void ExpectMetadataCountByTag(
    const char* tag,
    const uint64_t expected_count,
    const char* label) {
  uint64_t observed_count = 0;
  CheckCuda(
      memsaver_get_metadata_count_by_tag(tag, &observed_count),
      "get_metadata_count_by_tag");
  std::cout << "[basic_test] " << label << " observed count == "
            << observed_count << std::endl;
  if (observed_count == expected_count) {
    return;
  }
  std::cerr << "[basic_test] " << label << " expected count == "
            << expected_count << " but observed " << observed_count
            << std::endl;
  std::exit(1);
}

inline void WarmUpRegularBytes(
    const std::vector<uint64_t>& sizes,
    const bool run_fill,
    const bool run_cpu_copy) {
  {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(sizes.size());
    for (const uint64_t size : sizes) {
      tensors.push_back(AllocBytesTensor(size));
    }
    if (run_fill) {
      uint8_t fill_value = 0x11;
      for (auto& tensor : tensors) {
        tensor.fill_(fill_value);
        fill_value = static_cast<uint8_t>(fill_value + 0x11);
      }
    }
    SyncCuda();
    if (run_cpu_copy) {
      for (const auto& tensor : tensors) {
        const auto host_copy = tensor.cpu();
        (void)host_copy;
      }
    }
  }
  EmptyTorchCache();
}

inline void DisturbPhysicalMemory() {
  void* p1 = nullptr;
  void* p2 = nullptr;
  CheckCuda(cudaMalloc(&p1, 1ULL * kMiB), "cudaMalloc disturb p1");
  CheckCuda(cudaMalloc(&p2, 20ULL * kMiB), "cudaMalloc disturb p2");
  CheckCuda(cudaMemset(p1, 0x6B, 1ULL * kMiB), "cudaMemset disturb p1");
  CheckCuda(cudaMemset(p2, 0x4D, 20ULL * kMiB), "cudaMemset disturb p2");
  SyncCuda();
  CheckCuda(cudaFree(p1), "cudaFree disturb p1");
  CheckCuda(cudaFree(p2), "cudaFree disturb p2");
}

struct GemmResult {
  torch::Tensor a;
  torch::Tensor b;
  torch::Tensor c;
};

inline GemmResult GemmFunc() {
  auto fp16_cuda =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  torch::NoGradGuard no_grad;

  auto a = torch::randn({512, 512}, fp16_cuda);
  auto b = torch::randn({512, 4096}, fp16_cuda);
  auto c = torch::matmul(a, b);
  return GemmResult{std::move(a), std::move(b), std::move(c)};
}

inline void CheckMetadataLookupFailsForTensor(
    const torch::Tensor& tensor,
    const char* label) {
  uint8_t* cpu_backup = nullptr;
  const auto status = memsaver_get_cpu_backup_pointer(
      static_cast<const uint8_t*>(tensor.data_ptr()), 16, &cpu_backup);
  CheckTrue(status == cudaErrorInvalidValue, label);
}

struct ThreadedGemmOutput {
  cudaError_t setup_status = cudaSuccess;
  std::string setup_error;
  GemmResult result;
};

#endif
