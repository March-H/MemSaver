#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "memsaver/memsaver_c.h"

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024ULL;
constexpr uint64_t kAligned2MiB = 2ULL * kMiB;

void CheckCuda(const cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "[basic_test] " << expr << " failed: "
            << cudaGetErrorString(status) << " (" << static_cast<int>(status)
            << ")" << std::endl;
  std::exit(1);
}

void CheckTrue(const bool value, const char* message) {
  if (value) {
    return;
  }
  std::cerr << "[basic_test] assertion failed: " << message << std::endl;
  std::exit(1);
}

bool MaybeSkipNoGpu() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    std::cout << "[basic_test] skipped (no CUDA device available)" << std::endl;
    return true;
  }
  return false;
}

void SyncCuda() { CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

uint64_t DeviceUsedBytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return static_cast<uint64_t>(total_bytes - free_bytes);
}

void EmptyTorchCache() {
  c10::cuda::CUDACachingAllocator::emptyCache();
  SyncCuda();
}

void WarmUpTorchMatmul() {
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

uint64_t CurrentDeltaBytes(const uint64_t baseline) {
  const uint64_t current = DeviceUsedBytes();
  const int64_t observed_delta =
      static_cast<int64_t>(current) - static_cast<int64_t>(baseline);
  return observed_delta > 0 ? static_cast<uint64_t>(observed_delta) : 0ULL;
}

void ExpectDeltaExact(
    const uint64_t baseline,
    const uint64_t expected_delta,
    const char* label) {
  const uint64_t observed_delta = CurrentDeltaBytes(baseline);
  std::cout << "[basic_test] " << label << " observed delta == " << observed_delta / 1024.0 / 1024.0
            << " MB" << std::endl;
  if (observed_delta == expected_delta) {
    return;
  }
  std::cerr << "[basic_test] " << label << " expected delta == " << expected_delta
            << " but observed " << observed_delta << std::endl;
  std::exit(1);
}

void ExpectReleasedExact(
    const uint64_t before_delta,
    const uint64_t after_delta,
    const uint64_t expected_release,
    const char* label) {
  const uint64_t released = before_delta > after_delta ? before_delta - after_delta : 0ULL;
  std::cout << "[basic_test] " << label << " observed released == " << released / 1024.0 / 1024.0
            << " MB" << std::endl;
  if (released == expected_release) {
    return;
  }
  std::cerr << "[basic_test] " << label << " expected released == "
            << expected_release << " but observed " << released << std::endl;
  std::exit(1);
}

torch::Tensor AllocBytesTensor(const uint64_t bytes) {
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  return torch::empty({static_cast<int64_t>(bytes)}, options);
}

uintptr_t TensorAddress(const torch::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr());
}

using SetInterestingFn = cudaError_t (*)(bool);
using SetTagFn = cudaError_t (*)(const char*);
using SetBackupFn = cudaError_t (*)(bool);
using SetAllocationModeFn = cudaError_t (*)(memsaver_allocation_mode_t);
using PauseResumeFn = cudaError_t (*)(const char*);
using RegionBeginFn = cudaError_t (*)(const char*, bool);
using RegionEndFn = cudaError_t (*)();
using GetMetadataCountByTagFn = cudaError_t (*)(const char*, uint64_t*);
using GetCpuBackupPointerFn =
    cudaError_t (*)(const uint8_t*, uint64_t, uint8_t**);

template <typename Fn>
Fn LoadSymbol(const char* name) {
  void* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol == nullptr) {
    std::cerr << "[basic_test] missing symbol: " << name
              << " (is libmemsaver_preload.so preloaded?)" << std::endl;
    std::exit(1);
  }
  return reinterpret_cast<Fn>(symbol);
}

struct PreloadApi {
  SetInterestingFn set_interesting = nullptr;
  SetTagFn set_tag = nullptr;
  SetBackupFn set_backup = nullptr;
  SetAllocationModeFn set_mode = nullptr;
  PauseResumeFn pause = nullptr;
  PauseResumeFn resume = nullptr;
  RegionBeginFn region_begin = nullptr;
  RegionEndFn region_end = nullptr;
  GetMetadataCountByTagFn get_metadata_count_by_tag = nullptr;
  GetCpuBackupPointerFn get_cpu_backup_pointer = nullptr;
};

PreloadApi LoadPreloadApi() {
  PreloadApi api;
  api.set_interesting = LoadSymbol<SetInterestingFn>(
      "memsaver_preload_set_interesting_region");
  api.set_tag = LoadSymbol<SetTagFn>("memsaver_preload_set_current_tag");
  api.set_backup =
      LoadSymbol<SetBackupFn>("memsaver_preload_set_enable_cpu_backup");
  api.set_mode = LoadSymbol<SetAllocationModeFn>(
      "memsaver_preload_set_allocation_mode");
  api.pause = LoadSymbol<PauseResumeFn>("memsaver_preload_pause");
  api.resume = LoadSymbol<PauseResumeFn>("memsaver_preload_resume");
  api.region_begin =
      LoadSymbol<RegionBeginFn>("memsaver_preload_region_begin");
  api.region_end = LoadSymbol<RegionEndFn>("memsaver_preload_region_end");
  api.get_metadata_count_by_tag = LoadSymbol<GetMetadataCountByTagFn>(
      "memsaver_preload_get_metadata_count_by_tag");
  api.get_cpu_backup_pointer = LoadSymbol<GetCpuBackupPointerFn>(
      "memsaver_preload_get_cpu_backup_pointer");
  return api;
}

void ConfigureRegularRegion(
    const PreloadApi& api,
    const char* tag,
    const bool enable_cpu_backup) {
  CheckCuda(api.set_interesting(true), "set_interesting(true)");
  CheckCuda(api.set_mode(MEMSAVER_ALLOCATION_MODE_NORMAL), "set_mode(normal)");
  CheckCuda(api.set_tag(tag), "set_tag");
  CheckCuda(api.set_backup(enable_cpu_backup), "set_backup");
}

void CheckManagedMetadataExistsForTensor(
    const PreloadApi& api,
    const torch::Tensor& tensor,
    const char* label) {
  uint8_t* cpu_backup = reinterpret_cast<uint8_t*>(0x1);
  const auto status = api.get_cpu_backup_pointer(
      static_cast<const uint8_t*>(tensor.data_ptr()), 16, &cpu_backup);
  CheckCuda(status, label);
  CheckTrue(cpu_backup == nullptr,
            "active managed allocation should expose null cpu backup pointer");
}

void ExpectMetadataCountByTag(
    const PreloadApi& api,
    const char* tag,
    const uint64_t expected_count,
    const char* label) {
  uint64_t observed_count = 0;
  CheckCuda(
      api.get_metadata_count_by_tag(tag, &observed_count),
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

void WarmUpRegularBytes(
    const PreloadApi& api,
    const std::vector<uint64_t>& sizes,
    const bool run_fill,
    const bool run_cpu_copy) {
  CheckCuda(api.set_interesting(false), "set_interesting(false) warmup");
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

void DisturbPhysicalMemory(const PreloadApi& api) {
  CheckCuda(api.set_interesting(false), "set_interesting(false) disturb");
  void* p1 = nullptr;
  void* p2 = nullptr;
  CheckCuda(cudaMalloc(&p1, 1ULL * kMiB), "cudaMalloc disturb p1");
  CheckCuda(cudaMalloc(&p2, 20ULL * kMiB), "cudaMalloc disturb p2");
  CheckCuda(cudaMemset(p1, 0x6B, 1ULL * kMiB), "cudaMemset disturb p1");
  CheckCuda(cudaMemset(p2, 0x4D, 20ULL * kMiB), "cudaMemset disturb p2");
  SyncCuda();
  CheckCuda(cudaFree(p1), "cudaFree disturb p1");
  CheckCuda(cudaFree(p2), "cudaFree disturb p2");
  CheckCuda(api.set_interesting(true), "set_interesting(true) disturb");
}

void TestCase1NaiveAllocations(const PreloadApi& api) {
  {
    WarmUpRegularBytes(
        api, {1ULL * kMiB, 20ULL * kMiB, 100ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/false);
  }
  ConfigureRegularRegion(api, "naive", /*enable_cpu_backup=*/false);
  const uint64_t baseline = DeviceUsedBytes();
  const uint64_t expected_naive_total =
      kAligned2MiB + 20ULL * kMiB + 100ULL * kMiB;

  auto t1 = AllocBytesTensor(1ULL * kMiB);
  auto t2 = AllocBytesTensor(20ULL * kMiB);
  auto t3 = AllocBytesTensor(100ULL * kMiB);
  SyncCuda();

  ExpectDeltaExact(baseline, expected_naive_total, "case1 naive total bytes");
  const uint64_t alloc_delta = CurrentDeltaBytes(baseline);

  CheckManagedMetadataExistsForTensor(
      api, t1, "get_cpu_backup_pointer(case1:t1)");
  CheckManagedMetadataExistsForTensor(
      api, t2, "get_cpu_backup_pointer(case1:t2)");
  CheckManagedMetadataExistsForTensor(
      api, t3, "get_cpu_backup_pointer(case1:t3)");
  ExpectMetadataCountByTag(api, "naive", 3ULL, "case1 metadata count(naive)");

  const uintptr_t addr1 = TensorAddress(t1);
  const uintptr_t addr2 = TensorAddress(t2);
  const uintptr_t addr3 = TensorAddress(t3);
  CheckCuda(api.pause("naive"), "pause(naive) case1");
  SyncCuda();
  const uint64_t paused_delta = CurrentDeltaBytes(baseline);
  ExpectDeltaExact(baseline, 0ULL, "case1 paused delta");
  ExpectReleasedExact(
      alloc_delta,
      paused_delta,
      expected_naive_total,
      "case1 pause(naive) released bytes");
  CheckCuda(api.resume("naive"), "resume(naive) case1");
  SyncCuda();
  ExpectDeltaExact(baseline, expected_naive_total, "case1 resume delta");
  CheckTrue(TensorAddress(t1) == addr1, "case1 t1 address changed");
  CheckTrue(TensorAddress(t2) == addr2, "case1 t2 address changed");
  CheckTrue(TensorAddress(t3) == addr3, "case1 t3 address changed");

  t1.reset();
  t2.reset();
  t3.reset();
  EmptyTorchCache();
}

void TestCase2CpuBackupPreserves(const PreloadApi& api) {
  {
    WarmUpRegularBytes(
        api, {1ULL * kMiB, 20ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }

  ConfigureRegularRegion(api, "naive", /*enable_cpu_backup=*/true);
  const uint64_t baseline = DeviceUsedBytes();

  auto t1 = AllocBytesTensor(1ULL * kMiB);
  auto t2 = AllocBytesTensor(20ULL * kMiB);
  t1.fill_(0x11);
  t2.fill_(0x22);
  SyncCuda();
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case2 allocation delta");
  
  const auto before_t1 = t1.cpu();
  const auto before_t2 = t2.cpu();
  const uintptr_t addr1 = TensorAddress(t1);
  const uintptr_t addr2 = TensorAddress(t2);
  const uint64_t alloc_delta = CurrentDeltaBytes(baseline);

  CheckCuda(api.pause("naive"), "pause(naive) case2");
  SyncCuda();
  const uint64_t paused_delta = CurrentDeltaBytes(baseline);
  ExpectDeltaExact(baseline, 0, "case2 paused delta");
  ExpectReleasedExact(
      alloc_delta,
      paused_delta,
      22ULL * kMiB,
      "case2 pause should release managed bytes");

  CheckCuda(api.resume("naive"), "resume(naive) case2");
  SyncCuda();
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case2 resume delta");

  CheckTrue(TensorAddress(t1) == addr1, "case2 t1 address changed");
  CheckTrue(TensorAddress(t2) == addr2, "case2 t2 address changed");
  CheckTrue(torch::equal(t1.cpu(), before_t1), "case2 t1 data mismatch after resume");
  CheckTrue(torch::equal(t2.cpu(), before_t2), "case2 t2 data mismatch after resume");

  t1.reset();
  t2.reset();
  EmptyTorchCache();
}

void TestCase3NoBackupLosesData(const PreloadApi& api) {
  {
    WarmUpRegularBytes(
        api, {1ULL * kMiB, 20ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }
  ConfigureRegularRegion(api, "naive", /*enable_cpu_backup=*/false);
  const uint64_t baseline = DeviceUsedBytes();

  auto t1 = AllocBytesTensor(1ULL * kMiB);
  auto t2 = AllocBytesTensor(20ULL * kMiB);
  t1.fill_(0xAB);
  t2.fill_(0xCD);
  SyncCuda();

  const auto before_t1 = t1.cpu();
  const auto before_t2 = t2.cpu();
  const uintptr_t addr1 = TensorAddress(t1);
  const uintptr_t addr2 = TensorAddress(t2);
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case3 allocation delta");

  CheckCuda(api.pause("naive"), "pause(naive) case3");
  DisturbPhysicalMemory(api);
  CheckCuda(api.resume("naive"), "resume(naive) case3");
  SyncCuda();

  ExpectDeltaExact(baseline, 22ULL * kMiB, "case3 resume delta");
  CheckTrue(TensorAddress(t1) == addr1, "case3 t1 address changed");
  CheckTrue(TensorAddress(t2) == addr2, "case3 t2 address changed");
  CheckTrue(!torch::equal(t1.cpu(), before_t1), "case3 t1 data unexpectedly preserved");
  CheckTrue(!torch::equal(t2.cpu(), before_t2), "case3 t2 data unexpectedly preserved");

  t1.reset();
  t2.reset();
  EmptyTorchCache();
}

void TestCase4MatmulWithTags(const PreloadApi& api) {
  {
    CheckCuda(api.set_interesting(false), "set_interesting(false) case4 warmup");
    {
      auto warm_fp16_cuda =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
      torch::NoGradGuard warm_no_grad;
      auto warm_a = torch::randn({512, 512}, warm_fp16_cuda);
      auto warm_b = torch::randn({512, 4096}, warm_fp16_cuda);
      auto warm_c = torch::matmul(warm_a, warm_b);
      (void)warm_c;
      SyncCuda();
    }
    EmptyTorchCache();
  }

  auto fp16_cuda =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  const uint64_t baseline = DeviceUsedBytes();

  CheckCuda(api.region_begin("MatA", false), "region_begin(MatA) case4");
  auto a = torch::randn({512, 512}, fp16_cuda);
  SyncCuda();
  CheckCuda(api.region_end(), "region_end(MatA) case4");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "case4 after MatA");

  CheckCuda(api.region_begin("MatB", false), "region_begin(MatB) case4");
  auto b = torch::randn({512, 4096}, fp16_cuda);
  SyncCuda();
  CheckCuda(api.region_end(), "region_end(MatB) case4");
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case4 after MatB");

  CheckCuda(api.set_interesting(false), "set_interesting(false) case4 matmul");
  torch::Tensor c;
  {
    torch::NoGradGuard no_grad;
    c = torch::matmul(a, b);
  }
  SyncCuda();
  CheckCuda(api.set_interesting(true), "set_interesting(true) case4 restore");

  CheckTrue(c.defined(), "case4 result tensor C should be defined");
  CheckTrue(c.sizes() == torch::IntArrayRef({512, 4096}),
            "case4 result tensor C shape mismatch");
  CheckManagedMetadataExistsForTensor(
      api, a, "get_cpu_backup_pointer(case4:MatA)");
  CheckManagedMetadataExistsForTensor(
      api, b, "get_cpu_backup_pointer(case4:MatB)");
  ExpectMetadataCountByTag(api, "MatA", 1ULL, "case4 metadata count(MatA)");
  ExpectMetadataCountByTag(api, "MatB", 1ULL, "case4 metadata count(MatB)");
  constexpr uint64_t kCase4FinalDelta = 42ULL * kMiB;
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 final delta");

  CheckCuda(api.pause(nullptr), "pause(all) case4");
  SyncCuda();
  const uint64_t paused_all_delta = CurrentDeltaBytes(baseline);
  ExpectReleasedExact(
      kCase4FinalDelta,
      paused_all_delta,
      22ULL * kMiB,
      "case4 pause(all) released bytes");
  CheckCuda(api.resume(nullptr), "resume(all) case4");
  SyncCuda();
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 resume(all) delta");

  CheckCuda(api.pause("MatA"), "pause(MatA) case4");
  SyncCuda();
  const uint64_t paused_mat_a_delta = CurrentDeltaBytes(baseline);
  ExpectReleasedExact(
      kCase4FinalDelta,
      paused_mat_a_delta,
      2ULL * kMiB,
      "case4 pause(MatA) released bytes");
  CheckCuda(api.resume("MatA"), "resume(MatA) case4");
  SyncCuda();
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 resume(MatA) delta");

  CheckCuda(api.pause("MatB"), "pause(MatB) case4");
  SyncCuda();
  const uint64_t paused_mat_b_delta = CurrentDeltaBytes(baseline);
  ExpectReleasedExact(
      kCase4FinalDelta,
      paused_mat_b_delta,
      20ULL * kMiB,
      "case4 pause(MatB) released bytes");
  CheckCuda(api.resume("MatB"), "resume(MatB) case4");
  SyncCuda();
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 resume(MatB) delta");

  a.reset();
  b.reset();
  c.reset();
  EmptyTorchCache();
}

struct GemmResult {
  torch::Tensor b;
  torch::Tensor c;
};

GemmResult GemmFunc() {
  auto fp16_cuda =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  torch::NoGradGuard no_grad;

  auto a = torch::randn({512, 512}, fp16_cuda);
  auto b = torch::randn({512, 4096}, fp16_cuda);
  auto c = torch::matmul(a, b);
  return GemmResult{std::move(b), std::move(c)};
}

void CheckGemmResultLayout(const GemmResult& result, const char* label) {
  CheckTrue(result.b.defined(), "gemm B should be defined");
  CheckTrue(result.c.defined(), "gemm C should be defined");
  const uintptr_t b_addr = TensorAddress(result.b);
  const uintptr_t c_addr = TensorAddress(result.c);
  const uintptr_t b_end = b_addr + static_cast<uintptr_t>(result.b.nbytes());
  if (c_addr != b_end) {
    std::cerr << "[basic_test] " << label
              << " expected B/C contiguous layout, got B=" << b_addr
              << " C=" << c_addr << " B_end=" << b_end << std::endl;
    std::exit(1);
  }
}

void TestCase5GemmSameThread(const PreloadApi& api) {
  {
    CheckCuda(api.set_interesting(false), "set_interesting(false) case5 warmup");
    {
      const GemmResult warm_result = GemmFunc();
      (void)warm_result;
      SyncCuda();
    }
    EmptyTorchCache();
  }
  ConfigureRegularRegion(api, "gemm", /*enable_cpu_backup=*/false);
  const uint64_t baseline = DeviceUsedBytes();

  {
    const GemmResult result = GemmFunc();
    SyncCuda();
    ExpectDeltaExact(baseline, 22ULL * kMiB, "case5 gemm delta");
    CheckGemmResultLayout(result, "case5");
    CheckManagedMetadataExistsForTensor(
        api, result.b, "get_cpu_backup_pointer(case5:B)");
    CheckManagedMetadataExistsForTensor(
        api, result.c, "get_cpu_backup_pointer(case5:C)");
  }

  EmptyTorchCache();
}

struct ThreadedGemmOutput {
  cudaError_t setup_status = cudaSuccess;
  std::string setup_error;
  GemmResult result;
};

void TestCase6GemmChildThread(const PreloadApi& api) {
  const char* old_enable = std::getenv("MEMSAVER_ENABLE");
  const bool had_old_enable = old_enable != nullptr;
  const std::string old_enable_value = had_old_enable ? old_enable : "";

  // Warmup in child thread, then clear cache before case6 assertions.
  {
    CheckTrue(unsetenv("MEMSAVER_ENABLE") == 0,
              "unsetenv MEMSAVER_ENABLE case6 warmup");
    CheckCuda(api.set_interesting(false), "set_interesting(false) case6 warmup");
    std::thread warmup_worker([]() {
      const GemmResult warm_result = GemmFunc();
      (void)warm_result;
      SyncCuda();
    });
    warmup_worker.join();
    EmptyTorchCache();
  }

  // Subcase A: no MEMSAVER_ENABLE in child-thread config => unmanaged.
  {
    CheckTrue(unsetenv("MEMSAVER_ENABLE") == 0,
              "unsetenv MEMSAVER_ENABLE case6 subcaseA");
    CheckCuda(api.set_interesting(false), "set_interesting(false) case6 subcaseA");

    ThreadedGemmOutput output;
    std::thread worker([&]() {
      try {
        output.result = GemmFunc();
      } catch (const std::exception& e) {
        output.setup_status = cudaErrorUnknown;
        output.setup_error = e.what();
      } catch (...) {
        output.setup_status = cudaErrorUnknown;
        output.setup_error = "unknown exception";
      }
    });
    worker.join();

    if (output.setup_status != cudaSuccess) {
      std::cerr << "[basic_test] case6 subcaseA worker setup failed at "
                << output.setup_error << ": "
                << cudaGetErrorString(output.setup_status) << std::endl;
      std::exit(1);
    }

    SyncCuda();
    uint8_t* cpu_ptr = nullptr;
    const cudaError_t b_status = api.get_cpu_backup_pointer(
        static_cast<const uint8_t*>(output.result.b.data_ptr()), 16, &cpu_ptr);
    CheckTrue(
        b_status == cudaErrorInvalidValue,
        "case6 subcaseA expected B to be unmanaged (metadata lookup should fail)");
    const cudaError_t c_status = api.get_cpu_backup_pointer(
        static_cast<const uint8_t*>(output.result.c.data_ptr()), 16, &cpu_ptr);
    CheckTrue(
        c_status == cudaErrorInvalidValue,
        "case6 subcaseA expected C to be unmanaged (metadata lookup should fail)");

    output.result.b.reset();
    output.result.c.reset();
    EmptyTorchCache();
  }

  // Subcase B: MEMSAVER_ENABLE=1 for child thread => managed.
  {
    CheckTrue(setenv("MEMSAVER_ENABLE", "1", 1) == 0,
              "setenv MEMSAVER_ENABLE=1 case6");
    const uint64_t baseline = DeviceUsedBytes();

    ThreadedGemmOutput output;
    std::thread worker([&]() {
      try {
        output.result = GemmFunc();
      } catch (const std::exception& e) {
        output.setup_status = cudaErrorUnknown;
        output.setup_error = e.what();
      } catch (...) {
        output.setup_status = cudaErrorUnknown;
        output.setup_error = "unknown exception";
      }
    });
    worker.join();

    if (output.setup_status != cudaSuccess) {
      std::cerr << "[basic_test] case6 subcaseB worker setup failed at "
                << output.setup_error << ": "
                << cudaGetErrorString(output.setup_status) << std::endl;
      std::exit(1);
    }

    SyncCuda();
    ExpectDeltaExact(baseline, 22ULL * kMiB, "case6 subcaseB managed delta");
    CheckGemmResultLayout(output.result, "case6 subcaseB");
    CheckManagedMetadataExistsForTensor(
        api, output.result.b, "get_cpu_backup_pointer(case6:B)");
    CheckManagedMetadataExistsForTensor(
        api, output.result.c, "get_cpu_backup_pointer(case6:C)");

    output.result.b.reset();
    output.result.c.reset();
    EmptyTorchCache();
  }

  if (had_old_enable) {
    CheckTrue(setenv("MEMSAVER_ENABLE", old_enable_value.c_str(), 1) == 0,
              "restore MEMSAVER_ENABLE case6");
  } else {
    CheckTrue(unsetenv("MEMSAVER_ENABLE") == 0,
              "unsetenv MEMSAVER_ENABLE case6 restore");
  }
  CheckCuda(api.set_interesting(false), "set_interesting(false) case6 restore");
}

void TestCase7BasicPauseResumeNoBackup(const PreloadApi& api) {
  {
    WarmUpRegularBytes(
        api, {1ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }
  ConfigureRegularRegion(api, "basic", /*enable_cpu_backup=*/false);
  const uint64_t baseline = DeviceUsedBytes();

  auto t = AllocBytesTensor(1ULL * kMiB);
  t.fill_(0x5A);
  SyncCuda();
  const auto before = t.cpu();
  const uintptr_t addr = TensorAddress(t);
  ExpectDeltaExact(baseline, 2ULL * kMiB, "case7 allocation delta");

  CheckCuda(api.pause("basic"), "pause(basic) case7");
  DisturbPhysicalMemory(api);
  CheckCuda(api.resume("basic"), "resume(basic) case7");
  SyncCuda();

  const auto after_resume = t.cpu();
  CheckTrue(TensorAddress(t) == addr, "case7 address changed");
  CheckTrue(!torch::equal(after_resume, before),
            "case7 resumed data unexpectedly preserved");

  t.fill_(0x11);
  SyncCuda();
  const auto after_write = t.cpu();
  CheckTrue(
      std::all_of(
          after_write.data_ptr<uint8_t>(),
          after_write.data_ptr<uint8_t>() + after_write.numel(),
          [](const uint8_t v) { return v == 0x11; }),
      "case7 write/read check failed after resume");

  ExpectDeltaExact(baseline, 2ULL * kMiB, "case7 resume delta");

  t.reset();
  EmptyTorchCache();
}

void TestCase8TagAndCpuBackupPointer(const PreloadApi& api) {
  {
    WarmUpRegularBytes(
        api, {1ULL * kMiB, 20ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }
  CheckCuda(api.set_interesting(true), "set_interesting(true) case8");
  CheckCuda(api.set_mode(MEMSAVER_ALLOCATION_MODE_NORMAL), "set_mode(normal) case8");

  CheckCuda(api.set_tag("with_backup"), "set_tag(with_backup) case8");
  CheckCuda(api.set_backup(true), "set_backup(true) case8");
  auto with_backup = AllocBytesTensor(1ULL * kMiB);
  with_backup.fill_(0x2A);
  SyncCuda();
  const auto before = with_backup.cpu();

  CheckCuda(api.set_tag("without_backup"), "set_tag(without_backup) case8");
  CheckCuda(api.set_backup(false), "set_backup(false) case8");
  auto without_backup = AllocBytesTensor(20ULL * kMiB);
  without_backup.fill_(0x7C);
  SyncCuda();

  CheckCuda(api.pause("with_backup"), "pause(with_backup) case8");
  uint8_t* backup_ptr = nullptr;
  CheckCuda(
      api.get_cpu_backup_pointer(
          static_cast<const uint8_t*>(with_backup.data_ptr()), 64, &backup_ptr),
      "get_cpu_backup_pointer(with_backup) case8");
  CheckTrue(backup_ptr != nullptr,
            "case8 with_backup should have non-null CPU backup pointer");

  uint8_t* active_ptr = reinterpret_cast<uint8_t*>(0x1);
  CheckCuda(
      api.get_cpu_backup_pointer(
          static_cast<const uint8_t*>(without_backup.data_ptr()),
          64,
          &active_ptr),
      "get_cpu_backup_pointer(without_backup) case8");
  CheckTrue(active_ptr == nullptr,
            "case8 active without_backup should expose null backup pointer");

  CheckCuda(api.resume("with_backup"), "resume(with_backup) case8");
  SyncCuda();
  CheckTrue(torch::equal(with_backup.cpu(), before),
            "case8 with_backup data mismatch after resume");

  with_backup.reset();
  without_backup.reset();
  EmptyTorchCache();
}

}  // namespace

int main() {
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  const PreloadApi api = LoadPreloadApi();
  WarmUpTorchMatmul();

  TestCase1NaiveAllocations(api);
  TestCase2CpuBackupPreserves(api);
  TestCase3NoBackupLosesData(api);
  TestCase4MatmulWithTags(api);
  TestCase5GemmSameThread(api);
  TestCase6GemmChildThread(api);
  TestCase7BasicPauseResumeNoBackup(api);
  TestCase8TagAndCpuBackupPointer(api);

  std::cout << "[basic_test] all tests passed" << std::endl;
  return 0;
}
