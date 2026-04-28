#include <thread>

#include "test_utils.h"

void TestCase1NaiveAllocations(MemSaver& memsaver) {
  {
    WarmUpRegularBytes(
        {1ULL * kMiB, 20ULL * kMiB, 100ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/false);
  }
  const uint64_t baseline = DeviceUsedBytes();
  const uint64_t expected_naive_total =
      kAligned2MiB + 20ULL * kMiB + 100ULL * kMiB;

  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::REGULAR),
      "enter_region(naive) case1");
  torch::Tensor t1 = AllocBytesTensor(1ULL * kMiB);
  torch::Tensor t2 = AllocBytesTensor(20ULL * kMiB);
  torch::Tensor t3 = AllocBytesTensor(100ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive) case1");

  ExpectDeltaExact(baseline, expected_naive_total, "case1 naive total bytes");

  CheckManagedMetadataExistsForTensor(
      t1, "get_cpu_backup_pointer(case1:t1)");
  CheckManagedMetadataExistsForTensor(
      t2, "get_cpu_backup_pointer(case1:t2)");
  CheckManagedMetadataExistsForTensor(
      t3, "get_cpu_backup_pointer(case1:t3)");
  ExpectMetadataCountByTag("naive", 3ULL, "case1 metadata count(naive)");

  t1 = torch::Tensor();
  t2 = torch::Tensor();
  t3 = torch::Tensor();
  SyncCuda();

  // 将pool从cache里驱逐后，其将不会有额外引用，将会自动触发析构
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(naive) case1");
  // 析构后，对应tag的metadata应该为0
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "case1 metadata count(naive) after evict");
  ExpectDeltaExact(baseline, 0, "case1 naive total bytes");
}

void TestCase2CpuBackupPreserves(MemSaver& memsaver) {
  {
    WarmUpRegularBytes(
        {1ULL * kMiB, 20ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }
  const uint64_t baseline = DeviceUsedBytes();

  CheckCuda(
      memsaver.enter_region("naive_backup", true, AllocationKind::REGULAR),
      "enter_region(naive_backup) case2");
  torch::Tensor t1 = AllocBytesTensor(1ULL * kMiB);
  torch::Tensor t2 = AllocBytesTensor(20ULL * kMiB);
  t1.fill_(0x11);
  t2.fill_(0x22);
  SyncCuda();
  CheckCuda(
      memsaver.leave_region(),
      "leave_region(naive_backup) case2");
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case2 allocation delta");
  
  const auto before_t1 = t1.cpu();
  const auto before_t2 = t2.cpu();
  const uintptr_t addr1 = TensorAddress(t1);
  const uintptr_t addr2 = TensorAddress(t2);
  const uint64_t alloc_delta = CurrentDeltaBytes(baseline);

  CheckCuda(
      memsaver_pause("naive_backup"), "pause(naive_backup) case2");
  SyncCuda();
  const uint64_t paused_delta = CurrentDeltaBytes(baseline);
  ExpectDeltaExact(baseline, 0, "case2 paused delta");
  ExpectReleasedExact(
      alloc_delta,
      paused_delta,
      22ULL * kMiB,
      "case2 pause should release managed bytes");

  CheckCuda(
      memsaver_resume("naive_backup"), "resume(naive_backup) case2");
  SyncCuda();
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case2 resume delta");

  CheckTrue(TensorAddress(t1) == addr1, "case2 t1 address changed");
  CheckTrue(TensorAddress(t2) == addr2, "case2 t2 address changed");
  CheckTrue(torch::equal(t1.cpu(), before_t1), "case2 t1 data mismatch after resume");
  CheckTrue(torch::equal(t2.cpu(), before_t2), "case2 t2 data mismatch after resume");

  t1 = torch::Tensor();
  t2 = torch::Tensor();
  SyncCuda();

  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive_backup",
          true,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(naive_backup) case2");
}

void TestCase3NoBackupLosesData(MemSaver& memsaver) {
  {
    WarmUpRegularBytes(
        {1ULL * kMiB, 20ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/true);
  }
  const uint64_t baseline = DeviceUsedBytes();

  CheckCuda(
      memsaver.enter_region(
          "naive_no_backup",
          false,
          AllocationKind::REGULAR),
      "enter_region(naive_no_backup) case3");
  torch::Tensor t1 = AllocBytesTensor(1ULL * kMiB);
  torch::Tensor t2 = AllocBytesTensor(20ULL * kMiB);
  t1.fill_(0xAB);
  t2.fill_(0xCD);
  SyncCuda();
  CheckCuda(
      memsaver.leave_region(),
      "leave_region(naive_no_backup) case3");

  const auto before_t1 = t1.cpu();
  const auto before_t2 = t2.cpu();
  const uintptr_t addr1 = TensorAddress(t1);
  const uintptr_t addr2 = TensorAddress(t2);
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case3 allocation delta");

  CheckCuda(
      memsaver_pause("naive_no_backup"), "pause(naive_no_backup) case3");
  DisturbPhysicalMemory();
  CheckCuda(
      memsaver_resume("naive_no_backup"), "resume(naive_no_backup) case3");
  SyncCuda();

  ExpectDeltaExact(baseline, 22ULL * kMiB, "case3 resume delta");
  CheckTrue(TensorAddress(t1) == addr1, "case3 t1 address changed");
  CheckTrue(TensorAddress(t2) == addr2, "case3 t2 address changed");
  CheckTrue(!torch::equal(t1.cpu(), before_t1), "case3 t1 data unexpectedly preserved");
  CheckTrue(!torch::equal(t2.cpu(), before_t2), "case3 t2 data unexpectedly preserved");

  t1 = torch::Tensor();
  t2 = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive_no_backup",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(naive_no_backup) case3");
}

void TestCase4MatmulWithTags(MemSaver& memsaver) {
  {
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

  CheckCuda(
      memsaver.enter_region("MatA", false, AllocationKind::REGULAR),
      "enter_region(MatA) case4");
  torch::Tensor a = torch::randn({512, 512}, fp16_cuda);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(MatA) case4");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "case4 after MatA");

  CheckCuda(
      memsaver.enter_region("MatB", false, AllocationKind::REGULAR),
      "enter_region(MatB) case4");
  torch::Tensor b = torch::randn({512, 4096}, fp16_cuda);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(MatB) case4");
  ExpectDeltaExact(baseline, 22ULL * kMiB, "case4 after MatB");

  torch::NoGradGuard no_grad;
  torch::Tensor c = torch::matmul(a, b);
  SyncCuda();

  CheckTrue(c.defined(), "case4 result tensor C should be defined");
  CheckTrue(c.sizes() == torch::IntArrayRef({512, 4096}),
            "case4 result tensor C shape mismatch");
  CheckManagedMetadataExistsForTensor(
      a, "get_cpu_backup_pointer(case4:MatA)");
  CheckManagedMetadataExistsForTensor(
      b, "get_cpu_backup_pointer(case4:MatB)");
  ExpectMetadataCountByTag("MatA", 1ULL, "case4 metadata count(MatA)");
  ExpectMetadataCountByTag("MatB", 1ULL, "case4 metadata count(MatB)");
  constexpr uint64_t kCase4FinalDelta = 42ULL * kMiB;
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 final delta");

  CheckCuda(memsaver_pause(nullptr), "pause(all) case4");
  SyncCuda();
  const uint64_t paused_all_delta = CurrentDeltaBytes(baseline);
  ExpectReleasedExact(
      kCase4FinalDelta,
      paused_all_delta,
      22ULL * kMiB,
      "case4 pause(all) released bytes");
  CheckCuda(memsaver_resume(nullptr), "resume(all) case4");
  SyncCuda();
  ExpectDeltaExact(baseline, kCase4FinalDelta, "case4 resume(all) delta");

  a = torch::Tensor();
  b = torch::Tensor();
  c = torch::Tensor();
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "MatA",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(MatA) case4");
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "MatB",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(MatB) case4");
}

void TestCase5GemmSameThread(MemSaver& memsaver) {
  {
    {
      const GemmResult warm_result = GemmFunc();
      (void)warm_result;
      SyncCuda();
    }
    EmptyTorchCache();
  }
  const uint64_t baseline = DeviceUsedBytes();

  // 非region创建的tensor，应该不属于任何一个pool
  GemmResult result;
  {
    CheckCuda(
        memsaver.enter_region("gemm", false, AllocationKind::REGULAR),
        "enter_region(gemm) case5");
    result = GemmFunc();
    SyncCuda();
    CheckCuda(memsaver.leave_region(), "leave_region(gemm) case5");
  }

  ExpectDeltaExact(baseline, 22ULL * kMiB, "case5 gemm delta");
  CheckTrue(result.a.defined(), "case5 A should be defined");
  CheckTrue(result.b.defined(), "case5 B should be defined");
  CheckTrue(result.c.defined(), "case5 C should be defined");
  {
    const uintptr_t a_addr = TensorAddress(result.a);
    const uintptr_t b_addr = TensorAddress(result.b);
    const uintptr_t c_addr = TensorAddress(result.c);
    const uintptr_t a_end =
        a_addr + static_cast<uintptr_t>(result.a.nbytes());
    const uintptr_t b_end =
        b_addr + static_cast<uintptr_t>(result.b.nbytes());
    CheckTrue(
        b_addr != a_end,
        "case5 expected A and BC to be non-contiguous");
    CheckTrue(
        c_addr == b_end,
        "case5 expected B and C to be contiguous");
  }
  CheckManagedMetadataExistsForTensor(
      result.a, "get_cpu_backup_pointer(case5:A)");
  CheckManagedMetadataExistsForTensor(
      result.b, "get_cpu_backup_pointer(case5:B)");
  CheckManagedMetadataExistsForTensor(
      result.c, "get_cpu_backup_pointer(case5:C)");
  ExpectMetadataCountByTag("gemm", 2ULL, "case5 metadata count(gemm)");

  result.a = torch::Tensor();
  result.b = torch::Tensor();
  result.c = torch::Tensor();
  SyncCuda();
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "gemm",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(gemm) case5");
  ExpectMetadataCountByTag("gemm", 0ULL, "case5 metadata count(gemm) after evict");
  ExpectDeltaExact(baseline, 0, "case5 gemm delta after evict");
}

void TestCase6GemmChildThread(MemSaver& memsaver) {
  {
    std::thread warmup_worker([]() {
      const GemmResult warm_result = GemmFunc();
      (void)warm_result;
      SyncCuda();
    });
    warmup_worker.join();
    EmptyTorchCache();
  }

  {
    const uint64_t baseline = DeviceUsedBytes();
    GemmResult result;
    std::thread worker([&]() {
      result = GemmFunc();
    });
    worker.join();

    SyncCuda();
    CheckMetadataLookupFailsForTensor(
        result.a,
        "case6 subcaseA expected A to be unmanaged");
    CheckMetadataLookupFailsForTensor(
        result.b,
        "case6 subcaseA expected B to be unmanaged");
    CheckMetadataLookupFailsForTensor(
        result.c,
        "case6 subcaseA expected C to be unmanaged");

    result.a = torch::Tensor();
    result.b = torch::Tensor();
    result.c = torch::Tensor();
    SyncCuda();
    EmptyTorchCache();
    ExpectDeltaExact(baseline, 0, "case6 subcaseA unmanaged delta after release");
  }

  {
    const uint64_t baseline = DeviceUsedBytes();

    ThreadedGemmOutput output;
    std::thread worker([&]() {
      output.setup_status =
          memsaver.enter_region("gemm_thread", false, AllocationKind::REGULAR);
      if (output.setup_status != cudaSuccess) {
        output.setup_error = "enter_region(gemm_thread)";
        return;
      }
      output.result = GemmFunc();
      output.setup_status = memsaver.leave_region();
      if (output.setup_status != cudaSuccess) {
        output.setup_error = "leave_region(gemm_thread)";
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
    CheckTrue(output.result.a.defined(), "case6 subcaseB A should be defined");
    CheckTrue(output.result.b.defined(), "case6 subcaseB B should be defined");
    CheckTrue(output.result.c.defined(), "case6 subcaseB C should be defined");
    {
      const uintptr_t a_addr = TensorAddress(output.result.a);
      const uintptr_t b_addr = TensorAddress(output.result.b);
      const uintptr_t c_addr = TensorAddress(output.result.c);
      const uintptr_t a_end =
          a_addr + static_cast<uintptr_t>(output.result.a.nbytes());
      const uintptr_t b_end =
          b_addr + static_cast<uintptr_t>(output.result.b.nbytes());
      CheckTrue(
          b_addr != a_end,
          "case6 subcaseB expected A and BC to be non-contiguous");
      CheckTrue(
          c_addr == b_end,
          "case6 subcaseB expected B and C to be contiguous");
    }
    CheckManagedMetadataExistsForTensor(
        output.result.a, "get_cpu_backup_pointer(case6:A)");
    CheckManagedMetadataExistsForTensor(
        output.result.b, "get_cpu_backup_pointer(case6:B)");
    CheckManagedMetadataExistsForTensor(
        output.result.c, "get_cpu_backup_pointer(case6:C)");
    ExpectMetadataCountByTag(
        "gemm_thread", 2ULL, "case6 subcaseB metadata count(gemm_thread)");

    output.result.a = torch::Tensor();
    output.result.b = torch::Tensor();
    output.result.c = torch::Tensor();
    SyncCuda();
    EmptyTorchCache();
    CheckCuda(
        memsaver.evict_region_pool_from_cache(
            "gemm_thread",
            false,
            AllocationKind::REGULAR),
        "evict_region_pool_from_cache(gemm_thread) case6");
    ExpectMetadataCountByTag(
        "gemm_thread", 0ULL, "case6 subcaseB metadata count(gemm_thread) after evict");
    ExpectDeltaExact(baseline, 0, "case6 subcaseB managed delta after evict");
  }
}

void TestCase7RegionReentryReusesAddress(MemSaver& memsaver) {
  {
    WarmUpRegularBytes(
        {4ULL * kMiB}, /*run_fill=*/true, /*run_cpu_copy=*/false);
  }

  const uint64_t baseline = DeviceUsedBytes();

  torch::Tensor a;
  {
    CheckCuda(
        memsaver.enter_region("naive", false, AllocationKind::REGULAR),
        "enter_region(naive) case7 first");
    a = AllocBytesTensor(4ULL * kMiB);
    SyncCuda();
    CheckCuda(memsaver.leave_region(), "leave_region(naive) case7 first");
  }
  const uintptr_t addr_a = TensorAddress(a);

  a = torch::Tensor();
  SyncCuda();

  torch::Tensor b;
  {
    CheckCuda(
        memsaver.enter_region("naive", false, AllocationKind::REGULAR),
        "enter_region(naive) case7 second");
    b = AllocBytesTensor(4ULL * kMiB);
    SyncCuda();
    CheckCuda(memsaver.leave_region(), "leave_region(naive) case7 second");
  }
  const uintptr_t addr_b = TensorAddress(b);

  CheckTrue(
      addr_b == addr_a,
      "case7 expected B address to reuse A address");

  b = torch::Tensor();
  SyncCuda();
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::REGULAR),
      "evict_region_pool_from_cache(naive) case7");
  ExpectDeltaExact(baseline, 0, "case7 final delta");
}

int main() {
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  MemSaver memsaver;
  WarmUpTorchMatmul();

  TestCase1NaiveAllocations(memsaver);
  TestCase2CpuBackupPreserves(memsaver);
  TestCase3NoBackupLosesData(memsaver);
  TestCase4MatmulWithTags(memsaver);
  TestCase5GemmSameThread(memsaver);
  TestCase6GemmChildThread(memsaver);
  TestCase7RegionReentryReusesAddress(memsaver);

  std::cout << "[basic_test] all tests passed" << std::endl;
  return 0;
}
