#include "test_utils.h"

void TestCase1ArenaSharedBacking(MemSaver& memsaver, const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA),
      "enter_region(naive arena) case1");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena) case1");

  tensor.narrow(0, 0, static_cast<int64_t>(2ULL * kMiB)).fill_(0x11);
  SyncCuda();

  ExpectMetadataCountByTag("naive", 1ULL, "arena case1 metadata count");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena case1 allocation delta");
  CheckTrue(
      tensor.cpu().eq(0x11).all().item<bool>(),
      "arena case1 expected full 20MiB to read back as 0x11");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA),
      "evict_region_pool_from_cache(naive arena) case1");
  ExpectMetadataCountByTag("naive", 0ULL, "arena case1 metadata count after evict");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena case1 delta after evict");
}

void TestCase2ArenaActivateOffsets(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA),
      "enter_region(naive arena) case2");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena) case2");

  tensor.narrow(0, 0, static_cast<int64_t>(2ULL * kMiB)).fill_(0x11);
  SyncCuda();

  const uint64_t offsets[] = {2ULL * kMiB};
  CheckCuda(
      memsaver_activate_arena_offsets(
          "naive",
          offsets,
          1ULL,
          4ULL * kMiB),
      "activate_arena_offsets(naive) case2");
  tensor.narrow(
            0,
            static_cast<int64_t>(2ULL * kMiB),
            static_cast<int64_t>(4ULL * kMiB))
      .fill_(0x22);
  SyncCuda();

  ExpectMetadataCountByTag("naive", 1ULL, "arena case2 metadata count");
  ExpectDeltaExact(baseline, 6ULL * kMiB, "arena case2 allocation delta");
  const auto host = tensor.cpu();
  CheckTrue(
      host.narrow(0, 0, static_cast<int64_t>(2ULL * kMiB))
          .eq(0x11)
          .all()
          .item<bool>(),
      "arena case2 expected [0,2MiB) to remain 0x11");
  CheckTrue(
      host.narrow(
              0,
              static_cast<int64_t>(2ULL * kMiB),
              static_cast<int64_t>(4ULL * kMiB))
          .eq(0x22)
          .all()
          .item<bool>(),
      "arena case2 expected [2MiB,6MiB) to become 0x22");
  CheckTrue(
      host.narrow(
              0,
              static_cast<int64_t>(6ULL * kMiB),
              static_cast<int64_t>(14ULL * kMiB))
          .eq(0x11)
          .all()
          .item<bool>(),
      "arena case2 expected [6MiB,20MiB) to remain 0x11");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA),
      "evict_region_pool_from_cache(naive arena) case2");
  ExpectMetadataCountByTag("naive", 0ULL, "arena case2 metadata count after evict");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena case2 delta after evict");
}

void TestCase3ArenaDeactivateOffsets(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA),
      "enter_region(naive arena) case3");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena) case3");

  tensor.narrow(0, 0, static_cast<int64_t>(2ULL * kMiB)).fill_(0x11);
  SyncCuda();

  const uint64_t offsets[] = {2ULL * kMiB};
  CheckCuda(
      memsaver_activate_arena_offsets(
          "naive",
          offsets,
          1ULL,
          4ULL * kMiB),
      "activate_arena_offsets(naive) case3");
  tensor.narrow(
            0,
            static_cast<int64_t>(2ULL * kMiB),
            static_cast<int64_t>(4ULL * kMiB))
      .fill_(0x22);
  SyncCuda();

  CheckCuda(
      memsaver_deactivate_arena_offsets(
          "naive",
          offsets,
          1ULL,
          4ULL * kMiB),
      "deactivate_arena_offsets(naive) case3");
  SyncCuda();

  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena case3 delta after deactivate");
  CheckTrue(
      tensor.cpu().eq(0x11).all().item<bool>(),
      "arena case3 expected full 20MiB to revert to 0x11 after deactivate");

  tensor.narrow(
            0,
            static_cast<int64_t>(2ULL * kMiB),
            static_cast<int64_t>(4ULL * kMiB))
      .fill_(0x33);
  SyncCuda();

  ExpectDeltaExact(
      baseline,
      2ULL * kMiB,
      "arena case3 delta after writing deactivated range");
  CheckTrue(
      tensor.cpu().eq(0x33).all().item<bool>(),
      "arena case3 expected full 20MiB to become 0x33");
  ExpectMetadataCountByTag("naive", 1ULL, "arena case3 metadata count");

  tensor = torch::Tensor();
  SyncCuda();
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA),
      "evict_region_pool_from_cache(naive arena) case3");
  ExpectMetadataCountByTag("naive", 0ULL, "arena case3 metadata count after evict");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena case3 delta after evict");
}

int main() {
  SetTestName("arena_test");
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  MemSaver memsaver;
  WarmUpRegularBytes({1ULL * kMiB}, true, true);
  const uint64_t baseline = DeviceUsedBytes();

  TestCase1ArenaSharedBacking(memsaver, baseline);
  TestCase2ArenaActivateOffsets(memsaver, baseline);
  TestCase3ArenaDeactivateOffsets(memsaver, baseline);

  std::cout << "[" << CurrentTestName() << "] all tests passed" << std::endl;
  return 0;
}
