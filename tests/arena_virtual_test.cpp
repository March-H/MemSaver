#include "utils/test_utils.h"

void TestCase1ArenaVirtualReserveOnly(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL),
      "enter_region(naive arena_virtual) case1");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual) case1");

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual case1 metadata count");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual case1 allocation delta");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL),
      "evict_region_pool_from_cache(naive arena_virtual) case1");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual case1 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual case1 delta after evict");
}

void TestCase2ArenaVirtualActivate2MiB(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL),
      "enter_region(naive arena_virtual) case2");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual) case2");

  const uint64_t offsets[] = {0ULL};
  CheckCuda(
      memsaver_activate_arena_offsets(
          "naive",
          offsets,
          1ULL,
          2ULL * kMiB),
      "activate_arena_offsets(naive arena_virtual) case2");
  tensor.narrow(0, 0, static_cast<int64_t>(2ULL * kMiB)).fill_(0x11);
  SyncCuda();

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual case2 metadata count");
  ExpectDeltaExact(baseline, 2ULL * kMiB, "arena_virtual case2 allocation delta");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL),
      "evict_region_pool_from_cache(naive arena_virtual) case2");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual case2 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual case2 delta after evict");
}

void TestCase3ArenaVirtualActivate8MiB(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL),
      "enter_region(naive arena_virtual) case3");
  torch::Tensor tensor = AllocBytesTensor(20ULL * kMiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual) case3");

  const uint64_t offsets[] = {0ULL};
  CheckCuda(
      memsaver_activate_arena_offsets(
          "naive",
          offsets,
          1ULL,
          8ULL * kMiB),
      "activate_arena_offsets(naive arena_virtual) case3");
  tensor.narrow(0, 0, static_cast<int64_t>(8ULL * kMiB)).fill_(0x22);
  SyncCuda();

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual case3 metadata count");
  ExpectDeltaExact(baseline, 8ULL * kMiB, "arena_virtual case3 allocation delta");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL),
      "evict_region_pool_from_cache(naive arena_virtual) case3");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual case3 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual case3 delta after evict");
}

int main() {
  SetTestName("arena_virtual_test");
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  MemSaver memsaver;
  WarmUpRegularBytes({1ULL * kMiB}, true, true);
  const uint64_t baseline = DeviceUsedBytes();

  TestCase1ArenaVirtualReserveOnly(memsaver, baseline);
  TestCase2ArenaVirtualActivate2MiB(memsaver, baseline);
  TestCase3ArenaVirtualActivate8MiB(memsaver, baseline);

  std::cout << "[" << CurrentTestName() << "] all tests passed" << std::endl;
  return 0;
}
