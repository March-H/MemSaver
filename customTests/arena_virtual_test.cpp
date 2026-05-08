#include "utils/custom_torch_allocator.h"

#define main memsaver_original_arena_virtual_main
#include "../tests/arena_virtual_test.cpp"
#undef main

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kGiB = 1024ULL * kMiB;
constexpr uint64_t kArenaVirtualType3ReserveBytes = 100ULL * kGiB;
constexpr uint64_t kArenaVirtualType3StepDelta = 1126ULL * kMiB;

void TestCase4ArenaVirtualType3ReserveOnly(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL, 3),
      "enter_region(naive arena_virtual type3) case4");
  torch::Tensor tensor = AllocBytesTensor(kArenaVirtualType3ReserveBytes);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual type3) case4");

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual type3 case4 metadata count");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual type3 case4 allocation delta");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL,
          3),
      "evict_region_pool_from_cache(naive arena_virtual type3) case4");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual type3 case4 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual type3 case4 delta after evict");
}

void TestCase5ArenaVirtualType3ActivateSmall(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL, 3),
      "enter_region(naive arena_virtual type3 init) case5");
  torch::Tensor arena = AllocBytesTensor(kArenaVirtualType3ReserveBytes);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual type3 init) case5");
  arena = torch::Tensor();
  SyncCuda();

  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL, 3),
      "enter_region(naive arena_virtual type3 reuse) case5");
  torch::Tensor tensor = AllocBytesTensor(5ULL * kKiB);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual type3 reuse) case5");

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual type3 case5 metadata count");
  ExpectDeltaExact(baseline, 20ULL * kMiB, "arena_virtual type3 case5 allocation delta");

  tensor = torch::Tensor();
  SyncCuda();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL,
          3),
      "evict_region_pool_from_cache(naive arena_virtual type3) case5");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual type3 case5 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual type3 case5 delta after evict");
}

void TestCase6ArenaVirtualType3StepReuse(
    MemSaver& memsaver,
    const uint64_t baseline) {
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL, 3),
      "enter_region(naive arena_virtual type3 init) case6");
  torch::Tensor arena = AllocBytesTensor(kArenaVirtualType3ReserveBytes);
  SyncCuda();
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual type3 init) case6");
  arena = torch::Tensor();
  SyncCuda();

  uintptr_t small_address = 0;
  uintptr_t middle_address = 0;
  uintptr_t large_address = 0;
  CheckCuda(
      memsaver.enter_region("naive", false, AllocationKind::ARENA_VIRTUAL, 3),
      "enter_region(naive arena_virtual type3 loop) case6");
  for (uint64_t step = 0; step < 256ULL; ++step) {
    torch::Tensor small = AllocBytesTensor(5ULL * kKiB);
    torch::Tensor middle = AllocBytesTensor(100ULL * kMiB);
    torch::Tensor large = AllocBytesTensor(1ULL * kGiB);
    SyncCuda();

    if (step == 0) {
      small_address = TensorAddress(small);
      middle_address = TensorAddress(middle);
      large_address = TensorAddress(large);
    } else {
      CheckTrue(TensorAddress(small) == small_address, "arena_virtual type3 case6 small address should be stable");
      CheckTrue(TensorAddress(middle) == middle_address, "arena_virtual type3 case6 middle address should be stable");
      CheckTrue(TensorAddress(large) == large_address, "arena_virtual type3 case6 large address should be stable");
    }
    CheckTrue(
        CurrentDeltaBytes(baseline) == kArenaVirtualType3StepDelta,
        "arena_virtual type3 case6 allocation delta in step");

    small = torch::Tensor();
    middle = torch::Tensor();
    large = torch::Tensor();
    SyncCuda();
  }
  CheckCuda(memsaver.leave_region(), "leave_region(naive arena_virtual type3 loop) case6");

  ExpectMetadataCountByTag("naive", 1ULL, "arena_virtual type3 case6 metadata count");
  ExpectDeltaExact(baseline, kArenaVirtualType3StepDelta, "arena_virtual type3 case6 final delta");

  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          "naive",
          false,
          AllocationKind::ARENA_VIRTUAL,
          3),
      "evict_region_pool_from_cache(naive arena_virtual type3) case6");
  ExpectMetadataCountByTag(
      "naive",
      0ULL,
      "arena_virtual type3 case6 metadata count after evict");
  ExpectDeltaExact(baseline, 0ULL, "arena_virtual type3 case6 delta after evict");
}

int main() {
  InstallCustomTorchAllocator();
  const int status = memsaver_original_arena_virtual_main();
  if (status != 0) {
    return status;
  }

  SetTestName("arena_virtual_type3_test");
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  MemSaver memsaver;
  WarmUpRegularBytes({1ULL * kMiB}, true, true);
  const uint64_t baseline = DeviceUsedBytes();

  TestCase4ArenaVirtualType3ReserveOnly(memsaver, baseline);
  TestCase5ArenaVirtualType3ActivateSmall(memsaver, baseline);
  TestCase6ArenaVirtualType3StepReuse(memsaver, baseline);

  std::cout << "[" << CurrentTestName() << "] all tests passed" << std::endl;
  return 0;
}
