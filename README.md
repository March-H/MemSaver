# MemSaver

MemSaver is a C++/CUDA library that integrates with PyTorch's CUDA caching allocator at the MemPool and segment layer. It lets you route allocations in a tagged region into a dedicated pool, pause and resume managed GPU memory while keeping virtual addresses stable, and switch between regular and arena-style allocation behavior.

This repository exposes the public header [`include/memsaver/entrypoint.h`](./include/memsaver/entrypoint.h) and the `memsaver` library target.

## Current Scope

- Region-scoped allocation control through `MemSaver`
- Tagged pool reuse keyed by `(tag, enable_cpu_backup, allocation_mode)`
- Pause and resume for managed allocations
- Optional CPU backup for regular allocations
- Arena mode with explicit offset activation and deactivation
- Torch-based runtime tests for both regular and arena behavior

## Requirements

- Linux x86_64
- CUDA toolkit
- CMake 3.20 or newer
- C++17
- PyTorch or LibTorch available to CMake

`CMakeLists.txt` first tries `find_package(Torch)` and then falls back to `python -c "import torch; print(torch.utils.cmake_prefix_path)"` to locate the Torch CMake package. In practice, the current source tree expects Torch headers and libraries to be available when building `memsaver`.

## Build

Recommended:

```bash
./build.sh
```

Equivalent CMake flow:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build output:

- `memsaver`

Optional test binaries are built when Torch is found:

- `memsaver_torch_basic_test`
- `memsaver_torch_arena_test`

You can also build specific targets:

```bash
./build.sh --target memsaver
./build.sh --target memsaver_torch_basic_test
./build.sh --target memsaver_torch_arena_test
```

## Install And Consume

```bash
cmake --install build --prefix /your/install/prefix
```

From another CMake project:

```cmake
find_package(MemSaver CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MemSaver::memsaver)
```

## Public API

The main entrypoint is the `MemSaver` class:

```cpp
class MemSaver {
 public:
  cudaError_t enter_region(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode = AllocationKind::REGULAR);
  cudaError_t leave_region();
  cudaError_t evict_region_pool_from_cache(
      const std::string& tag,
      bool enable_cpu_backup,
      AllocationKind mode = AllocationKind::REGULAR);
};
```

Additional exported functions in [`include/memsaver/entrypoint.h`](./include/memsaver/entrypoint.h):

- `memsaver_malloc` and `memsaver_free`
- `memsaver_torch_malloc` and `memsaver_torch_free`
- `memsaver_pause` and `memsaver_resume`
- `memsaver_empty_cache`
- `memsaver_activate_arena_offsets`
- `memsaver_deactivate_arena_offsets`
- `memsaver_get_metadata_count_by_tag`
- `memsaver_get_cpu_backup_pointer`

## Minimal Usage

```cpp
#include <memsaver/entrypoint.h>
#include <torch/torch.h>

MemSaver memsaver;
auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

memsaver.enter_region("weights", true, AllocationKind::REGULAR);
torch::Tensor tensor = torch::empty({20 * 1024 * 1024}, options);
memsaver.leave_region();

memsaver_pause("weights");
memsaver_resume("weights");

tensor = torch::Tensor();
memsaver.evict_region_pool_from_cache("weights", true, AllocationKind::REGULAR);
```

## Region Semantics

- Regions are thread-local.
- Nested regions on the same thread are rejected.
- Child threads do not inherit an active region; they must call `enter_region` themselves.
- `leave_region()` stops routing allocations into the region pool and releases the pool back to Torch's allocator, but the cached pool entry remains reusable until `evict_region_pool_from_cache(...)` removes it.
- `memsaver_pause(nullptr)` and `memsaver_resume(nullptr)` operate on all managed tags.
- `enable_cpu_backup` is normalized off for `AllocationKind::ARENA`; CPU backup only applies to `AllocationKind::REGULAR`.

## Allocation Modes

`AllocationKind::REGULAR`

- Designed for regular managed allocations
- Supports CPU backup
- Covered by pause and resume tests with and without preserved tensor contents

`AllocationKind::ARENA`

- Designed for arena-style virtual ranges
- Exposes `memsaver_activate_arena_offsets(...)` and `memsaver_deactivate_arena_offsets(...)`
- Current tests exercise sparse activation of subranges inside a larger Torch tensor and verify that deactivated ranges fall back to the common backing behavior

## Tests

Build-only helper:

```bash
./tests/run_all_cpp_tests.sh
```

Torch runtime tests:

```bash
./tests/run_torch_basic_test.sh
./tests/run_torch_arena_test.sh
```

If the test binaries were added to CTest, you can also run:

```bash
ctest --test-dir build --output-on-failure
```

Current coverage includes:

- Regular allocations with tag-based metadata tracking
- Pause and resume with CPU backup enabled
- Pause and resume without CPU backup
- Matmul scenarios that mix managed regions with Torch's default pool
- Same-thread and child-thread region behavior
- Region reentry and address reuse
- Arena common backing
- Arena offset activation
- Arena offset deactivation

## Repository Layout

```text
.
├── include/memsaver/entrypoint.h
├── src/entrypoint.cpp
├── src/internal/
├── tests/basic_test.cpp
├── tests/arena_test.cpp
├── tests/test_utils.h
├── tests/run_torch_basic_test.sh
├── tests/run_torch_arena_test.sh
└── build.sh
```
