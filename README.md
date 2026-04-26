# MemSaver

MemSaver is a pure C++/CUDA third-party library for temporarily releasing GPU memory while preserving virtual addresses.

It provides:
- A stable C ABI (`include/memsaver/memsaver_c.h`)
- A lightweight C++ RAII wrapper (`include/memsaver/memsaver.hpp`)
- An `LD_PRELOAD` hook library (`libmemsaver_preload.so`) that intercepts `cudaMalloc/cudaFree`

Supported platform (v1):
- Linux x86_64
- CUDA 12.x

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Artifacts:
- `libmemsaver_core.so`
- `libmemsaver_core.a`
- `libmemsaver_preload.so`

## Install And Consume

Install:

```bash
cmake --install build --prefix /your/install/prefix
```

Use from CMake project:

```cmake
find_package(MemSaver CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MemSaver::memsaver_core_shared)
```

## C API Example

```c
#include <memsaver/memsaver_c.h>

memsaver_ctx_t* ctx = NULL;
memsaver_ctx_create(&ctx);

memsaver_set_interesting_region(ctx, true);
memsaver_set_current_tag(ctx, "weights");
memsaver_set_enable_cpu_backup(ctx, true);

void* ptr = NULL;
memsaver_malloc(ctx, &ptr, 1 << 20);

memsaver_pause(ctx, "weights");
memsaver_resume(ctx, "weights");

memsaver_free(ctx, ptr);
memsaver_ctx_destroy(ctx);
```

## Preload Mode

`libmemsaver_preload.so` hooks `cudaMalloc/cudaFree`:

```bash
LD_PRELOAD=/path/to/libmemsaver_preload.so ./your_cuda_program
```

Thread-local default interesting-region state can be configured via:
- `MEMSAVER_ENABLE` (`0/1`, `true/false`)

Notes:
- `CPU backup` and `allocation mode` are configured only by preload control APIs.
- Preload config is `thread_local`; child threads do not inherit parent-thread
  runtime overrides automatically.

The preload library also exports control symbols (for integration/tests):
- `memsaver_preload_set_interesting_region`
- `memsaver_preload_set_current_tag`
- `memsaver_preload_set_enable_cpu_backup`
- `memsaver_preload_set_allocation_mode`
- `memsaver_preload_region_begin`
- `memsaver_preload_region_end`
- `memsaver_preload_pause`
- `memsaver_preload_resume`

`memsaver_preload_region_begin/end` provide a torch-style region:
- inside region: allocations are routed to a tag-split private pool and managed by MemSaver
- outside region: allocation follows current preload switch/config (e.g. default torch pool when disabled)

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Included tests:
- preload smoke test (via `LD_PRELOAD`)
- torch basic test (built only when Torch C++ package is found)
