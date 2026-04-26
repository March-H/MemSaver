# MemSaver（中文说明）

MemSaver 是一个纯 C++/CUDA 的 third-party 库，用于在保持虚拟地址不变的前提下，临时释放 GPU 物理显存。

它提供：
- 稳定的 C ABI（`include/memsaver/memsaver_c.h`）
- 轻量的 C++ RAII 封装（`include/memsaver/memsaver.hpp`）
- `LD_PRELOAD` Hook 库（`libmemsaver_preload.so`），可拦截 `cudaMalloc/cudaFree`

当前支持平台（v1）：
- Linux x86_64
- CUDA 12.x

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物：
- `libmemsaver_core.so`
- `libmemsaver_core.a`
- `libmemsaver_preload.so`

## 安装与集成

安装：

```bash
cmake --install build --prefix /your/install/prefix
```

在 CMake 项目中使用：

```cmake
find_package(MemSaver CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MemSaver::memsaver_core_shared)
```

## C API 示例

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

## Preload 模式

`libmemsaver_preload.so` 可 Hook `cudaMalloc/cudaFree`：

```bash
LD_PRELOAD=/path/to/libmemsaver_preload.so ./your_cuda_program
```

可通过环境变量设置线程局部默认“接管开关”：
- `MEMSAVER_ENABLE`（`0/1` 或 `true/false`）

说明：
- `CPU backup` 与 `allocation mode` 仅通过 preload 控制 API 设置。
- preload 配置是 `thread_local`；子线程不会自动继承父线程运行时配置。

preload 库还导出了一组控制符号（用于集成和测试）：
- `memsaver_preload_set_interesting_region`
- `memsaver_preload_set_current_tag`
- `memsaver_preload_set_enable_cpu_backup`
- `memsaver_preload_set_allocation_mode`
- `memsaver_preload_region_begin`
- `memsaver_preload_region_end`
- `memsaver_preload_pause`
- `memsaver_preload_resume`

`memsaver_preload_region_begin/end` 提供 torch 风格 region 语义：
- region 内：分配进入按 tag 切分的私有池，并由 MemSaver 接管
- region 外：按当前 preload 开关/配置走常规路径（例如关闭时走 torch 默认池）

## 测试

```bash
ctest --test-dir build --output-on-failure
```

当前包含测试：
- preload 冒烟测试（通过 `LD_PRELOAD`）
- torch 基础测试（仅在检测到 Torch C++ 包时构建）
