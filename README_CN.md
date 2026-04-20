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

可通过环境变量设置线程局部默认配置：
- `MEMSAVER_INIT_ENABLE`（`0/1` 或 `true/false`）
- `MEMSAVER_INIT_ENABLE_CPU_BACKUP`（`0/1` 或 `true/false`）
- `MEMSAVER_INIT_ALLOCATION_MODE`（`normal` 或 `arena`）

preload 库还导出了一组控制符号（用于集成和测试）：
- `memsaver_preload_set_interesting_region`
- `memsaver_preload_set_current_tag`
- `memsaver_preload_set_enable_cpu_backup`
- `memsaver_preload_set_allocation_mode`
- `memsaver_preload_pause`
- `memsaver_preload_resume`

## 测试

```bash
ctest --test-dir build --output-on-failure
```

当前包含测试：
- 核心行为（`pause/resume`、tag 过滤、CPU backup）
- arena 行为（地址复用、reset 约束、OOM）
- 多 GPU 行为（GPU 数量不足时自动跳过）
- preload 冒烟测试（通过 `LD_PRELOAD`）
