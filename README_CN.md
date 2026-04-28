# MemSaver（中文说明）

MemSaver 是一个 C++/CUDA 库，集成在 PyTorch CUDA caching allocator 的 MemPool 和 segment 层。它可以把一个带 tag 的 region 内的分配路由到独立 pool，在保持虚拟地址稳定的前提下 pause 和 resume 已接管的 GPU 显存，并支持 regular 与 arena 两种分配模式。

仓库对外暴露的公开头文件是 [`include/memsaver/entrypoint.h`](./include/memsaver/entrypoint.h)，构建产物为核心库。

## 当前能力

- 通过 `MemSaver` 控制 region 作用域内的分配
- 按 `(tag, enable_cpu_backup, allocation_mode)` 复用独立 pool
- 对已接管分配执行 pause 和 resume
- regular 模式下可选 CPU backup
- arena 模式下支持按 offset 激活和取消激活
- 提供基于 Torch 的 regular 与 arena 运行时测试

## 依赖要求

- Linux x86_64
- CUDA toolkit
- CMake 3.20 及以上
- C++17
- 能被 CMake 找到的 PyTorch 或 LibTorch

`CMakeLists.txt` 会先尝试 `find_package(Torch)`，如果失败，再通过 `python -c "import torch; print(torch.utils.cmake_prefix_path)"` 查找 Torch 的 CMake 包。按当前源码结构，构建 `memsaver_core` 时也需要 Torch 头文件和库可用。

## 构建

推荐方式：

```bash
./build.sh
```

等价的 CMake 流程：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物：

- `libmemsaver_core.so`
- `libmemsaver_core.a`

检测到 Torch 后还会构建测试程序：

- `memsaver_torch_basic_test`
- `memsaver_torch_arena_test`

也可以只构建指定 target：

```bash
./build.sh --target memsaver_core_shared --target memsaver_core_static
./build.sh --target memsaver_torch_basic_test
./build.sh --target memsaver_torch_arena_test
```

## 安装与集成

```bash
cmake --install build --prefix /your/install/prefix
```

在其他 CMake 项目中使用：

```cmake
find_package(MemSaver CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MemSaver::memsaver_core_shared)
```

安装时会同时导出 `MemSaver::memsaver_core_shared` 和 `MemSaver::memsaver_core_static`。

## 公开 API

核心入口是 `MemSaver` 类：

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

[`include/memsaver/entrypoint.h`](./include/memsaver/entrypoint.h) 里还导出了这些函数：

- `memsaver_malloc` 和 `memsaver_free`
- `memsaver_torch_malloc` 和 `memsaver_torch_free`
- `memsaver_pause` 和 `memsaver_resume`
- `memsaver_empty_cache`
- `memsaver_activate_arena_offsets`
- `memsaver_deactivate_arena_offsets`
- `memsaver_get_metadata_count_by_tag`
- `memsaver_get_cpu_backup_pointer`

## 最小使用示例

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

## Region 语义

- region 是线程局部的
- 同一线程不允许嵌套 region
- 子线程不会继承父线程的激活 region，需要自行调用 `enter_region`
- `leave_region()` 会停止把后续分配路由到该 region 的 pool，并把 pool 释放回 Torch allocator，但缓存项仍可复用，直到显式调用 `evict_region_pool_from_cache(...)`
- `memsaver_pause(nullptr)` 和 `memsaver_resume(nullptr)` 会作用到所有已管理 tag
- `AllocationKind::ARENA` 下会忽略 `enable_cpu_backup`，CPU backup 只对 `AllocationKind::REGULAR` 生效

## 分配模式

`AllocationKind::REGULAR`

- 用于普通的受管分配
- 支持 CPU backup
- 当前测试覆盖了开启和关闭 CPU backup 时的 pause 和 resume 行为

`AllocationKind::ARENA`

- 用于 arena 风格的虚拟地址范围
- 提供 `memsaver_activate_arena_offsets(...)` 和 `memsaver_deactivate_arena_offsets(...)`
- 当前测试覆盖了在大张量中按 offset 激活子区间，以及取消激活后回退到共享 backing 的行为

## 测试

只构建核心库的辅助脚本：

```bash
./tests/run_all_cpp_tests.sh
```

Torch 运行时测试：

```bash
./tests/run_torch_basic_test.sh
./tests/run_torch_arena_test.sh
```

如果测试 target 已加入 CTest，也可以执行：

```bash
ctest --test-dir build --output-on-failure
```

当前覆盖内容包括：

- regular 分配与 tag 维度 metadata 统计
- 开启 CPU backup 时的 pause 和 resume
- 关闭 CPU backup 时的 pause 和 resume
- 受管 region 与 Torch 默认 pool 混合出现时的 matmul 场景
- 同线程与子线程下的 region 行为
- 重入同一 region 后的地址复用
- arena 共享 backing
- arena offset 激活
- arena offset 取消激活

## 仓库结构

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
