# torch_memory_saver Product / Technical Spec

## 1. 背景

在大模型推理和训练场景中，GPU 上存在一类“长期对象”：

- KV cache
- 模型权重
- graph capture 里的大中间张量
- 训练引擎中的长生命周期缓冲区

这些对象并不总是在每一时刻都需要占用物理显存，但它们往往又要求地址稳定，不能简单 delete 后重建，否则会破坏 CUDA Graph、缓存复用和部分运行时假设。

`torch_memory_saver` 要解决的问题是：

在不破坏地址稳定性的前提下，允许这类对象临时释放物理显存，并在需要时恢复。

## 2. 产品目标

项目目标是提供一个 PyTorch 扩展，使用户可以：

- 在特定代码区域内创建“受管”的 CUDA tensor
- 按 tag 或整体释放这些 tensor 背后的物理显存
- 在之后重新恢复这些 tensor
- 让这些 tensor 的虚拟地址尽量保持不变
- 在需要时选择是否保留原始数据内容

## 3. 非目标

这个项目不做以下事情：

- 不提供通用 CPU offload 框架
- 不负责张量生命周期管理本身
- 不替代 PyTorch allocator 的全部功能
- 不保证未开启 CPU backup 的数据在恢复后仍然正确
- 不追求支持所有 CUDA allocator 特性组合

## 4. 目标用户

主要用户包括：

- 大模型推理框架开发者
- 训练引擎/运行时开发者
- 依赖 CUDA Graph 的系统开发者
- 希望对 KV cache 或模型权重做显存调度的基础设施工程师

## 5. 核心能力

### 5.1 Managed region

用户可以在一个上下文中创建受管分配：

```python
with torch_memory_saver.region(tag="kv_cache", enable_cpu_backup=False):
    kv = ...
```

在该 region 中创建的 CUDA tensor 会被 memory saver 接管。

### 5.2 Pause / Resume

用户可以：

- `pause()` / `resume()` 全量操作
- `pause(tag)` / `resume(tag)` 按组操作

暂停的目标效果：

- 保留 tensor 对应的虚拟地址
- 释放物理显存
- 若开启 CPU backup，则保留内容

恢复的目标效果：

- 把物理显存重新绑定到原虚拟地址
- 若有 CPU backup，则恢复原内容

### 5.3 CPU backup

用户可按 region 选择是否开启 CPU backup。

开启时：

- 数据语义保持

关闭时：

- 仅保证地址语义和后续可访问性
- 不保证恢复后的数值内容

### 5.4 CUDA Graph 兼容

在 `preload` 模式下，项目应支持 graph 相关场景，使 graph 内受管对象在 pause/resume 后仍然可 replay。

### 5.5 disable 逃逸区

用户可在 memory saver 上下文内部暂时关闭托管，以执行一小段普通 CUDA 分配逻辑。

### 5.6 多 tag / 多卡 / 嵌套 region

项目必须支持：

- 多个 tag 独立控制
- 多 GPU 恢复到原设备
- 已开启状态下的嵌套 region，并能在退出时恢复外层配置

## 6. 用户接口

Python 导入接口：

```python
from torch_memory_saver import torch_memory_saver, configure_subprocess
```

全局单例 `torch_memory_saver` 提供：

- `region(tag="default", enable_cpu_backup=False)`
- `cuda_graph(cuda_graph, pool=None, stream=None, capture_error_mode="global", tag="default", enable_cpu_backup=False)`
- `disable()`
- `pause(tag=None)`
- `resume(tag=None)`
- `get_cpu_backup(x, zero_copy=False)`
- `hook_mode`
- `memory_margin_bytes`
- `enabled`

约束：

- `enabled` 恒为真
- `memory_margin_bytes` 仅要求 setter
- `hook_mode` 支持 `"preload"` 与 `"torch"`

## 7. 技术方案概述

### 7.1 总体思路

采用“虚拟地址保持 + 物理显存解绑/重绑”的设计，而不是简单删除重建。

底层能力应建立在 CUDA 虚拟内存管理之上，等价于：

- reserve VA
- create allocation handle
- map
- unmap/release
- remap to same VA

### 7.2 线程局部配置

需要一套线程局部上下文状态，至少追踪：

- 当前 tag
- 当前线程是否处于 managed / interesting 状态
- 当前线程是否开启 CPU backup

原因：

- region 进入/退出需要切换状态
- 嵌套 region 需要恢复旧状态
- disable 需要临时关闭托管
- graph capture 需要在 capture 期间维持正确配置

### 7.3 分配元数据

需要维护全局 allocation registry，用于支持：

- free
- pause
- resume
- get_cpu_backup

元数据至少包含：

- ptr
- size
- device
- tag
- active/paused
- CPU backup 开关
- CPU backup 指针
- 底层 allocation handle

## 8. Hook 模式设计

### 8.1 preload 模式

默认模式。

机制：

- 通过 `LD_PRELOAD` 注入原生动态库
- hook 底层 `cudaMalloc/cudaFree`
- 在 interesting region 中接管分配
- 在非 interesting region 中转发到真实 allocator

配套要求：

- 提供 `configure_subprocess()` 帮助用户设置子进程环境
- 该函数必须 prepend 到 `LD_PRELOAD`
- 不得覆盖已有 preload 条目
- 退出上下文后必须恢复原值

### 8.2 torch 模式

机制：

- 使用 `CUDAPluggableAllocator`
- 在 region 内通过 `torch.cuda.MemPool` 使用自定义 allocator

约束：

- 保持与 preload 模式一致的 `pause/resume/tag/backup` 语义
- graph 内中间分配的 pauseable 能力主要由 preload 模式承担

## 9. 平台支持

平台范围只要求支持 CPU + CUDA GPU。

要求：

- GPU 路径基于 CUDA
- CPU 用于 Python 运行环境和可选的 CPU backup
- 平台范围限定为 CUDA

## 10. 构建与打包

项目应通过 `setup.py` 构建，不依赖 `pyproject.toml`。

需要产出两个原生扩展：

- preload 模式扩展
- torch 模式扩展

同时要提供：

- `MANIFEST.in`
- `Makefile`
- `scripts/build.sh`
- `scripts/rename_wheels.sh`

这些文件的目标分别是：

- 本地安装/重装
- wheel 构建
- sdist 构建
- wheel 文件名规范化

## 11. 测试策略

至少应包含：

### 11.1 配置测试

验证 `configure_subprocess()`：

- 在 `LD_PRELOAD` 为空时正确设置
- 在已有值时 prepend 而非覆盖
- 上下文退出后恢复原值

### 11.2 示例回归测试

应覆盖：

- basic pause/resume
- CPU backup
- CUDA Graph
- multi-device
- RL/tutorial 场景
- training engine 场景
- nested region 场景

建议在子进程中执行，尤其是 preload 模式。

## 12. 兼容性和边界条件

必须处理：

- `PYTORCH_CUDA_ALLOC_CONF=...expandable_segments:True...` 时明确拒绝运行
- preload 模式下动态定位自身 `.so`
- 仅对 contiguous CUDA tensor 提供 `get_cpu_backup`
## 13. 交付物

最终交付物包括：

- 可安装仓库
- Python 包
- 原生扩展源码
- 构建脚本
- README
- 完整测试与示例

推荐的 Python 包结构：

- `torch_memory_saver/__init__.py`
- `torch_memory_saver/entrypoint.py`
- `torch_memory_saver/binary_wrapper.py`
- `torch_memory_saver/utils.py`
- `torch_memory_saver/testing_utils.py`
- `torch_memory_saver/hooks/base.py`
- `torch_memory_saver/hooks/mode_preload.py`
- `torch_memory_saver/hooks/mode_torch.py`

## 14. 验收标准

项目验收以可观察行为为准：

1. `pip install .` 成功
2. `import torch_memory_saver` 成功
3. 全局单例与 `configure_subprocess` 可导入
4. `preload` 和 `torch` 两种模式均可用
5. 受管张量 pause/resume 后地址不变
6. CPU backup 打开时内容恢复正确
7. tag 粒度控制生效
8. preload 模式下 graph 场景可用
9. `disable()` 临时逃逸有效
10. 多卡恢复正确
11. 嵌套 region 能恢复外层状态
12. 测试集通过

## 15. 实施优先级

若实现成本受限，优先级如下：

1. CUDA 正确性
2. Python API 兼容
3. 地址稳定的 pause/resume 语义
4. preload 模式
5. torch 模式
6. 打包和发布脚本细节
