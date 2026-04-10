# torch_memory_saver Rebuild Prompt For Model

你是一名资深的系统/基础设施工程师。现在你要在完全看不到原始源码的前提下，从零实现一个名为 `torch_memory_saver` 的仓库。不要写 demo，不要写 mock，不要写“思路说明”，直接产出完整代码、构建脚本、测试和文档。目标是：让这个仓库在功能、公开接口、使用方式和关键行为上，等价于一个“可暂停释放 CUDA tensor 物理显存、之后再恢复、同时尽量保持虚拟地址不变”的 PyTorch 扩展项目。范围只要求支持 CPU + CUDA GPU。

## 你的任务

你要实现一个 Python + C++/CUDA 项目，支持两种 hook 模式：

- `preload`
- `torch`

这个项目必须让用户能够把某些在 PyTorch 中创建的 CUDA tensor 标记为“可暂停”。暂停时：

- 保留虚拟地址
- 释放实际物理显存
- 如果启用了 CPU backup，则把内容备份到 CPU

恢复时：

- 在原虚拟地址重新映射新的物理显存
- 如果启用了 CPU backup，则把内容恢复回去
- 如果没有 CPU backup，则张量内容可以变化，但地址仍应可继续使用

## 核心场景

你的实现必须覆盖这些场景：

1. 基础 pause/resume
   用户在一个上下文里创建大 tensor，`pause()` 后显存下降，`resume()` 后显存回升，tensor 的 `data_ptr()` 前后一致。

2. tag 分组
   用户可以通过 `tag` 把不同张量分组，并对某一个 tag 单独 `pause/resume`。

3. CPU backup
   启用了 CPU backup 的张量，在 pause/resume 后内容保持不变；未启用的张量不保证内容保持。

4. CUDA Graph
   在 `preload` 模式下，用户可以用一个辅助 graph 上下文创建可 pause 的 graph 内存；pause 后显存明显下降，resume 后 graph replay 继续成功。

5. disable 区域
   用户处在 memory saver 管理区域中时，可以临时进入 `disable()`，在这一小段代码里按普通 CUDA allocator 分配临时对象，不被 memory saver 接管；退出后恢复原状态。

6. 多卡
   即使当前 device 在 GPU 1，用户也可以在 region 内显式对 `cuda:0` 和 `cuda:1` 分配；pause/resume 后这些 allocation 要回到原设备。

7. 嵌套 region
   用户可以在默认已开启的 memory saver 上下文中，再进入一个临时切换 tag / CPU backup 策略的嵌套 region；退出后外层状态必须恢复。

## 公开 Python API

实现后必须支持：

```python
from torch_memory_saver import torch_memory_saver, configure_subprocess
```

其中全局单例 `torch_memory_saver` 至少暴露：

- `region(tag: str = "default", enable_cpu_backup: bool = False)`
- `cuda_graph(cuda_graph, pool=None, stream=None, capture_error_mode="global", tag="default", enable_cpu_backup=False)`
- `disable()`
- `pause(tag: str | None = None)`
- `resume(tag: str | None = None)`
- `get_cpu_backup(x: torch.Tensor, zero_copy: bool = False)`
- 可写属性 `hook_mode`，支持 `"preload"` 与 `"torch"`
- 可写属性 `memory_margin_bytes`
- 只读属性 `enabled`，始终为 `True`

要求：

- `pause(None)` / `resume(None)` 表示全部
- `get_cpu_backup(x)` 只要求支持 contiguous CUDA tensor
- `cuda_graph(...)` 在 `preload` 模式下必须具备“graph 内受管分配也可 pause”的能力
- `torch` 模式下不强求 graph 内中间内存可 pause，但其余能力仍要保留

## 两种 hook 模式的要求

### preload

你必须构建一个可通过 `LD_PRELOAD` 注入的动态库。它需要：

- 拦截底层 `cudaMalloc/cudaFree`
- 只在“当前线程处于 interesting region”时接管分配
- 不在 interesting region 时转发到真实底层分配器

同时实现：

```python
with configure_subprocess():
    ...
```

它必须：

- 将你的 preload `.so` prepend 到 `LD_PRELOAD`
- 不能覆盖已有 `LD_PRELOAD`
- 退出上下文后恢复原值

### torch

你必须使用 PyTorch 的 `CUDAPluggableAllocator` 路径，把 allocator 接到 `torch.cuda.MemPool` 上，让 `region()` 内的分配走你的自定义 allocator。

## 实现约束

不要用“删除 tensor 再新建”的方式伪装实现 pause/resume。你必须真正建立在 CUDA 虚拟内存管理能力之上，效果上等价于：

- 预留 VA
- 创建物理 allocation handle
- map 到该 VA
- pause 时 unmap/release 物理显存
- resume 时重新分配并 map 回相同 VA

你需要自己设计一套底层元数据系统，至少能记录：

- 指针地址
- size
- device
- tag
- active / paused 状态
- 是否启用 CPU backup
- CPU backup 指针
- 底层 allocation handle

你还需要一套线程局部状态，至少跟踪：

- 当前 tag
- 当前线程是否在 interesting region
- 当前线程是否启用 CPU backup

## 平台兼容

只要求支持 CPU + CUDA GPU。

要求：

- GPU 路径基于 CUDA
- CPU 只作为运行环境和 CPU backup 存储位置
- 平台范围限定为 CUDA

## 包和目录

请生成一个可安装的仓库，至少包含：

- `torch_memory_saver/`
- `csrc/`
- `test/`
- `scripts/`
- `setup.py`
- `README.md`
- `Makefile`
- `MANIFEST.in`

建议保留这些 Python 模块边界：

- `torch_memory_saver/__init__.py`
- `torch_memory_saver/entrypoint.py`
- `torch_memory_saver/binary_wrapper.py`
- `torch_memory_saver/utils.py`
- `torch_memory_saver/testing_utils.py`
- `torch_memory_saver/hooks/base.py`
- `torch_memory_saver/hooks/mode_preload.py`
- `torch_memory_saver/hooks/mode_torch.py`

## 测试和示例

请产出一组真实可运行的测试/示例，覆盖这些文件名：

- `test/test_configure_subprocess.py`
- `test/test_examples.py`
- `test/examples/simple.py`
- `test/examples/cpu_backup.py`
- `test/examples/cuda_graph.py`
- `test/examples/multi_device.py`
- `test/examples/rl_example.py`
- `test/examples/training_engine.py`
- `test/examples/nested_region.py`

这些测试必须验证：

- `configure_subprocess()` 正确 prepend 并恢复 `LD_PRELOAD`
- 基础 pause/resume 场景中显存变化显著且地址不变
- CPU backup 与非 backup 的差异行为
- graph 场景下显存可释放并成功 replay
- 多卡分配与恢复
- 默认开启 + disable 场景
- 嵌套 region 的状态恢复

建议使用子进程运行 GPU 示例，尤其是 preload 模式。

## 需要处理的边界条件

- 如果 `PYTORCH_CUDA_ALLOC_CONF` 中含 `expandable_segments:True`，直接报错并说明不支持
- preload 模式下要能正确找到你自己的 `.so`
- `get_cpu_backup(x)` 对不支持的 tensor 明确报错
## 你不能交付的东西

不要交付：

- 只有 Python mock、没有真实显存释放能力的版本
- 只能运行示例、不能安装的实验代码
- 只支持单模式或单卡的缩水实现
- 把规格再复述一遍但没有完整仓库文件

## 验收标准

你的最终结果必须满足：

1. `pip install .` 成功
2. `import torch_memory_saver` 成功
3. `from torch_memory_saver import torch_memory_saver, configure_subprocess` 成功
4. `hook_mode="preload"` 和 `"torch"` 均可使用
5. 受管张量 pause/resume 后地址保持不变
6. CPU backup 打开时内容保持
7. tag 粒度 pause/resume 生效
8. preload 模式 graph 场景可用
9. `disable()` 生效
10. 多卡可用
11. 嵌套 region 恢复外层状态

按“完整仓库”直接输出，不要输出额外解释。
