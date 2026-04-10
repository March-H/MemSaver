# torch_memory_saver Rebuild Prompt

下面这份内容不是“把现有源码翻译成文字版伪代码”，而是一份面向从零实现的产品规格。把它交给另一个代码模型时，假设对方完全看不到原仓库源码，只知道要做什么、暴露什么接口、满足什么行为和通过什么测试。目标是让它基于功能理解，把 `torch_memory_saver` 这个项目重新实现出来。

## Prompt

你要从零实现一个 Python + C++/CUDA 项目，项目名为 `torch_memory_saver`。不要假设你能看到任何原始代码，也不要尝试“照着已有实现改写”。你的任务是根据下面的功能规格、公开接口、构建要求和验收标准，独立设计并完成一个与目标项目行为等价的仓库。

核心要求只有一句话：

这个项目要让一部分 PyTorch CUDA tensor 所占用的物理显存被“暂停释放”，并在之后“恢复回来”，同时尽量保持这些 tensor 的虚拟地址不变，从而兼容依赖地址稳定性的场景，例如 CUDA Graph、KV cache、部分模型权重管理和训练引擎内的显存调度。

## 1. 你要交付的仓库

请交付一个可直接安装和测试的仓库，仓库名 `torch_memory_saver`，至少包含这些组成部分：

- Python 包 `torch_memory_saver/`
- 原生源码目录 `csrc/`
- 测试目录 `test/`
- 顶层 `setup.py`
- 顶层 `README.md`
- 顶层 `Makefile`
- 顶层 `MANIFEST.in`
- 构建脚本目录 `scripts/`

目录名字和 Python 模块名字要与这些名称保持一致，因为外部代码会按这些路径导入和调用。

你不需要做成逐字节相同的源码仓库，但必须让外部使用方式、构建方式和行为契约尽量一致。

## 2. 项目要解决的问题

这个库的目标不是普通的“把 tensor 移到 CPU 再移回来”，也不是普通的 cache eviction 工具。它要解决的是：

- 某些 CUDA tensor 是长期对象，比如 KV cache、模型权重、graph capture 里的中间对象。
- 在某些阶段这些对象暂时不用，但又希望之后恢复时继续复用原来的虚拟地址。
- 如果只是删除重建，会破坏地址稳定性，CUDA Graph 等场景会出问题。
- 所以这个库需要利用 CUDA 虚拟内存映射能力，把“虚拟地址保留”和“物理显存释放”拆开。

也就是说，暂停时应该尽量做到：

- 虚拟地址仍然保留在原地
- 对应的物理显存解绑/释放
- 如果启用了 CPU backup，则内容也被保存

恢复时应该尽量做到：

- 在同一个虚拟地址重新映射新的物理显存
- 如果启用了 CPU backup，则把原内容拷回
- 否则数据内容可以视为未定义，但地址应不变

## 3. 公开 Python 接口

你必须提供一个名为 `torch_memory_saver` 的 Python 包，并满足下面这些导入方式：

```python
from torch_memory_saver import torch_memory_saver, configure_subprocess
```

其中：

- `torch_memory_saver` 是一个全局单例对象
- `configure_subprocess` 是一个上下文管理器，只在 `preload` hook 模式下有意义

这个全局对象至少要暴露这些接口和属性：

- `region(tag: str = "default", enable_cpu_backup: bool = False)`
- `cuda_graph(cuda_graph, pool=None, stream=None, capture_error_mode="global", tag="default", enable_cpu_backup=False)`
- `disable()`
- `pause(tag: str | None = None)`
- `resume(tag: str | None = None)`
- `get_cpu_backup(x: torch.Tensor, zero_copy: bool = False)`
- `hook_mode` 可写，支持 `"preload"` 和 `"torch"`
- `memory_margin_bytes` 只需要支持 setter
- `enabled` 属性恒为 `True`

行为约束：

- `region(...)` 是上下文管理器。只有在这个上下文里创建的张量，才应该被纳入 memory saver 管理。
- `tag` 用来把不同对象分组，后续可以选择性 `pause("tag")` / `resume("tag")`。
- `enable_cpu_backup=True` 时，暂停前要把内容保留到 CPU；恢复后内容不变。
- `enable_cpu_backup=False` 时，恢复后地址应可继续访问，但内容可以变化。
- `pause(None)` / `resume(None)` 表示对全部受管对象生效。
- `cuda_graph(...)` 的目的不是替代 `torch.cuda.graph` 的全部功能，而是让 graph capture 里的受管分配也具备 pause/resume 能力。
- `disable()` 表示在当前处于 memory saver 管理中的上下文里，临时关闭这套机制，允许一小段代码按普通 CUDA 分配方式运行。

## 4. 两种 hook 模式

必须支持两种 hook 模式，语义如下：

### 4.1 preload 模式

这是默认模式。

思路：

- 通过 `LD_PRELOAD` 把你生成的一个动态库注入到 Python 子进程。
- 这个动态库要拦截底层 CUDA 的内存分配与释放接口。
- 只在“当前线程被标记为 interesting region”时，把分配接管到 memory saver；否则走真实分配函数。

对外要求：

- `configure_subprocess()` 需要把你的 preload 动态库 prepend 到 `LD_PRELOAD`，而不是覆盖已有值。
- 退出 context 后，原来的 `LD_PRELOAD` 必须被恢复。
- 如果环境中本来就有别的 `LD_PRELOAD` 项，也必须保留。

### 4.2 torch 模式

思路：

- 使用 PyTorch 的 `CUDAPluggableAllocator`
- 让 `torch.cuda.MemPool` 在 region 内使用你的自定义 allocator

对外要求：

- 用户可以通过：

```python
torch_memory_saver.hook_mode = "torch"
```

切到该模式。

- 该模式下仍然要支持：
  - `region`
  - `pause/resume`
  - `tag`
  - `enable_cpu_backup`
  - `multi-device`

- 但 `cuda_graph(...)` 只要求在 `preload` 模式下支持 pauseable graph；`torch` 模式下可以直接回退到普通 graph，不必强行支持 graph 内中间分配被 pause。

## 5. 实现层面的关键机制

实现方式不要求和任何现有代码一模一样，但必须满足这些本质约束：

### 5.1 不是 delete/recreate，而是 VA 保持 + PA 释放

你的暂停/恢复逻辑必须建立在 CUDA 虚拟内存映射能力之上，而不是简单地删除张量再新建。

你需要利用类似下面的能力：

- 预留虚拟地址
- 创建物理分配句柄
- 将物理显存映射到该虚拟地址
- 暂停时解绑/释放物理显存
- 恢复时重新分配物理显存并映射回原虚拟地址

### 5.2 线程局部状态

你需要有一套线程局部配置，至少包含这些概念：

- 当前 tag
- 当前线程是否处在 interesting region
- 当前线程是否启用 CPU backup

因为 `region()`、`disable()`、graph capture 等上下文切换都需要临时修改这几个状态，再在退出时恢复。

### 5.3 分配元数据表

你需要维护一张全局分配表，记录每个受管 allocation 的信息，例如：

- 原始虚拟地址
- 尺寸
- device
- tag
- 当前状态是 active 还是 paused
- 是否启用 CPU backup
- 是否存在 CPU backup
- 底层分配句柄

`pause` / `resume` / `free` / `get_cpu_backup` 都要依赖这张表。

### 5.4 CPU backup

启用 CPU backup 时：

- pause 前把 GPU 内容拷到 pinned host memory
- resume 后再拷回 GPU

不启用时：

- 恢复后内容不做正确性保证
- 但地址应仍可访问、可写、可继续用于后续计算

`get_cpu_backup(x)` 的语义：

- 如果 `x` 对应的受管 allocation 当前是 paused 且启用了 CPU backup，则返回 CPU 端内容
- 如果当前仍是 active，则返回 `None`
- 只要求支持 contiguous CUDA tensor

## 6. CUDA Graph 相关要求

这个项目的一个关键卖点是：在 `preload` 模式下，用户可以用一个辅助上下文包裹 graph capture，让 graph 内某些对象也能被 pause/resume。

要支持的核心场景：

- KV cache 在普通 `region(tag="kv_cache")` 下创建
- graph capture 中的大中间 tensor 在 `tag="graph"` 下创建
- 先 pause KV cache，再 pause graph 内存
- 释放出的显存足以临时容纳别的超大张量
- 然后先恢复 graph，再恢复 KV cache
- graph replay 仍然能成功

本质标准只有两个：

- graph replay 依赖的地址稳定性不能被破坏
- 暂停期间真的能释放出显著显存

## 7. disable() 的要求

`disable()` 不是简单的 no-op。

它要解决的问题是：

- 用户当前处在 memory saver 管理区域中
- 但某一小段代码希望完全按普通 CUDA allocator 行为分配临时对象

因此在 `disable()` 里：

- 新分配的对象不应被纳入当前 memory saver 管理集合
- 这段代码仍然应能正常执行和占用显存
- 退出 `disable()` 后，memory saver 的工作状态要恢复

同时，为了和 PyTorch allocator/mempool 配合，你需要认真处理 `MemPool` 的生命周期，而不是只改一个布尔开关。

## 8. 多 tag、多设备、嵌套 region

这些都属于必须支持的功能，不是“可选增强”。

### 多 tag

要支持：

- 不同对象用不同 tag 建立
- `pause("type1")` 只影响 type1
- `resume("type2")` 只恢复 type2

### 多设备

要支持至少这种场景：

- 当前 device 在 1 号卡
- 但 region 内仍然可以显式在 `cuda:0` 或 `cuda:1` 上分配
- pause/resume 后，这些 allocation 仍要在各自原本的 device 上恢复

### 嵌套 region

要支持这种情况：

- 进程启动时就通过环境变量把 memory saver 默认置为开启
- 默认区域的 allocation 会启用 CPU backup
- 在内部再进入一个 `region(tag="grad_buffer", enable_cpu_backup=False)` 的嵌套子区域
- 子区域退出后，外层默认 tag 和 backup 配置必须恢复

也就是说，`region()` 不能只会“从关闭切到开启”，它还必须会“在已经开启时临时切换 tag / CPU backup 策略，再恢复回来”。

## 9. 平台支持要求

平台范围只要求支持 CPU + CUDA GPU。

重点要求：

- GPU 路径以 CUDA 为准
- CPU 只作为 Python 运行环境和可选的 CPU backup 存储位置
- 平台范围限定为 CUDA

## 10. 构建和打包要求

项目必须通过 `setup.py` 构建，不依赖 `pyproject.toml`。

要求：

- Python >= 3.9
- 仅要求支持 CUDA
- 构建两个原生扩展：
  - 一个服务于 preload 模式
  - 一个服务于 torch 模式

扩展的职责分别是：

- preload 动态库：
  - 对外提供底层分配 hook
  - 同时提供给 Python `ctypes` 调用的控制接口
- torch 动态库：
  - 对外导出 `CUDAPluggableAllocator` 需要的 malloc/free 符号
  - 也提供同一套控制接口，便于 Python 层统一控制

额外要求：

- `MANIFEST.in` 要保证 C/C++ 头文件被包含进 source distribution
- 顶层 `Makefile` 要提供：
  - 本地重装
  - wheel 构建
  - sdist 构建
  - 上传
- `scripts/` 目录里要有 wheel 构建脚本和重命名脚本

这里不要求你逐字复刻原脚本，但要保留这些用途和大体行为。

## 11. 建议保留的 Python 包结构

虽然你可以自由实现细节，但为了兼容外部导入方式，建议保留这组模块边界：

- `torch_memory_saver/__init__.py`
- `torch_memory_saver/entrypoint.py`
- `torch_memory_saver/binary_wrapper.py`
- `torch_memory_saver/utils.py`
- `torch_memory_saver/testing_utils.py`
- `torch_memory_saver/hooks/base.py`
- `torch_memory_saver/hooks/mode_preload.py`
- `torch_memory_saver/hooks/mode_torch.py`

其中职责建议如下：

- `entrypoint.py`：全局单例、公开 API、上下文管理、与底层桥接
- `binary_wrapper.py`：`ctypes` 加载动态库并声明符号签名
- `utils.py`：找 `.so`、环境变量上下文工具
- `hooks/`：两种 hook 模式的路径解析与 allocator 接入

这不是在要求你照搬某个实现，而是在定义一个外部兼容的模块分层。

## 12. 必须满足的可观察行为

下面这些行为必须在实现中得到满足。

### 12.1 basic example

用户可以这样写：

```python
with torch_memory_saver.region():
    x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device="cuda")

addr = x.data_ptr()
torch_memory_saver.pause()
torch_memory_saver.resume()
assert x.data_ptr() == addr
```

并且：

- pause 后显存明显下降
- resume 后显存明显上升

### 12.2 CPU backup example

用户可以这样写：

```python
with torch_memory_saver.region(enable_cpu_backup=True):
    x = torch.full((20_000_000,), 10, dtype=torch.uint8, device="cuda")

torch_memory_saver.pause()
torch_memory_saver.resume()
assert x[:3].tolist() == [10, 10, 10]
```

而如果同样大小的张量是在 `enable_cpu_backup=False` 下创建，则恢复后不要求值保持原样。

### 12.3 tag example

要支持：

```python
with torch_memory_saver.region(tag="type1"):
    a = ...

with torch_memory_saver.region(tag="type2"):
    b = ...

torch_memory_saver.pause("type1")
torch_memory_saver.resume("type1")
```

且只影响对应 tag。

### 12.4 training engine example

要支持这样一种 preload-only 运行方式：

- 进程通过环境变量让 memory saver 默认处于开启状态
- 一批长期模型权重默认启用 CPU backup
- `pause()` 后显存显著下降
- `disable()` 块中仍能正常执行临时大分配和计算
- 退出 `disable()` 后显存恢复到 pause 后附近
- `resume()` 后权重和功能恢复

### 12.5 nested region example

要支持：

- 进程启动默认开启 backup
- 内层 region 临时关闭 backup
- 退出内层后恢复默认配置
- 外层权重 pause/resume 后内容保持
- 内层 grad buffer pause/resume 后内容不保证，但仍应可访问和覆写

## 13. 你需要写出的测试和示例

请交付一个与下面这组名字兼容的测试/示例集合：

- `test/test_configure_subprocess.py`
- `test/test_examples.py`
- `test/examples/simple.py`
- `test/examples/cpu_backup.py`
- `test/examples/cuda_graph.py`
- `test/examples/multi_device.py`
- `test/examples/rl_example.py`
- `test/examples/training_engine.py`
- `test/examples/nested_region.py`

这些文件的内部实现可以自行组织，但它们应覆盖这些验收点：

- `configure_subprocess()` 正确 prepend 并恢复 `LD_PRELOAD`
- `simple`：pause/resume 显存变化与地址不变
- `cpu_backup`：backup 开关的差异行为
- `cuda_graph`：graph 场景下按 tag 释放/恢复
- `multi_device`：多卡分配和恢复
- `rl_example`：较完整的教程式场景，包含 model weights、KV cache、graph、地址稳定性
- `training_engine`：默认开启 + disable + cache eviction 类场景
- `nested_region`：嵌套 region 恢复线程局部状态

建议这些测试通过子进程运行，尤其是 preload 模式，因为 `LD_PRELOAD` 和 CUDA 进程退出阶段的行为更安全。

## 14. 边界条件和兼容性约束

这些约束也要处理：

- 如果 `PYTORCH_CUDA_ALLOC_CONF` 含有 `expandable_segments:True`，直接禁用并抛出清晰错误，因为该模式与本项目不兼容。
- preload 模式下需要正确找到自己注入的 `.so` 路径，而不是硬编码一个固定路径。
- `get_cpu_backup(x)` 只要求支持 contiguous CUDA tensor；如果张量不满足条件，可以明确报错。
## 15. 不要做的事

不要把这个项目实现成下面这些东西：

- 一个简单的“张量搬到 CPU 再搬回来”的包装器
- 一个只用 Python mock 出来、没有真实显存释放能力的 demo
- 一个只支持单 tag、单卡、单模式的缩水版
- 一个只能跑示例、不能作为包安装的实验代码
- 一个把需求重新翻译成注释很多但功能不完整的仓库

## 16. 验收标准

最终交付必须满足：

1. `pip install .` 可以构建并安装项目。
2. `import torch_memory_saver` 成功。
3. `from torch_memory_saver import torch_memory_saver, configure_subprocess` 成功。
4. `torch_memory_saver.hook_mode = "preload"` 和 `"torch"` 都可用。
5. 通过 `region()` 创建的受管张量，在 pause/resume 后地址保持不变。
6. 开启 CPU backup 的张量，恢复后内容保持不变。
7. 未开启 CPU backup 的张量，恢复后内容可以变化，但张量仍可访问、可继续参与计算。
8. 选择性 tag pause/resume 生效。
9. preload 模式下 `cuda_graph(...)` 可以支撑 graph 场景。
10. `disable()` 能临时绕过 memory saver。
11. 多卡场景可用。
12. 嵌套 region 可以正确恢复先前线程局部状态。
13. `test/test_configure_subprocess.py` 和 `test/test_examples.py` 所覆盖的场景都能通过。

如果你在设计时需要权衡，请按下面的优先级排序：

1. 行为正确
2. Python API 和模块名兼容
3. pause/resume 的地址稳定性
4. preload/torch 两种 hook 模式都可用
5. CUDA 正确
6. 代码结构清晰、可维护

不要输出解释性文章，直接产出完整仓库文件。
