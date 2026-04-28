# MemSaver 项目结构说明

本文档描述纯 C++/CUDA 改造后当前仓库的目录结构。

```text
.
├── include/                          # 对外公开 API
│   └── memsaver/
│       ├── memsaver_c.h              # C ABI 头文件
│       └── memsaver.hpp              # C++ 封装头文件
├── src/                              # 核心实现
│   ├── memsaver_c.cpp                # C ABI 实现
│   ├── preload.cpp                   # preload Hook 实现
│   └── internal/                     # 内部模块（非公开）
└── tests/                            # C++ 测试与辅助文件
    ├── basic_test.cpp                # Torch 集成/基础行为测试
    ├── run_all_cpp_tests.sh          # 测试构建入口
    ├── run_torch_basic_test.sh       # Torch 基础测试脚本
    ├── test_utils.h                  # 测试辅助工具
    └── 测试项目.md                   # 测试用例说明
```

## 说明

- `include/` 放可安装的对外公开头文件。
- `src/internal/` 放内部实现细节，不作为公开接口。
- `tests/` 放通过 CTest 执行的运行时测试。
