# MemSaver 项目结构说明

本文档描述当前仓库的目录结构。

```text
.
├── include/
│   └── memsaver/
│       └── entrypoint.h              # 对外公开 API
├── src/
│   ├── entrypoint.cpp                # 公开入口实现
│   └── internal/                     # 内部实现细节
├── tests/
│   ├── basic_test.cpp                # regular 分配与 region 行为测试
│   ├── arena_test.cpp                # arena 分配行为测试
│   ├── run_all_cpp_tests.sh          # 构建 memsaver target
│   ├── run_torch_basic_test.sh       # 构建并运行 Torch 基础测试
│   ├── run_torch_arena_test.sh       # 构建并运行 Torch arena 测试
│   ├── test_utils.h                  # 共享测试工具
│   └── 测试项目.md                   # 测试用例说明
├── build.sh                          # 构建辅助脚本
├── README.md                         # 英文说明
├── README_CN.md                      # 中文说明
├── PROJECT_STRUCTURE.md              # 本文档英文版
└── PROJECT_STRUCTURE_CN.md           # 本文档
```

## 说明

- `include/` 放可安装的公开头文件。
- `src/internal/` 放内部实现细节，不作为公开接口。
- `tests/` 放基于 Torch 的运行时测试和辅助脚本。
