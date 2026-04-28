# MemSaver Project Structure

This document describes the current repository layout after the pure C++/CUDA migration.

```text
.
├── include/                          # Public API surface
│   └── memsaver/
│       ├── memsaver_c.h              # C ABI header
│       └── memsaver.hpp              # C++ wrapper header
├── src/                              # Core implementation
│   ├── memsaver_c.cpp                # C ABI implementation
│   ├── preload.cpp                   # Preload hook implementation
│   └── internal/                     # Internal modules (non-public)
└── tests/                            # C++ tests and helpers
    ├── basic_test.cpp                # Torch integration/basic behavior test
    ├── run_all_cpp_tests.sh          # Test build entrypoint
    ├── run_torch_basic_test.sh       # Torch basic test runner
    ├── test_utils.h                  # Test helper utilities
    └── 测试项目.md                   # Test case notes
```

## Notes

- `include/` contains installable public headers.
- `src/internal/` contains private implementation details.
- `tests/` contains runtime tests executed via CTest.
