# MemSaver Project Structure

This document describes the current repository layout.

```text
.
├── include/
│   └── memsaver/
│       └── entrypoint.h              # Public API
├── src/
│   ├── entrypoint.cpp                # Public entrypoint implementation
│   └── internal/                     # Internal implementation details
├── tests/
│   ├── basic_test.cpp                # Regular allocation and region behavior tests
│   ├── arena_test.cpp                # Arena allocation behavior tests
│   ├── run_all_cpp_tests.sh          # Build the memsaver target
│   ├── run_torch_basic_test.sh       # Build and run the Torch basic test
│   ├── run_torch_arena_test.sh       # Build and run the Torch arena test
│   ├── test_utils.h                  # Shared test helpers
│   └── 测试项目.md                   # Test case notes
├── build.sh                          # Build helper
├── README.md                         # English overview
├── README_CN.md                      # Chinese overview
├── PROJECT_STRUCTURE.md              # This document
└── PROJECT_STRUCTURE_CN.md           # Chinese version of this document
```

## Notes

- `include/` contains the installable public header.
- `src/internal/` contains private implementation details.
- `tests/` contains Torch-based runtime tests and helper scripts.
