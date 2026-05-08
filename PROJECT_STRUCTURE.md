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
│   ├── sh/run_all_cpp_tests.sh       # Build the memsaver target
│   ├── sh/run_basic_test.sh          # Build and run the basic test
│   ├── sh/run_arena_test.sh          # Build and run the arena test
│   ├── sh/run_arena_virtual_test.sh  # Build and run the arena-virtual test
│   ├── sh/run_model_load_test.sh     # Build and run the model-load test
│   ├── utils/test_utils.h            # Shared test helpers
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
- `tests/` contains runtime tests and helper scripts.
