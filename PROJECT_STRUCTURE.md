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
└── tests/                            # C++ tests
    └── cpp/
        ├── core_tests.cpp            # Core behavior tests
        └── preload_smoke.cpp         # Preload smoke test
```

## Notes

- `include/` contains installable public headers.
- `src/internal/` contains private implementation details.
- `tests/cpp/` contains runtime tests executed via CTest.
