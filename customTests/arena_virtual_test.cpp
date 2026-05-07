#include "utils/custom_torch_allocator.h"

#define main memsaver_original_arena_virtual_main
#include "../tests/arena_virtual_test.cpp"
#undef main

int main() {
  InstallCustomTorchAllocator();
  return memsaver_original_arena_virtual_main();
}
