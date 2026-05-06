#include "utils/custom_torch_allocator.h"

#define main memsaver_original_basic_main
#include "../tests/basic_test.cpp"
#undef main

int main() {
  InstallCustomTorchAllocator();
  return memsaver_original_basic_main();
}
