#include "utils/custom_torch_allocator.h"

#define main memsaver_original_model_load_main
#include "../tests/model_load_test.cpp"
#undef main

int main() {
  InstallCustomTorchAllocator();
  return memsaver_original_model_load_main();
}
