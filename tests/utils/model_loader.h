#ifndef MEMSAVER_TESTS_MODEL_LOADER_H_
#define MEMSAVER_TESTS_MODEL_LOADER_H_

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

struct LoadedModelTensor {
  std::string name;
  torch::Tensor tensor;
  uint64_t nbytes = 0;
};

struct LoadedModelWeights {
  std::string model_dir;
  std::vector<LoadedModelTensor> tensors;
  uint64_t total_bytes = 0;
};

LoadedModelWeights LoadModelLikeXllmOnCuda(
    const std::string& model_dir,
    const torch::Device& device = torch::Device(torch::kCUDA, 0));

#endif
