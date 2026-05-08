#include "utils/model_loader.h"

#include <cuda_runtime_api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils/allocator_installer.h"
#include "utils/test_utils.h"

struct ModelSpec {
  std::string path;
  std::string tag;
};

std::vector<ModelSpec> GetModelSpecs() {
  const std::vector<std::string> paths = {
      "/workspace/share/models/Qwen2.5-3B",
      "/workspace/share/models/Qwen3-1.7B",
      "/workspace/share/models/Qwen3-8B",
  };

  std::vector<ModelSpec> specs;
  specs.reserve(paths.size());
  for (const auto& path : paths) {
    specs.push_back(ModelSpec{
        path,
        std::filesystem::path(path).filename().string(),
    });
  }
  return specs;
}

uint64_t SafetensorsDataBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  CheckTrue(input.good(), ("failed to open " + path).c_str());

  uint64_t header_size = 0;
  input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  CheckTrue(input.good(), ("failed to read header size from " + path).c_str());

  const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(path));
  CheckTrue(
      file_size >= sizeof(uint64_t) + header_size,
      ("invalid safetensors size for " + path).c_str());
  return file_size - sizeof(uint64_t) - header_size;
}

uint64_t ModelDataBytes(const std::string& model_dir) {
  uint64_t total = 0;
  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.path().extension() == ".safetensors") {
      total += SafetensorsDataBytes(entry.path().string());
    }
  }
  CheckTrue(total != 0, ("no safetensors data found in " + model_dir).c_str());
  return total;
}

void TestLoadOneModel(MemSaver& memsaver, const ModelSpec& spec) {
  const uint64_t expected_model_bytes = ModelDataBytes(spec.path);
  const uint64_t baseline = DeviceUsedBytes();

  CheckCuda(
      memsaver.enter_region(spec.tag.c_str(), false, AllocationKind::REGULAR, 1),
      ("enter_region(" + spec.tag + ")").c_str());

  uint64_t preallocated_delta = 0;
  {
    auto preallocated = torch::empty(
        {static_cast<int64_t>(expected_model_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    SyncCuda();
    preallocated_delta = CurrentDeltaBytes(baseline);
    std::cout << "[" << CurrentTestName() << "] " << spec.tag
              << " preallocated bytes == "
              << expected_model_bytes / 1024.0 / 1024.0 << " MB"
              << ", preallocated allocation delta == "
              << preallocated_delta / 1024.0 / 1024.0 << " MB"
              << std::endl;
  }
  SyncCuda();

  LoadedModelWeights loaded =
      LoadModelLikeXllmOnCuda(spec.path, torch::Device(torch::kCUDA, 0));
  SyncCuda();
  CheckCuda(
      memsaver.leave_region(),
      ("leave_region(" + spec.tag + ")").c_str());

  const uint64_t observed_delta = CurrentDeltaBytes(baseline);
  CheckTrue(!loaded.tensors.empty(), "loaded model tensors should not be empty");
  CheckTrue(loaded.total_bytes == expected_model_bytes,
            "loaded model bytes should match safetensors data bytes");
  std::cout << "[" << CurrentTestName() << "] " << spec.tag
            << " loaded total bytes == "
            << loaded.total_bytes / 1024.0 / 1024.0 << " MB"
            << ", observed allocation delta == "
            << observed_delta / 1024.0 / 1024.0 << " MB"
            << std::endl;
  CheckTrue(observed_delta == preallocated_delta,
            "observed model allocation delta should match preallocated delta");
  ExpectMetadataCountByTag(spec.tag.c_str(), 1ULL, (spec.tag + " metadata count").c_str());

  for (const auto& loaded_tensor : loaded.tensors) {
    CheckManagedMetadataExistsForTensor(
        loaded_tensor.tensor,
        spec.tag + ":" + loaded_tensor.name);
  }

  loaded = LoadedModelWeights();
  SyncCuda();
  EmptyAllocatorCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          spec.tag.c_str(),
          false,
          AllocationKind::REGULAR,
          1),
      ("evict_region_pool_from_cache(" + spec.tag + ")").c_str());
  ExpectMetadataCountByTag(
      spec.tag.c_str(),
      0ULL,
      (spec.tag + " metadata count after evict").c_str());
  std::cout << "[" << CurrentTestName() << "] " << spec.tag
            << " residual delta after evict == "
            << CurrentDeltaBytes(baseline) / 1024.0 / 1024.0 << " MB"
            << std::endl;
}

int main() {
  InstallAllocator();
  SetTestName("model_load_test");
  if (MaybeSkipNoGpu()) {
    return 0;
  }

  WarmUpModelLoaderPath();

  MemSaver memsaver;
  for (const auto& spec : GetModelSpecs()) {
    TestLoadOneModel(memsaver, spec);
  }

  std::cout << "[" << CurrentTestName() << "] all tests passed" << std::endl;
  return 0;
}
