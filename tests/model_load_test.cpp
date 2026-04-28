#include "utils/model_loader.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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

void TestLoadOneModel(MemSaver& memsaver, const ModelSpec& spec) {
  const uint64_t baseline = DeviceUsedBytes();

  CheckCuda(
      memsaver.enter_region(spec.tag.c_str(), false, AllocationKind::REGULAR),
      ("enter_region(" + spec.tag + ")").c_str());
  LoadedModelWeights loaded =
      LoadModelLikeXllmOnCuda(spec.path, torch::Device(torch::kCUDA, 0));
  SyncCuda();
  CheckCuda(
      memsaver.leave_region(),
      ("leave_region(" + spec.tag + ")").c_str());

  CheckTrue(!loaded.tensors.empty(), "loaded model tensors should not be empty");
  CheckTrue(loaded.total_bytes != 0, "loaded model bytes should not be zero");
  CheckTrue(CurrentDeltaBytes(baseline) >= loaded.total_bytes,
            "observed model allocation delta should cover loaded tensor bytes");
  CheckTrue(MetadataCountByTag(spec.tag, spec.tag + " metadata count") != 0,
            "loaded model should create managed metadata");

  for (const auto& loaded_tensor : loaded.tensors) {
    CheckManagedMetadataExistsForTensor(
        loaded_tensor.tensor,
        spec.tag + ":" + loaded_tensor.name);
  }

  loaded = LoadedModelWeights();
  SyncCuda();
  EmptyTorchCache();
  CheckCuda(
      memsaver.evict_region_pool_from_cache(
          spec.tag.c_str(), false, AllocationKind::REGULAR),
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
