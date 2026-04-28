#include "model_loader.h"

#include <rapidjson/document.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "test_utils.h"

namespace {

[[noreturn]] void FailModelLoader(const std::string& message) {
  std::cerr << "[" << CurrentTestName() << "] " << message << std::endl;
  std::exit(1);
}

class ScopedMmap {
 public:
  explicit ScopedMmap(const std::string& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      FailModelLoader("failed to open " + path);
    }

    struct stat stat_buf {};
    if (fstat(fd_, &stat_buf) != 0) {
      FailModelLoader("failed to stat " + path);
    }
    if (stat_buf.st_size <= 0) {
      FailModelLoader("empty safetensors file " + path);
    }

    size_ = static_cast<size_t>(stat_buf.st_size);
    mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      FailModelLoader("failed to mmap " + path);
    }
  }

  ~ScopedMmap() {
    if (mapped_ != MAP_FAILED) {
      munmap(mapped_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  const uint8_t* data() const {
    return static_cast<const uint8_t*>(mapped_);
  }

  size_t size() const {
    return size_;
  }

 private:
  std::string path_;
  void* mapped_ = MAP_FAILED;
  size_t size_ = 0;
  int fd_ = -1;
};

uint64_t ReadLittleEndianU64(const uint8_t* data) {
  uint64_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

torch::ScalarType ParseSafetensorsDtype(const std::string_view dtype) {
  if (dtype == "BOOL") {
    return torch::kBool;
  }
  if (dtype == "U8") {
    return torch::kUInt8;
  }
  if (dtype == "I8") {
    return torch::kInt8;
  }
  if (dtype == "I16") {
    return torch::kInt16;
  }
  if (dtype == "I32") {
    return torch::kInt32;
  }
  if (dtype == "I64") {
    return torch::kInt64;
  }
  if (dtype == "F16") {
    return torch::kFloat16;
  }
  if (dtype == "BF16") {
    return torch::kBFloat16;
  }
  if (dtype == "F32") {
    return torch::kFloat32;
  }
  if (dtype == "F64") {
    return torch::kFloat64;
  }
  if (dtype == "F8_E4M3") {
    return torch::kFloat8_e4m3fn;
  }
  if (dtype == "F8_E5M2") {
    return torch::kFloat8_e5m2;
  }
  FailModelLoader("unsupported safetensors dtype " + std::string(dtype));
}

uint64_t ElementSizeOf(const torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kBool:
    case torch::kUInt8:
    case torch::kInt8:
    case torch::kFloat8_e4m3fn:
    case torch::kFloat8_e5m2:
      return 1;
    case torch::kInt16:
    case torch::kFloat16:
    case torch::kBFloat16:
      return 2;
    case torch::kInt32:
    case torch::kFloat32:
      return 4;
    case torch::kInt64:
    case torch::kFloat64:
      return 8;
    default:
      FailModelLoader("unsupported torch dtype");
  }
}

std::vector<int64_t> ParseShape(const rapidjson::Value& shape_value) {
  if (!shape_value.IsArray()) {
    FailModelLoader("shape must be an array");
  }

  std::vector<int64_t> shape;
  shape.reserve(shape_value.Size());
  for (const auto& dim : shape_value.GetArray()) {
    if (!dim.IsInt64()) {
      FailModelLoader("shape element must be int64");
    }
    shape.push_back(dim.GetInt64());
  }
  return shape;
}

uint64_t NumelOf(const std::vector<int64_t>& shape) {
  uint64_t numel = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      FailModelLoader("negative tensor dimension");
    }
    if (dim == 0) {
      return 0;
    }
    if (numel > std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(dim)) {
      FailModelLoader("tensor numel overflow");
    }
    numel *= static_cast<uint64_t>(dim);
  }
  return numel;
}

void AppendSafetensorsFileLikeXllm(
    const std::string& file_path,
    const torch::Device& device,
    LoadedModelWeights* loaded) {
  ScopedMmap mapped_file(file_path);
  const uint8_t* file_data = mapped_file.data();
  const size_t file_size = mapped_file.size();
  if (file_size < sizeof(uint64_t)) {
    FailModelLoader("invalid safetensors file " + file_path);
  }

  const uint64_t header_size = ReadLittleEndianU64(file_data);
  if (header_size > file_size - sizeof(uint64_t)) {
    FailModelLoader("invalid safetensors header size for " + file_path);
  }

  const char* header_data =
      reinterpret_cast<const char*>(file_data + sizeof(uint64_t));
  rapidjson::Document header;
  header.Parse(header_data, static_cast<size_t>(header_size));
  if (header.HasParseError() || !header.IsObject()) {
    FailModelLoader("failed to parse safetensors header for " + file_path);
  }

  const uint8_t* tensor_base =
      file_data + sizeof(uint64_t) + static_cast<size_t>(header_size);
  const uint64_t tensor_bytes =
      static_cast<uint64_t>(file_size) - sizeof(uint64_t) - header_size;
  torch::NoGradGuard no_grad;

  for (auto it = header.MemberBegin(); it != header.MemberEnd(); ++it) {
    const std::string tensor_name = it->name.GetString();
    if (tensor_name == "__metadata__") {
      continue;
    }

    const rapidjson::Value& spec = it->value;
    if (!spec.IsObject()) {
      FailModelLoader("tensor spec must be an object for " + tensor_name);
    }

    const auto dtype_it = spec.FindMember("dtype");
    const auto shape_it = spec.FindMember("shape");
    const auto offsets_it = spec.FindMember("data_offsets");
    if (dtype_it == spec.MemberEnd() || !dtype_it->value.IsString() ||
        shape_it == spec.MemberEnd() || offsets_it == spec.MemberEnd() ||
        !offsets_it->value.IsArray() || offsets_it->value.Size() != 2) {
      FailModelLoader("invalid tensor spec for " + tensor_name);
    }

    const torch::ScalarType dtype =
        ParseSafetensorsDtype(dtype_it->value.GetString());
    const std::vector<int64_t> shape = ParseShape(shape_it->value);
    const rapidjson::Value& offsets = offsets_it->value;
    if (!offsets[0].IsUint64() || !offsets[1].IsUint64()) {
      FailModelLoader("invalid tensor offsets for " + tensor_name);
    }

    const uint64_t start = offsets[0].GetUint64();
    const uint64_t end = offsets[1].GetUint64();
    if (start > end || end > tensor_bytes) {
      FailModelLoader("out-of-range tensor offsets for " + tensor_name);
    }

    const uint64_t expected_nbytes = NumelOf(shape) * ElementSizeOf(dtype);
    if (expected_nbytes != end - start) {
      FailModelLoader("tensor byte size mismatch for " + tensor_name);
    }

    auto cpu_tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(tensor_base + start)),
        shape,
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    auto device_tensor =
        torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
    device_tensor.copy_(cpu_tensor);

    loaded->total_bytes += expected_nbytes;
    loaded->tensors.push_back(
        LoadedModelTensor{tensor_name, std::move(device_tensor), expected_nbytes});
  }
}

std::vector<std::string> CollectSafetensorsFiles(const std::string& model_dir) {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.path().extension() == ".safetensors") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    FailModelLoader("no safetensors files found in " + model_dir);
  }
  return files;
}

}  // namespace

LoadedModelWeights LoadModelLikeXllmOnCuda(
    const std::string& model_dir,
    const torch::Device& device) {
  LoadedModelWeights loaded;
  loaded.model_dir = model_dir;
  const std::vector<std::string> files = CollectSafetensorsFiles(model_dir);
  for (const auto& file : files) {
    AppendSafetensorsFileLikeXllm(file, device, &loaded);
  }
  return loaded;
}
