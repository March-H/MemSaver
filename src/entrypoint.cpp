#include "memsaver/entrypoint.h"

#include <functional>
#include <mutex>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include "internal/context_impl.h"
#include "internal/utils.h"

class ThreadLocalConfig {
public:
    std::string current_tag_ = "";

    bool is_interesting_region() const {
        return is_interesting_region_;
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() const {
        return enable_cpu_backup_;
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

    void set_allocation_mode(AllocationKind mode) {
        allocation_mode_ = mode;
    }

    AllocationKind get_allocation_mode() const {
        return allocation_mode_;
    }

    void set_device(CUdevice device) {
        device_ = device;
    }

    CUdevice get_device() const {
        return device_;
    }

    void set_owner(MemSaver* owner) {
        owner_ = owner;
    }

    MemSaver* get_owner() const {
        return owner_;
    }

private:
    bool is_interesting_region_ = false;
    bool enable_cpu_backup_ = false;
    AllocationKind allocation_mode_ = AllocationKind::REGULAR;
    CUdevice device_ = 0;
    MemSaver* owner_ = nullptr;
};

thread_local ThreadLocalConfig thread_local_config;

struct MemSaver::RegionCacheKey {
  std::string tag;
  bool enable_cpu_backup = false;
  AllocationKind mode = AllocationKind::REGULAR;

  bool operator==(const RegionCacheKey& other) const {
    return tag == other.tag &&
           enable_cpu_backup == other.enable_cpu_backup &&
           mode == other.mode;
  }
};

cudaError_t EnsureValidMode(const AllocationKind mode) {
  const bool valid = mode == AllocationKind::REGULAR ||
                     mode == AllocationKind::ARENA;
  RETURN_IF_FALSE(valid, cudaErrorInvalidValue,
                  "EnsureValidMode: invalid allocation mode");
  return cudaSuccess;
}

bool NormalizeEnableCpuBackup(
    const bool enable_cpu_backup,
    const AllocationKind mode) {
  return mode == AllocationKind::REGULAR && enable_cpu_backup;
}

cudaError_t BuildRuntimeConfig(RuntimeConfig* runtime_config) {
  RETURN_IF_FALSE(runtime_config != nullptr, cudaErrorInvalidValue,
                  "BuildRuntimeConfig: runtime_config should not be null");
  runtime_config->interesting_region = thread_local_config.is_interesting_region();
  runtime_config->enable_cpu_backup = thread_local_config.enable_cpu_backup();
  runtime_config->tag = thread_local_config.current_tag_;
  runtime_config->allocation_mode = thread_local_config.get_allocation_mode();
  RETURN_IF_CUDA_ERROR(EnsureValidMode(runtime_config->allocation_mode));
  return cudaSuccess;
}

cudaError_t memsaver_malloc(void** ptr, size_t size) {
  RETURN_IF_FALSE(ptr != nullptr, cudaErrorInvalidValue,
                  "memsaver_malloc: ptr should not be null");
  if (!thread_local_config.is_interesting_region()) {
    LOGE("it only work with custom mem_pool allocator, it should not happend");
    return cudaErrorInvalidValue;
  }

  RuntimeConfig runtime_config;
  RETURN_IF_CUDA_ERROR(BuildRuntimeConfig(&runtime_config));

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().Malloc(ptr, device, size, runtime_config);
}

cudaError_t memsaver_free(void* ptr) {
  return ContextImpl::instance().Free(ptr);
}

void* memsaver_torch_malloc(size_t size, int device, cudaStream_t stream) {
  (void)device;
  (void)stream;
  if (!thread_local_config.is_interesting_region()) {
    LOGE("memsaver_torch_malloc requires an interesting region");
    return nullptr;
  }
  void* ptr = nullptr;
  if (memsaver_malloc(&ptr, size) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void memsaver_torch_free(void* ptr, size_t size, int device, cudaStream_t stream) {
  (void)size;
  (void)device;
  (void)stream;
  memsaver_free(ptr);
}

cudaError_t memsaver_pause(const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return ContextImpl::instance().Pause(tag);
}

cudaError_t memsaver_resume(const char* tag_or_null) {
  const std::string tag = (tag_or_null == nullptr) ? "" : tag_or_null;
  return ContextImpl::instance().Resume(tag);
}

cudaError_t memsaver_empty_cache() {
  c10::cuda::CUDACachingAllocator::emptyCache();
  return cudaSuccess;
}

cudaError_t memsaver_activate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                  "memsaver_activate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().ActivateArenaOffsets(
      tag, device, offsets, num_offsets, size_bytes);
}

cudaError_t memsaver_deactivate_arena_offsets(
    const char* tag,
    const uint64_t* offsets,
    const uint64_t num_offsets,
    const uint64_t size_bytes) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                  "memsaver_deactivate_arena_offsets: tag should not be null");

  CUdevice device = 0;
  RETURN_IF_CUDA_ERROR(GetCurrentCudaDevice(&device));

  return ContextImpl::instance().DeactivateArenaOffsets(
      tag, device, offsets, num_offsets, size_bytes);
}

cudaError_t memsaver_get_metadata_count_by_tag(
    const char* tag,
    uint64_t* out_count) {
  RETURN_IF_FALSE(tag != nullptr, cudaErrorInvalidValue,
                  "memsaver_get_metadata_count_by_tag: tag should not be null");

  return ContextImpl::instance().GetMetadataCountByTag(tag, out_count);
}

cudaError_t memsaver_get_cpu_backup_pointer(
    const uint8_t* gpu_ptr,
    const uint64_t size,
    uint8_t** out_cpu_ptr) {
  return ContextImpl::instance().GetCpuBackupPointer(gpu_ptr, size, out_cpu_ptr);
}


struct MemSaver::RegionCacheKeyHash {
  size_t operator()(const RegionCacheKey& key) const {
    size_t value = std::hash<std::string>{}(key.tag);
    value ^= std::hash<bool>{}(key.enable_cpu_backup) + 0x9e3779b9 + (value << 6) + (value >> 2);
    value ^= std::hash<int>{}(static_cast<int>(key.mode)) + 0x9e3779b9 + (value << 6) + (value >> 2);
    return value;
  }
};

struct MemSaver::CachedPool {
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator;
  std::unique_ptr<c10::cuda::MemPool> pool;
  std::unordered_set<CUdevice> devices;
};

struct MemSaver::Impl {
  std::mutex mutex;
  std::unordered_map<RegionCacheKey, std::shared_ptr<CachedPool>, RegionCacheKeyHash>
      cached_pools;
};

MemSaver::MemSaver() : impl_(std::make_unique<Impl>()) {}

MemSaver::~MemSaver() = default;

std::shared_ptr<MemSaver::CachedPool> MemSaver::get_or_create_cached_pool(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode) {
  const RegionCacheKey key{
      tag,
      NormalizeEnableCpuBackup(enable_cpu_backup, mode),
      mode};
  std::lock_guard<std::mutex> guard(impl_->mutex);
  const auto it = impl_->cached_pools.find(key);
  if (it != impl_->cached_pools.end()) {
    return it->second;
  }

  auto cached_pool = std::make_shared<CachedPool>();
  cached_pool->allocator =
      torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
          memsaver_torch_malloc,
          memsaver_torch_free);
  cached_pool->pool = std::make_unique<c10::cuda::MemPool>(
      cached_pool->allocator.get());
  impl_->cached_pools.emplace(key, cached_pool);
  return cached_pool;
}

std::shared_ptr<MemSaver::CachedPool> MemSaver::get_cached_pool(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode) {
  const RegionCacheKey key{
      tag,
      NormalizeEnableCpuBackup(enable_cpu_backup, mode),
      mode};
  std::lock_guard<std::mutex> guard(impl_->mutex);
  const auto it = impl_->cached_pools.find(key);
  if (it == impl_->cached_pools.end()) {
    return nullptr;
  }
  return it->second;
}

cudaError_t MemSaver::enter_region(
    const std::string& tag,
    bool enable_cpu_backup,
    AllocationKind mode) {
  const cudaError_t mode_status = EnsureValidMode(mode);
  if (mode_status != cudaSuccess) {
    return mode_status;
  }

  // 禁止嵌套region
  RETURN_IF_FALSE(
      !thread_local_config.is_interesting_region(),
      cudaErrorInvalidValue,
      "enter_region: a region is already active on this thread");

  CUdevice current_device = 0;
  const cudaError_t device_status = GetCurrentCudaDevice(&current_device);
  if (device_status != cudaSuccess) {
    return device_status;
  }

  std::shared_ptr<CachedPool> cached_pool =
      get_or_create_cached_pool(tag, enable_cpu_backup, mode);
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    cached_pool->devices.insert(current_device);
  }

  thread_local_config.current_tag_ = tag;
  thread_local_config.set_interesting_region(true);
  thread_local_config.set_enable_cpu_backup(
      NormalizeEnableCpuBackup(enable_cpu_backup, mode));
  thread_local_config.set_allocation_mode(mode);
  thread_local_config.set_device(current_device);
  thread_local_config.set_owner(this);

  const c10::DeviceIndex device = static_cast<c10::DeviceIndex>(current_device);
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      device,
      cached_pool->pool->id(),
      [](cudaStream_t) { return true; });
  return cudaSuccess;
}

cudaError_t MemSaver::leave_region() {
  RETURN_IF_FALSE(
      thread_local_config.is_interesting_region() &&
          thread_local_config.get_owner() == this,
      cudaErrorInvalidValue,
      "leave_region: no active region for this MemSaver on this thread");

  std::shared_ptr<CachedPool> cached_pool = get_or_create_cached_pool(
      thread_local_config.current_tag_,
      thread_local_config.enable_cpu_backup(),
      thread_local_config.get_allocation_mode());
  RETURN_IF_FALSE(
      cached_pool != nullptr,
      cudaErrorInvalidValue,
      "leave_region: cached pool not found");

  const c10::DeviceIndex device =
      static_cast<c10::DeviceIndex>(thread_local_config.get_device());
  const c10::cuda::MempoolId_t mempool_id = cached_pool->pool->id();

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      device,
      mempool_id);
  c10::cuda::CUDACachingAllocator::releasePool(
      device,
      mempool_id);
  thread_local_config = ThreadLocalConfig{};
  return cudaSuccess;
}

// 由于torch的mem_pool的析构机制，当将pool内没有使用中的tensor并调用这个函数后，等价于触发这个pool的析构
cudaError_t MemSaver::evict_region_pool_from_cache(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode) {
  const cudaError_t mode_status = EnsureValidMode(mode);
  if (mode_status != cudaSuccess) {
    return mode_status;
  }

  RETURN_IF_FALSE(
      !thread_local_config.is_interesting_region(),
      cudaErrorInvalidValue,
      "evict_region_pool_from_cache: cannot evict a pool while a region is active on this thread");

  const bool normalized_enable_cpu_backup =
      NormalizeEnableCpuBackup(enable_cpu_backup, mode);

  {
    const RegionCacheKey key{tag, normalized_enable_cpu_backup, mode};
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const auto it = impl_->cached_pools.find(key);
    if (it != impl_->cached_pools.end()) {
      impl_->cached_pools.erase(it);
    }
  }

  return cudaSuccess;
}
