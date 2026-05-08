#include "memsaver/entrypoint.h"

#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "allocator/allocator.h"
#include "allocator/mem_pool.h"
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

  void set_block_pool_type(int value) {
    block_pool_type_ = value;
  }

  int block_pool_type() const {
    return block_pool_type_;
  }

 private:
  bool is_interesting_region_ = false;
  bool enable_cpu_backup_ = false;
  int block_pool_type_ = 0;
  AllocationKind allocation_mode_ = AllocationKind::REGULAR;
  CUdevice device_ = 0;
  MemSaver* owner_ = nullptr;
};

thread_local ThreadLocalConfig thread_local_config;

struct MemSaver::RegionCacheKey {
  std::string tag;
  bool enable_cpu_backup = false;
  AllocationKind mode = AllocationKind::REGULAR;
  int block_pool_type = 0;

  bool operator==(const RegionCacheKey& other) const {
    return tag == other.tag &&
           enable_cpu_backup == other.enable_cpu_backup &&
           mode == other.mode &&
           block_pool_type == other.block_pool_type;
  }
};

cudaError_t EnsureValidMode(const AllocationKind mode) {
  const bool valid = mode == AllocationKind::REGULAR ||
                     mode == AllocationKind::ARENA ||
                     mode == AllocationKind::ARENA_VIRTUAL;
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

void* memsaver_allocator_malloc(size_t size, int device, cudaStream_t stream) {
  (void)device;
  (void)stream;
  if (!thread_local_config.is_interesting_region()) {
    LOGE("memsaver_allocator_malloc requires an interesting region");
    return nullptr;
  }
  void* ptr = nullptr;
  if (memsaver_malloc(&ptr, size) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void memsaver_allocator_free(void* ptr, size_t size, int device, cudaStream_t stream) {
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
  CachingAllocator::instance().empty_cache();
  return cudaSuccess;
}

const char* memsaver_current_region_tag() {
  return thread_local_config.current_tag_.c_str();
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
    value ^= std::hash<bool>{}(key.enable_cpu_backup) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<int>{}(static_cast<int>(key.mode)) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<int>{}(key.block_pool_type) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    return value;
  }
};

struct MemSaver::CachedPool {
  std::size_t pool_id = 0;
  std::shared_ptr<MemPool> pool;
  std::unordered_set<CUdevice> devices;
};

struct MemSaver::Impl {
  std::mutex mutex;
  std::size_t next_pool_id = 1;
  std::unordered_map<RegionCacheKey, std::shared_ptr<CachedPool>, RegionCacheKeyHash>
      cached_pools;
};

MemSaver::MemSaver() : impl_(std::make_unique<Impl>()) {}

MemSaver::~MemSaver() = default;

std::shared_ptr<MemSaver::CachedPool> MemSaver::get_or_create_cached_pool(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode,
    int block_pool_type) {
  const RegionCacheKey key{
      tag,
      NormalizeEnableCpuBackup(enable_cpu_backup, mode),
      mode,
      block_pool_type};
  std::lock_guard<std::mutex> guard(impl_->mutex);
  const auto it = impl_->cached_pools.find(key);
  if (it != impl_->cached_pools.end()) {
    it->second->pool = createMemPool(
        it->second->pool_id,
        memsaver_malloc,
        memsaver_free,
        block_pool_type);
    return it->second;
  }

  auto cached_pool = std::make_shared<CachedPool>();
  cached_pool->pool_id = impl_->next_pool_id++;
  cached_pool->pool = createMemPool(
      cached_pool->pool_id,
      memsaver_malloc,
      memsaver_free,
      block_pool_type);
  impl_->cached_pools.emplace(key, cached_pool);
  return cached_pool;
}

std::shared_ptr<MemSaver::CachedPool> MemSaver::get_cached_pool(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode,
    int block_pool_type) {
  const RegionCacheKey key{
      tag,
      NormalizeEnableCpuBackup(enable_cpu_backup, mode),
      mode,
      block_pool_type};
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
    AllocationKind mode,
    int block_pool_type) {
  const cudaError_t mode_status = EnsureValidMode(mode);
  if (mode_status != cudaSuccess) {
    return mode_status;
  }

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
      get_or_create_cached_pool(tag, enable_cpu_backup, mode, block_pool_type);
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
  thread_local_config.set_block_pool_type(block_pool_type);

  beginAllocateToPool(
      static_cast<int>(current_device),
      cached_pool->pool_id,
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
      thread_local_config.get_allocation_mode(),
      thread_local_config.block_pool_type());
  RETURN_IF_FALSE(
      cached_pool != nullptr,
      cudaErrorInvalidValue,
      "leave_region: cached pool not found");

  endAllocateToPool(
      static_cast<int>(thread_local_config.get_device()),
      cached_pool->pool_id);
  releasePool(
      static_cast<int>(thread_local_config.get_device()),
      cached_pool->pool_id);
  thread_local_config = ThreadLocalConfig{};
  return cudaSuccess;
}

cudaError_t MemSaver::evict_region_pool_from_cache(
    const std::string& tag,
    const bool enable_cpu_backup,
    const AllocationKind mode,
    int block_pool_type) {
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

  std::size_t pool_id = 0;
  std::unordered_set<CUdevice> devices;
  {
    const RegionCacheKey key{
        tag,
        normalized_enable_cpu_backup,
        mode,
        block_pool_type};
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const auto it = impl_->cached_pools.find(key);
    if (it != impl_->cached_pools.end()) {
      pool_id = it->second->pool_id;
      devices = it->second->devices;
      impl_->cached_pools.erase(it);
    }
  }

  for (const CUdevice device : devices) {
    releasePool(static_cast<int>(device), pool_id);
  }

  return ContextImpl::instance().ReleaseAllocations(
      tag,
      normalized_enable_cpu_backup,
      mode);
}
