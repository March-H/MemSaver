#ifndef MEMSAVER_CUSTOM_TORCH_ALLOCATOR_H_
#define MEMSAVER_CUSTOM_TORCH_ALLOCATOR_H_

#include <memory>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include "allocator/allocator.h"

inline void InstallCustomTorchAllocator() {
  static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
      allocator = []() {
        auto current_allocator =
            torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
                [](size_t size, int device, cudaStream_t stream) -> void* {
                  return CachingAllocator::instance().malloc(
                      size,
                      device,
                      stream);
                },
                [](void* ptr, size_t size, int device, cudaStream_t stream) {
                  (void)size;
                  (void)device;
                  (void)stream;
                  CachingAllocator::instance().free(ptr);
                });
        auto pluggable = std::dynamic_pointer_cast<
            torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
            current_allocator);
        pluggable->set_record_stream_fn(
            [](void* ptr, cudaStream_t stream) {
              CachingAllocator::instance().record_stream(ptr, stream);
            });
        torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(
            current_allocator);
        return current_allocator;
      }();
  (void)allocator;
}

#endif
