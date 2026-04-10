#pragma once

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#include <cuda.h>

#else
#error "USE_CUDA is not set"
#endif
