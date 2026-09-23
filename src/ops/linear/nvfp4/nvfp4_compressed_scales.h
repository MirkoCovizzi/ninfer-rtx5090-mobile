#pragma once

#include "core/weight.h"
#include "ninfer/ops/nvfp4_scale_compression.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void expand_nvfp4_scales(const Weight& weight, std::uint8_t* destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
