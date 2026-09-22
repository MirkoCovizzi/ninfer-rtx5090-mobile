#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Compact decode and speculative verification keep the same per-column arithmetic through B8/W16.
inline constexpr std::int32_t kBf16AttnInputSmallTMinTokens   = 1;
inline constexpr std::int32_t kBf16AttnInputSmallTMaxTokens   = 32;
inline constexpr std::int32_t kBf16AttnInputSmallTDispatchEnd = 8 * 16;

void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream);

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
