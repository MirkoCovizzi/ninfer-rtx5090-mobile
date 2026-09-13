#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops {

// Huawei's released kvarn_k4v2_g128 profile at D256.
inline constexpr std::int32_t kKvarnHeadDim    = 256;
inline constexpr std::int32_t kKvarnGroup      = 128;
inline constexpr std::int32_t kKvarnIterations = 8;
inline constexpr std::int32_t kKvarnSinkPages  = 1;
inline constexpr std::int32_t kKvarnTailSlots  = 3;

inline constexpr std::int32_t kKvarnKPackedOffset       = 0;
inline constexpr std::int32_t kKvarnKScaleOffset        = 16384;
inline constexpr std::int32_t kKvarnKZeroOffset         = 16896;
inline constexpr std::int32_t kKvarnKTokenScaleOffset   = 17408;
inline constexpr std::int32_t kKvarnVPackedOffset       = 17664;
inline constexpr std::int32_t kKvarnVChannelScaleOffset = 25856;
inline constexpr std::int32_t kKvarnVTokenScaleOffset   = 26368;
inline constexpr std::int32_t kKvarnVTokenZeroOffset    = 26624;
inline constexpr std::int32_t kKvarnRecordPayloadBytes  = 26880;
inline constexpr std::int32_t kKvarnRecordBytes         = 26880;
static_assert(kKvarnRecordBytes >= kKvarnRecordPayloadBytes && kKvarnRecordBytes % 256 == 0);

struct KvarnTileStorage {
    Tensor k_codes;          // U8   [G/2,D,N]
    Tensor k_scales;         // FP16 [D,N]
    Tensor k_zeros;          // FP16 [D,N]
    Tensor k_token_scales;   // FP16 [G,N]
    Tensor v_codes;          // U8   [D/4,G,N]
    Tensor v_channel_scales; // FP16 [D,N]
    Tensor v_token_scales;   // FP16 [G,N]
    Tensor v_token_zeros;    // FP16 [G,N]
};

struct KvarnTailStateView {
    Tensor k;             // BF16 [D,G,Hkv*tail_slots]
    Tensor v;             // BF16 [D,G,Hkv*tail_slots]
    Tensor logical_pages; // I32 [tail_slots]
    std::int32_t num_kv_heads = 0;
};

struct KvarnPagedLayerView {
    Tensor records;            // U8 [record_bytes / P,P,Hkv,Nphysical]
    Tensor tail_k;             // BF16 [D,P,Hkv*tail_slots]
    Tensor tail_v;             // BF16 [D,P,Hkv*tail_slots]
    Tensor tail_logical_pages; // I32 [tail_slots]
    Tensor block_table;        // I32 [Nlogical]
    std::int32_t num_kv_heads = 0;
};

struct KvarnPagedBatchLayerView {
    Tensor records;            // U8 [record_bytes / P,P,Hkv,Nphysical]
    Tensor tail_k;             // BF16 [D,P,Hkv*tail_slots,C]
    Tensor tail_v;             // BF16 [D,P,Hkv*tail_slots,C]
    Tensor tail_logical_pages; // I32 [tail_slots,C]
    Tensor block_tables;       // I32 [Nlogical,C]
    std::int32_t num_kv_heads = 0;
};

// Inputs are Hadamard-rotated contiguous BF16 [D,G,N] tiles. The represented decode is:
// K[d,t] = (code[d,t] * k_scales[d] + k_zeros[d]) * k_token_scales[t]
// V[d,t] = (code[d,t] * v_token_scales[t] + v_token_zeros[t]) * v_channel_scales[d]
void kvarn_store(const Tensor& rotated_k, const Tensor& rotated_v, KvarnTileStorage storage,
                 cudaStream_t stream);
void kvarn_dequant(const KvarnTileStorage& storage, Tensor& rotated_k, Tensor& rotated_v,
                   cudaStream_t stream);

// Orthonormal Sylvester-Hadamard transform over contiguous BF16 D256 vectors.
void kvarn_hadamard(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
