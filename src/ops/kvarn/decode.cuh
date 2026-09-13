#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/kvarn.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

namespace ninfer::ops::kvarn {

// Rotated current-chunk values remain unquantized until its attention has completed.
struct CurrentKV {
    const __nv_bfloat16* key      = nullptr;
    const __nv_bfloat16* value    = nullptr;
    const std::int32_t* positions = nullptr;
    std::int32_t width            = 0;
};

void decode_attention(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                      const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                      CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                      Tensor& output, cudaStream_t stream, CurrentKV current = {});

} // namespace ninfer::ops::kvarn
