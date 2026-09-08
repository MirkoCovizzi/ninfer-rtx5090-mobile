#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/kvarn.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

[[nodiscard]] std::size_t kvarn_attention_workspace_capacity_bytes(
    std::int32_t query_heads, CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width);

// Appends and attends in the orthonormal KVarN frame. Current-chunk K/V stay unquantized throughout
// attention. Completed non-sink groups are encoded only after their tokens are committed;
// provisional groups remain in the tail until acceptance. Rejected suffixes are overwritable.
// Q/K/V are disposable. For single-row final-query prefill, Q may contain only the last query
// while K/V and positions contain the entire appended chunk.
void kvarn_attention(Tensor query, Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                     KvarnPagedBatchLayerView cache, bool provisional,
                     CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                     Tensor& output, cudaStream_t stream);

void kvarn_attention_cached(Tensor query, const Tensor& positions, const Tensor& kv_table_rows,
                            float scale, const KvarnPagedBatchLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& output, cudaStream_t stream);

void kvarn_kv_append(Tensor key, Tensor value, const Tensor& positions, const Tensor& valid_columns,
                     const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache, bool provisional,
                     cudaStream_t stream);

// accepted_columns is an I32 prefix count per batch row. Completed non-sink groups in that prefix
// are encoded from their unquantized tails before their markers are retired. Sinks remain lossless.
void kvarn_commit_pages(const Tensor& positions, const Tensor& accepted_columns,
                        const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache,
                        cudaStream_t stream);

// Settles completed live groups through the final committed frontier and re-establishes its
// writable tail. A historical partial group is decoded; an already-live partial tail is preserved.
// Runtime calls this after output publication has selected the actual prefix, not at licensing.
// Batches one to sixteen disjoint layer views at the same frontier without device scratch.
void kvarn_restore_tail(std::int32_t frontier, std::span<const KvarnPagedLayerView> layers,
                        cudaStream_t stream);

} // namespace ninfer::ops
