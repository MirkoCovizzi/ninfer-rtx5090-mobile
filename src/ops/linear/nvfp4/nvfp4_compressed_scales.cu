#include "ops/linear/nvfp4/nvfp4_compressed_scales.h"
#include "ops/linear/nvfp4/nvfp4_compressed_scales.cuh"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Threads follow the existing 128x4 swizzle of the native NVFP4 scale plane.
__global__ void expand_kernel(const std::uint8_t* payload, const std::uint32_t* offsets,
                              int tiles_per_row, std::uint8_t* destination, int columns,
                              int chunks) {
    const int thread         = threadIdx.x;
    const int lane           = thread & 31;
    const int row            = (thread & 3) * 32 + (thread >> 2);
    const int prefix         = row / kPrefixRows;
    const unsigned peers     = __match_any_sync(0xffffffffU, prefix);
    const unsigned preceding = peers & (lane == 0 ? 0U : 0xffffffffU >> (32 - lane));
    for (int step = 0; step < 4; ++step) {
        const int chunk = blockIdx.x * 4 + step;
        if (chunk >= chunks) { break; }
        const auto* record     = payload + offsets[chunk];
        const unsigned palette = lane < kPaletteSize ? record[lane] : 0U;
        const auto* indices    = record + kPaletteSize + row * 4;
        std::uint8_t values[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const auto packed = indices[i];
            values[2 * i]     = packed & 15;
            values[2 * i + 1] = packed >> 4;
        }
        unsigned rank = record[kPrefixOffset + prefix * 2] |
                        (static_cast<unsigned>(record[kPrefixOffset + prefix * 2 + 1]) << 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            rank += __popc(__ballot_sync(0xffffffffU, values[i] == 15) & preceding);
        }
        const auto* literals  = record + kEscapeOffset;
        std::uint32_t word[2] = {};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const auto index  = values[i];
            const auto lookup = __shfl_sync(0xffffffffU, palette, index);
            const auto scale  = index == 15 ? literals[rank++] : lookup;
            word[i / 4] |= static_cast<std::uint32_t>(scale) << (8 * (i & 3));
        }
        const int row_tile = chunk / tiles_per_row;
        const int tile     = chunk % tiles_per_row;
        const auto base = (static_cast<std::size_t>(row_tile) * (columns / 64) + tile * 2) * 512 +
                          static_cast<std::size_t>(thread) * 4;
        *reinterpret_cast<std::uint32_t*>(destination + base)       = word[0];
        *reinterpret_cast<std::uint32_t*>(destination + base + 512) = word[1];
    }
}

} // namespace

void expand_nvfp4_scales(const Weight& weight, std::uint8_t* destination, cudaStream_t stream) {
    if (weight.compressed_scale_tiles_per_row != weight.k / (kScaleGroups * 16)) {
        throw std::invalid_argument("invalid compressed NVFP4 scale operands");
    }
    expand_nvfp4_scale_plane(static_cast<const std::uint8_t*>(weight.compressed_scales),
                             weight.compressed_scale_offsets, weight.n, weight.k, destination,
                             stream);
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

void expand_nvfp4_scale_plane(const std::uint8_t* payload, const std::uint32_t* offsets,
                              std::int32_t rows, std::int32_t columns, std::uint8_t* destination,
                              cudaStream_t stream) {
    (void)nvfp4_scale_plane_bytes(rows, columns);
    if (!payload || !offsets || !destination) {
        throw std::invalid_argument("invalid NVFP4 scale expansion pointers");
    }
    const int tiles_per_row = columns / (detail::kScaleGroups * 16);
    const int chunks        = (rows / detail::kScaleRows) * tiles_per_row;
    detail::expand_kernel<<<(chunks + 3) / 4, 128, 0, stream>>>(payload, offsets, tiles_per_row,
                                                                destination, columns, chunks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
