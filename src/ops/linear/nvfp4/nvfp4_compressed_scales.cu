#include "ops/linear/nvfp4/nvfp4_compressed_scales.h"
#include "ops/linear/nvfp4/nvfp4_compressed_scales.cuh"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ unsigned decode_palette(unsigned indices, uint4 palette) {
    // Four independent nibble lookups. Palette entry 15 is zero so escaped bytes
    // can be inserted with OR after decoding the ordinary palette indices.
    const unsigned selector = indices & 0x7777U;
    const unsigned lower    = __byte_perm(palette.x, palette.y, selector);
    const unsigned upper    = __byte_perm(palette.z, palette.w, selector);
    const unsigned mask     = __byte_perm(0xffffffffU, 0U, (indices >> 1) & 0x4444U);
    return (lower & mask) | (upper & ~mask);
}

__device__ __forceinline__ unsigned insert_literals(unsigned word, unsigned mask,
                                                    const std::uint8_t* literals, unsigned& rank) {
    while (mask) {
        const int bit = __ffs(mask) - 1;
        word |= static_cast<unsigned>(literals[rank++]) << (2 * bit);
        mask &= mask - 1;
    }
    return word;
}

// Two warps expand a tile, with each lane owning 16 consecutive native bytes.
__global__ void expand_kernel(const std::uint8_t* payload, const std::uint32_t* offsets,
                              std::uint8_t* destination, int chunks) {
    const int lane   = threadIdx.x & 31;
    const int thread = threadIdx.x & 63;
    const int prefix = thread / 4;
    const int chunk  = blockIdx.x * 2 + threadIdx.x / 64;
    if (chunk >= chunks) { return; }
    const auto* record        = payload + offsets[chunk];
    const auto palette        = *reinterpret_cast<const uint4*>(record);
    const auto packed         = reinterpret_cast<const uint2*>(record + kIndexOffset)[thread];
    const unsigned indices[2] = {packed.x, packed.y};
    unsigned escape_bits[2];
    unsigned escapes = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const auto v   = indices[j];
        escape_bits[j] = v & (v >> 1) & (v >> 2) & (v >> 3) & 0x11111111U;
        escapes += __popc(escape_bits[j]);
    }
    unsigned rank = 0;
    if (__any_sync(0xffffffffU, escapes)) {
        // Four adjacent lanes share a stored 64-byte escape prefix.
        unsigned cumulative = escapes;
#pragma unroll
        for (int delta = 1; delta < 4; delta *= 2) {
            const auto previous = __shfl_up_sync(0xffffffffU, cumulative, delta, 4);
            if ((lane & 3) >= delta) { cumulative += previous; }
        }
        rank = reinterpret_cast<const std::uint16_t*>(record + kPrefixOffset)[prefix] + cumulative -
               escapes;
    }
    const auto* literals = record + kEscapeOffset;
    std::uint32_t word[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        word[2 * j]     = decode_palette(indices[j], palette);
        word[2 * j + 1] = decode_palette(indices[j] >> 16, palette);
        word[2 * j]     = insert_literals(word[2 * j], escape_bits[j] & 0xffffU, literals, rank);
        word[2 * j + 1] = insert_literals(word[2 * j + 1], escape_bits[j] >> 16, literals, rank);
    }
    const auto base = static_cast<std::size_t>(chunk) * kScaleCount + thread * 16;
    *reinterpret_cast<uint4*>(destination + base) = make_uint4(word[0], word[1], word[2], word[3]);
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
    if (!payload || !offsets || !destination || reinterpret_cast<std::uintptr_t>(payload) % 16 ||
        reinterpret_cast<std::uintptr_t>(offsets) % alignof(std::uint32_t) ||
        reinterpret_cast<std::uintptr_t>(destination) % 16) {
        throw std::invalid_argument("invalid NVFP4 scale expansion pointers");
    }
    const int tiles_per_row = columns / (detail::kScaleGroups * 16);
    const int chunks        = (rows / detail::kScaleRows) * tiles_per_row;
    detail::expand_kernel<<<(chunks + 1) / 2, 128, 0, stream>>>(payload, offsets, destination,
                                                                chunks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
