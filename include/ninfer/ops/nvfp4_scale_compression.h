#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops {

/**
 * Op: lossless NVFP4 scale-plane compression
 *
 * Reads all N*K/16 E4M3FN scale bytes in block_scale_k16_m128x4_v1 order. N must be a
 * positive multiple of 128 and K a positive multiple of 128. Each 128-row by eight-K16-group
 * tile is independently encoded as 15 scale bytes in frequency order (ties by byte value),
 * one zero padding byte, two 4-bit indices per byte, 16 little-endian escape-rank prefixes,
 * then literal bytes for indices equal to 15 and zero padding to a 16-byte record boundary.
 * Indices and literals traverse the tile's 1,024 contiguous bytes in native swizzled order;
 * each prefix counts escapes preceding a group of 64 bytes. The decoded plane is bit-for-bit equal
 * to the input, including zero and all other stored byte patterns. Codes and divisor are not
 * transformed. The result owns its bytes and tile offsets; it has no device state.
 */
struct Nvfp4CompressedScales {
    std::vector<std::uint8_t> payload;
    std::vector<std::uint32_t> offsets;
    std::int32_t tiles_per_row = 0;
};

[[nodiscard]] std::size_t nvfp4_scale_plane_bytes(std::int32_t rows, std::int32_t columns);
[[nodiscard]] Nvfp4CompressedScales compress_nvfp4_scale_plane(std::span<const std::byte> scales,
                                                               std::int32_t rows,
                                                               std::int32_t columns);

// Op: exact inverse of compress_nvfp4_scale_plane. The two Device pointers address the
// uploaded payload and its per-tile uint32 offsets; destination is a caller-owned N*K/16-byte
// Device plane in the artifact's swizzled scale order. Payload and destination are 16-byte
// aligned, offsets are uint32-aligned, and pointers may not alias. All bytes are
// written on stream, including under CUDA Graph capture; no persistent state or scratch.
void expand_nvfp4_scale_plane(const std::uint8_t* payload, const std::uint32_t* offsets,
                              std::int32_t rows, std::int32_t columns, std::uint8_t* destination,
                              cudaStream_t stream);

} // namespace ninfer::ops
