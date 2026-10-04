#include "ops/linear/nvfp4/nvfp4_compressed_scales.h"
#include "ops/linear/nvfp4/nvfp4_compressed_scales.cuh"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

std::size_t nvfp4_scale_plane_bytes(std::int32_t rows, std::int32_t columns) {
    if (rows <= 0 || columns <= 0 || rows % detail::kScaleRows ||
        columns % (detail::kScaleGroups * 16)) {
        throw std::invalid_argument("NVFP4 scale compression requires N%128=0, K%128=0");
    }
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) / 16;
}

Nvfp4CompressedScales compress_nvfp4_scale_plane(std::span<const std::byte> scales,
                                                 std::int32_t rows, std::int32_t columns) {
    if (scales.size() != nvfp4_scale_plane_bytes(rows, columns)) {
        throw std::invalid_argument("NVFP4 scale plane has an invalid size");
    }
    Nvfp4CompressedScales result;
    result.tiles_per_row = columns / (detail::kScaleGroups * 16);
    const std::size_t chunks =
        static_cast<std::size_t>(rows / detail::kScaleRows) * result.tiles_per_row;
    result.offsets.reserve(chunks);
    for (int row_tile = 0; row_tile < rows / detail::kScaleRows; ++row_tile) {
        for (int tile = 0; tile < result.tiles_per_row; ++tile) {
            if (result.payload.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("NVFP4 compressed scale offsets exceed uint32");
            }
            result.offsets.push_back(static_cast<std::uint32_t>(result.payload.size()));
            std::array<std::uint8_t, detail::kScaleCount> values{};
            std::array<std::uint16_t, 256> histogram{};
            const auto base = (static_cast<std::size_t>(row_tile) * result.tiles_per_row + tile) *
                              detail::kScaleCount;
            for (int i = 0; i < detail::kScaleCount; ++i) {
                const auto value = std::to_integer<std::uint8_t>(scales[base + i]);
                values[i]        = value;
                ++histogram[value];
            }
            std::array<std::uint16_t, 256> order{};
            for (int code = 0; code < 256; ++code) { order[code] = code; }
            std::sort(order.begin(), order.end(), [&](auto a, auto b) {
                return histogram[a] != histogram[b] ? histogram[a] > histogram[b] : a < b;
            });
            std::array<std::uint8_t, 256> palette{};
            palette.fill(15);
            for (int i = 0; i < detail::kPaletteSize; ++i) {
                const auto code = static_cast<std::uint8_t>(order[i]);
                palette[code]   = static_cast<std::uint8_t>(i);
                result.payload.push_back(code);
            }
            result.payload.push_back(0); // Align the packed index words.
            const auto indices = result.payload.size();
            result.payload.resize(indices + detail::kIndexBytes, 0);
            const auto prefixes = result.payload.size();
            result.payload.resize(prefixes + detail::kPrefixBytes, 0);
            std::size_t escapes = 0;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i % detail::kPrefixValues == 0) {
                    const auto prefix                         = i / detail::kPrefixValues;
                    result.payload[prefixes + 2 * prefix]     = escapes & 0xff;
                    result.payload[prefixes + 2 * prefix + 1] = escapes >> 8;
                }
                const auto index = palette[values[i]];
                result.payload[indices + i / 2] |=
                    (i & 1) ? static_cast<std::uint8_t>(index << 4) : index;
                if (index == 15) {
                    result.payload.push_back(values[i]);
                    ++escapes;
                }
            }
            result.payload.resize((result.payload.size() + 15U) & ~std::size_t{15U}, 0);
        }
    }
    return result;
}

} // namespace ninfer::ops
