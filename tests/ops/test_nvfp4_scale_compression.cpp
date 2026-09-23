#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/nvfp4_scale_compression.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;

int check(int rows, int columns, int pattern) {
    const auto count = ops::nvfp4_scale_plane_bytes(rows, columns);
    std::vector<std::byte> scales(count);
    // Independent logical (row, group) -> native swizzle. Force escapes at prefix and
    // record boundaries, as well as a full 256-symbol tile and a low-entropy tile.
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < columns / 16; ++group) {
            const int symbol = pattern == 0 ? ((row * 13 + group * 17) & 255)
                                            : ((row + group) % 17 == 0 ? 231 : 42);
            const auto offset =
                (static_cast<std::size_t>(row / 128) * (columns / 64) + group / 4) * 512 +
                (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
            scales[offset] = static_cast<std::byte>(symbol);
        }
    }
    const auto encoded = ops::compress_nvfp4_scale_plane(scales, rows, columns);
    DeviceBuffer compressed(encoded.payload.size());
    DeviceBuffer offsets(encoded.offsets.size() * sizeof(std::uint32_t));
    DeviceBuffer output(count);
    compressed.copy_from_host(encoded.payload.data(), encoded.payload.size());
    offsets.copy_from_host(encoded.offsets.data(), offsets.bytes);
    ops::expand_nvfp4_scale_plane(static_cast<const std::uint8_t*>(compressed.p),
                                  static_cast<const std::uint32_t*>(offsets.p), rows, columns,
                                  static_cast<std::uint8_t*>(output.p), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::byte> actual(count);
    output.copy_to_host(actual.data(), count);
    if (actual != scales) {
        std::cerr << "NVFP4 scale expansion differs from the source scale plane\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        failures += check(128, 128, 0);
        failures += check(256, 256, 1);
        failures += check(5120, 17408, 1);
        std::cout << (failures ? "FAIL" : "OK") << " NVFP4 lossless scale expansion\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
