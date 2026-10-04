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

void check_host_decode(const ops::Nvfp4CompressedScales& encoded,
                       const std::vector<std::byte>& expected) {
    // Independent byte-wise decoder of the public representation. This also checks the
    // encoder without allowing a matching GPU indexing mistake to hide a codec error.
    std::vector<std::byte> decoded;
    for (const auto offset : encoded.offsets) {
        const auto* record = encoded.payload.data() + offset;
        if (offset % 16 || record[15] != 0) { throw std::runtime_error("unaligned scale record"); }
        unsigned escapes = 0;
        for (unsigned i = 0; i < 1024; ++i) {
            if (i % 64 == 0) {
                const auto p = 528 + i / 64 * 2;
                if ((record[p] | (unsigned(record[p + 1]) << 8)) != escapes) {
                    throw std::runtime_error("incorrect escape prefix");
                }
            }
            const auto index = (record[16 + i / 2] >> ((i % 2) * 4)) & 15;
            decoded.push_back(
                static_cast<std::byte>(index == 15 ? record[560 + escapes++] : record[index]));
        }
    }
    if (decoded != expected) { throw std::runtime_error("host scale oracle mismatch"); }
}

int check(int rows, int columns, int pattern) {
    const auto count = ops::nvfp4_scale_plane_bytes(rows, columns);
    std::vector<std::byte> scales(count);
    // Independent logical (row, group) -> native swizzle. Force escapes at prefix and
    // record boundaries, as well as a full 256-symbol tile and a low-entropy tile.
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < columns / 16; ++group) {
            const int symbol = pattern == 0   ? ((row * 13 + group * 17) & 255)
                               : pattern == 1 ? ((row + group) % 17 == 0 ? 231 : 42)
                                              : ((row * 29 + group * 13) % 19);
            const auto offset =
                (static_cast<std::size_t>(row / 128) * (columns / 64) + group / 4) * 512 +
                (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
            scales[offset] = static_cast<std::byte>(symbol);
        }
    }
    const auto encoded = ops::compress_nvfp4_scale_plane(scales, rows, columns);
    check_host_decode(encoded, scales);
    DeviceBuffer compressed(encoded.payload.size());
    DeviceBuffer offsets(encoded.offsets.size() * sizeof(std::uint32_t));
    DeviceBuffer output(count + 32);
    auto* destination = static_cast<std::uint8_t*>(output.p) + 16;
    compressed.copy_from_host(encoded.payload.data(), encoded.payload.size());
    offsets.copy_from_host(encoded.offsets.data(), offsets.bytes);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&] {
        ops::expand_nvfp4_scale_plane(static_cast<const std::uint8_t*>(compressed.p),
                                      static_cast<const std::uint32_t*>(offsets.p), rows, columns,
                                      destination, stream);
    };
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        CUDA_CHECK(cudaMemsetAsync(output.p, 0xa5, output.bytes, stream));
        if (replay == 0) {
            launch();
        } else {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::byte> actual(output.bytes);
        output.copy_to_host(actual.data(), output.bytes);
        if (!std::equal(scales.begin(), scales.end(), actual.begin() + 16) ||
            !std::all_of(actual.begin(), actual.begin() + 16,
                         [](auto v) { return v == std::byte{0xa5}; }) ||
            !std::all_of(actual.end() - 16, actual.end(),
                         [](auto v) { return v == std::byte{0xa5}; })) {
            std::cerr << "NVFP4 scale expansion or output guard mismatch\n";
            ++failures;
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        failures += check(128, 128, 0);
        failures += check(256, 256, 1);
        failures += check(128, 384, 2);
        failures += check(5120, 6144, 2);
        failures += check(5120, 17408, 1);
        failures += check(34816, 5120, 2);
        std::cout << (failures ? "FAIL" : "OK") << " NVFP4 lossless scale expansion\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
