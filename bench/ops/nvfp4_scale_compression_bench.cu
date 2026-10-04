#include "ninfer/ops/nvfp4_scale_compression.h"
#include "ninfer_bench_common.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

int main() {
    try {
        using namespace ninfer;
        DeviceContext device;
        DeviceBuffer flush(256ULL << 20);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::puts("N,K,symbols,source_bytes,compressed_bytes,cold_median_us,cold_p95_us");
        for (const auto shape :
             {std::array{5120, 6144}, std::array{5120, 17408}, std::array{34816, 5120}}) {
            for (const int symbols : {15, 17, 256}) {
                const auto count = ops::nvfp4_scale_plane_bytes(shape[0], shape[1]);
                std::vector<std::byte> scales(count);
                std::uint32_t random = 42;
                for (auto& scale : scales) {
                    random ^= random << 13;
                    random ^= random >> 17;
                    random ^= random << 5;
                    scale = static_cast<std::byte>(random % symbols);
                }
                const auto encoded = ops::compress_nvfp4_scale_plane(scales, shape[0], shape[1]);
                DeviceBuffer payload(encoded.payload.size());
                DeviceBuffer offsets(encoded.offsets.size() * sizeof(std::uint32_t));
                DeviceBuffer output(count);
                payload.copy_from_host(encoded.payload.data(), payload.bytes);
                offsets.copy_from_host(encoded.offsets.data(), offsets.bytes);
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    ops::expand_nvfp4_scale_plane(static_cast<const std::uint8_t*>(payload.p),
                                                  static_cast<const std::uint32_t*>(offsets.p),
                                                  shape[0], shape[1],
                                                  static_cast<std::uint8_t*>(output.p), s);
                });
                const auto timing = bench::measure_cold_launch(
                    [&](cudaStream_t s) { graph.launch(s); }, flush, stream, 10, 100);
                std::printf("%d,%d,%d,%zu,%zu,%.3f,%.3f\n", shape[0], shape[1], symbols, count,
                            payload.bytes + offsets.bytes, timing.median_us, timing.p95_us);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
