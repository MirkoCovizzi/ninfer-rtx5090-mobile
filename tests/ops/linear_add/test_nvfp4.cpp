#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ninfer/ops/nvfp4_scale_compression.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

bool uses_a4(std::int32_t k, Invocation invocation) {
    return invocation.policy == ops::LinearPolicy::AllowA4 &&
           (k == 17408 || invocation.tokens > 8 * 16);
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed,
              bool test_compressed_scales = false) {
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{5, ops::LinearPolicy::A16Only},
        Invocation{8, ops::LinearPolicy::A16Only},
        Invocation{16, ops::LinearPolicy::A16Only},
        Invocation{17, ops::LinearPolicy::A16Only},
        Invocation{24, ops::LinearPolicy::A16Only},
        Invocation{25, ops::LinearPolicy::A16Only},
        Invocation{32, ops::LinearPolicy::A16Only},
        Invocation{33, ops::LinearPolicy::A16Only},
        Invocation{48, ops::LinearPolicy::A16Only},
        Invocation{49, ops::LinearPolicy::A16Only},
        Invocation{64, ops::LinearPolicy::A16Only},
        Invocation{65, ops::LinearPolicy::A16Only},
        Invocation{128, ops::LinearPolicy::A16Only},
        Invocation{129, ops::LinearPolicy::A16Only},
        Invocation{1024, ops::LinearPolicy::A16Only},
        Invocation{48, ops::LinearPolicy::AllowA4},
        Invocation{49, ops::LinearPolicy::AllowA4},
        Invocation{1, ops::LinearPolicy::AllowA4},
        Invocation{2, ops::LinearPolicy::AllowA4},
        Invocation{4, ops::LinearPolicy::AllowA4},
        Invocation{17, ops::LinearPolicy::AllowA4},
        Invocation{8, ops::LinearPolicy::AllowA4},
        Invocation{16, ops::LinearPolicy::AllowA4},
        Invocation{32, ops::LinearPolicy::AllowA4},
        Invocation{64, ops::LinearPolicy::AllowA4},
        Invocation{96, ops::LinearPolicy::AllowA4},
        Invocation{128, ops::LinearPolicy::AllowA4},
        Invocation{129, ops::LinearPolicy::AllowA4},
        // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
        // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
        // single real token, which is the emptiest grid this route ever runs.
        Invocation{1023, ops::LinearPolicy::AllowA4},
        Invocation{1024, ops::LinearPolicy::AllowA4},
        Invocation{1025, ops::LinearPolicy::AllowA4},
    };
    // The invocation list is what drives the host buffers, so take the bound from it rather than
    // from a literal that silently caps it.
    const std::int32_t kMaximumTokens =
        std::max_element(
            invocations.begin(), invocations.end(),
            [](const Invocation& a, const Invocation& b) { return a.tokens < b.tokens; })
            ->tokens;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    const std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    const std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    int failures = 0;
    for (const Invocation invocation : invocations) {
        const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
        GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
        Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, invocation.policy, invocation.tokens, invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize NVFP4 linear_add");

        if (invocation.tokens == 128) {
            cudaStream_t stream;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            ops::linear_add(x, weight, residual, invocation.policy, workspace, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), initial_residual.data(), output.bytes(),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        const bool a4           = uses_a4(k, invocation);
        const std::string label = "NVFP4 linear_add [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] " + (a4 ? "A4" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a4 ? kA4Tolerance : kA16Tolerance);

        if (k == 6144 && (invocation.tokens == 32 || invocation.tokens == 128) && !a4) {
            // Supplement the independent oracle with the width-invariance required by greedy
            // speculative verification: all columns must match ordinary one-column updates.
            output.copy_from_host(initial_residual.data(), output.bytes());
            for (std::int32_t token = 0; token < invocation.tokens; ++token) {
                Tensor single_x(static_cast<std::uint16_t*>(device_activation.data()) +
                                    static_cast<std::size_t>(token) * k,
                                DType::BF16, {k, 1});
                Tensor single_out(static_cast<std::uint16_t*>(output.data()) +
                                      static_cast<std::size_t>(token) * n,
                                  DType::BF16, {n, 1});
                ops::linear_add(single_x, weight, single_out, invocation.policy, workspace,
                                nullptr);
            }
            cuda_check(cudaDeviceSynchronize(), "synchronize one-column NVFP4 linear_add");
            std::vector<std::uint16_t> single_bits(output_words);
            output.copy_to_host(single_bits.data(), output.bytes());
            if (single_bits != actual_bits) {
                std::cerr << label << ": batched residual differs from one-column updates\n";
                ++failures;
            }
        }
    }

    if (test_compressed_scales) {
        const std::size_t code_bytes   = static_cast<std::size_t>(n) * k / 2;
        const std::size_t scale_offset = (code_bytes + 255U) & ~std::size_t{255U};
        const auto scales              = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(host_weight.payload.data() + scale_offset),
            static_cast<std::size_t>(n) * k / 16);
        const auto packed = ops::compress_nvfp4_scale_plane(scales, n, k);
        DeviceBuffer codes(code_bytes), payload(packed.payload.size());
        DeviceBuffer offsets(packed.offsets.size() * sizeof(std::uint32_t));
        codes.copy_from_host(host_weight.payload.data(), code_bytes);
        payload.copy_from_host(packed.payload.data(), packed.payload.size());
        offsets.copy_from_host(packed.offsets.data(), offsets.bytes);
        Weight compressed                         = host_weight.device_weight(codes.p);
        compressed.scales                         = nullptr;
        compressed.payload_bytes                  = code_bytes;
        compressed.compressed_scales              = payload.p;
        compressed.compressed_scale_offsets       = static_cast<const std::uint32_t*>(offsets.p);
        compressed.compressed_scale_tiles_per_row = packed.tiles_per_row;
        for (const Invocation invocation : {Invocation{1, ops::LinearPolicy::A16Only},
                                            Invocation{128, ops::LinearPolicy::AllowA4},
                                            Invocation{129, ops::LinearPolicy::AllowA4}}) {
            const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
            GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
            output.copy_from_host(initial_residual.data(), output.bytes());
            Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
            Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
            const auto capacity = ops::linear_add_workspace_capacity_bytes(
                compressed, invocation.policy, invocation.tokens, invocation.tokens);
            WorkspaceArena workspace(capacity);
            ops::linear_add(x, compressed, residual, invocation.policy, workspace, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (workspace.peak_used() != capacity) {
                std::cerr << "compressed linear_add: incorrect workspace capacity\n";
                ++failures;
            }
            std::vector<std::uint16_t> actual_bits(output_words);
            output.copy_to_host(actual_bits.data(), output.bytes());
            std::vector<double> actual, expected;
            for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
                const int row           = rows[sampled_row];
                const float* weight_row = materialized_weight.data() + sampled_row * k;
                for (const int token : sampled_indices(invocation.tokens)) {
                    double sum = 0;
                    for (int column = 0; column < k; ++column) {
                        sum +=
                            static_cast<double>(weight_row[column]) *
                            bf16_to_f32(activation[static_cast<std::size_t>(token) * k + column]);
                    }
                    const std::size_t index = static_cast<std::size_t>(token) * n + row;
                    actual.push_back(bf16_to_f32(actual_bits[index]));
                    expected.push_back(sum + bf16_to_f32(initial_residual[index]));
                }
            }
            failures += verify_reduction(
                "compressed NVFP4 linear_add T=" + std::to_string(invocation.tokens), actual,
                expected, uses_a4(k, invocation) ? kA4Tolerance : kA16Tolerance);
            failures += output.verify_guards("compressed NVFP4 linear_add output");
        }
    }

    failures += device_activation.verify_guards("NVFP4 linear_add activation");
    failures += device_weight.verify_guards("NVFP4 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "NVFP4 linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "NVFP4 linear_add weight");
    return failures;
}

int check_residual_cancellation(std::int32_t k) {
    constexpr int n = 5120, max_tokens = 1025;
    auto packed = quantized_weight::make_patterned_weight(
        QType::NVFP4, n, k, 823U, {.weight_scale_divisor = 0.13F, .input_scale_divisor = 1.0F});
    // A single nonzero product per row is exactly representable before global scaling.
    // The residual nearly cancels it, exposing width-dependent rounding of scale+add.
    std::fill_n(packed.payload.begin(), static_cast<std::size_t>(n) * k / 2, 0);
    std::fill_n(packed.payload.begin() + packed.scale_plane_offset, packed.scale_plane_bytes, 0);
    for (int row = 0; row < n; ++row) {
        packed.payload[static_cast<std::size_t>(row) * k / 2] = 0x07; // E2M1: 6
        const auto scale                                      = packed.scale_plane_offset +
                           static_cast<std::size_t>(row / 128) * (k / 64) * 512 + (row % 32) * 16 +
                           (row % 128) / 32 * 4;
        packed.payload[scale] = 0x0d; // E4M3: 13/512
    }
    const double ideal       = 6.0 * quantized_weight::logical_weight_fp64(packed, 0, 0);
    const auto residual_bits = f32_to_bf16(-static_cast<float>(ideal));
    std::vector<std::uint16_t> input(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token) { input[token * k] = f32_to_bf16(6.0F); }
    // Include uncancelled rows as well: the existing tensor-level relative-L2 criterion
    // needs a nonzero reference norm, while the cancelled rows probe small absolute errors.
    std::vector<std::uint16_t> initial(static_cast<std::size_t>(n) * max_tokens);
    for (std::size_t i = 0; i < initial.size(); ++i) { initial[i] = i % 2 ? 0 : residual_bits; }
    std::vector<double> oracle(n);
    for (int row = 0; row < n; ++row) {
        oracle[row] = ideal + static_cast<double>(bf16_to_f32(initial[row]));
    }
    GuardedDeviceBuffer device_weight(packed.payload.size());
    device_weight.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(device_weight.data());
    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), device_input.bytes());

    int failures = 0;
    std::vector<int> widths;
    // Qualify both arithmetic profiles against the same represented-input oracle. Exact
    // width-invariance applies within a profile, not across A16/A4 dispatch boundaries.
    for (int t = 1; t <= 48; ++t) { widths.push_back(t); }
    for (int t : {49, 63, 64, 65, 127, 128, 129, 191, 192, 193, 383, 384, 385, 511, 512, 513, 1023,
                  1024, 1025}) {
        widths.push_back(t);
    }
    std::array<std::vector<std::uint16_t>, 2> references;
    for (const int t : widths) {
        const bool a4 = uses_a4(k, {t, ops::LinearPolicy::AllowA4});
        GuardedDeviceBuffer output(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
        output.copy_from_host(initial.data(), output.bytes());
        Tensor x(device_input.data(), DType::BF16, {k, t});
        Tensor residual(output.data(), DType::BF16, {n, t});
        const auto capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, t, t);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, ops::LinearPolicy::AllowA4, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize cancellation LinearAdd");
        std::vector<std::uint16_t> actual(static_cast<std::size_t>(n) * t);
        output.copy_to_host(actual.data(), output.bytes());
        const std::string label =
            "NVFP4 residual cancellation K=" + std::to_string(k) + " T=" + std::to_string(t);
        auto& reference = references[a4 ? 1 : 0];
        if (reference.empty()) { reference.assign(actual.begin(), actual.begin() + n); }
        bool equal = true;
        for (std::size_t i = 0; i < actual.size(); ++i) { equal &= actual[i] == reference[i % n]; }
        if (!equal) {
            std::cerr << label << ": identical columns changed with tile occupancy\n";
            ++failures;
        }
        std::vector<double> observed(n);
        for (int row = 0; row < n; ++row) { observed[row] = bf16_to_f32(actual[row]); }
        failures += verify_reduction(label, observed, oracle, a4 ? kA4Tolerance : kA16Tolerance);
        failures += output.verify_guards(label);
    }
    failures += device_input.verify_guards("cancellation input");
    failures += device_weight.verify_guards("cancellation weight");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_shape(5120, 6144, 811U);
    failures += run_shape(5120, 17408, 821U, true);
    failures += check_residual_cancellation(6144);
    failures += check_residual_cancellation(17408);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 linear_add\n";
    return failures == 0 ? 0 : 1;
}
