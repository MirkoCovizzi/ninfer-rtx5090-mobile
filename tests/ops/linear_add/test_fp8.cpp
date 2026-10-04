#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
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
constexpr ReductionCriterion kA8Tolerance{0.04, kBf16UnitRoundoff, 0.06};

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    constexpr std::int32_t kSamples = 32;
    for (std::int32_t sample = 0; sample < kSamples; ++sample) {
        const std::int32_t index = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(extent - 1) * sample) / (kSamples - 1));
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

std::vector<double> represent_a8(std::span<const std::uint16_t> activation, int k) {
    std::array<double, 127> codes{};
    for (int code = 0; code < 127; ++code) {
        codes[code] = quantized_weight::detail::decode_e4m3fn(static_cast<std::uint8_t>(code));
    }
    std::vector<double> represented(activation.size());
    for (std::size_t begin = 0; begin < activation.size(); begin += k) {
        float maximum = 0;
        for (int row = 0; row < k; ++row) {
            maximum = std::max(maximum, std::abs(bf16_to_f32(activation[begin + row])));
        }
        const float scale   = maximum / 448.0F;
        const float inverse = scale > 0 ? 1.0F / scale : 0;
        for (int row = 0; row < k; ++row) {
            const float input      = bf16_to_f32(activation[begin + row]);
            const float normalized = std::abs(input * inverse);
            const auto upper       = std::lower_bound(codes.begin(), codes.end(), normalized);
            std::size_t code       = static_cast<std::size_t>(upper - codes.begin());
            if (code == codes.size()) {
                code = codes.size() - 1;
            } else if (code > 0) {
                const double down = normalized - codes[code - 1];
                const double up   = codes[code] - normalized;
                if (down < up || (down == up && (code & 1U))) { --code; }
            }
            represented[begin + row] = std::copysign(codes[code] * scale, input);
        }
    }
    return represented;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed, bool wide_range = false,
              bool cancellation = false) {
    std::vector<Invocation> invocations{
        Invocation{1, ops::LinearPolicy::A16Only},   Invocation{2, ops::LinearPolicy::A16Only},
        Invocation{26, ops::LinearPolicy::A16Only},  Invocation{1, ops::LinearPolicy::AllowA8},
        Invocation{2, ops::LinearPolicy::AllowA8},   Invocation{48, ops::LinearPolicy::AllowA8},
        Invocation{65, ops::LinearPolicy::AllowA8},  Invocation{1024, ops::LinearPolicy::AllowA8},
        Invocation{8, ops::LinearPolicy::AllowA8},   Invocation{16, ops::LinearPolicy::AllowA8},
        Invocation{32, ops::LinearPolicy::AllowA8},  Invocation{64, ops::LinearPolicy::AllowA8},
        Invocation{96, ops::LinearPolicy::AllowA8},  Invocation{128, ops::LinearPolicy::AllowA8},
        Invocation{129, ops::LinearPolicy::AllowA8},
    };
    for (int columns = 2; columns <= 24; ++columns) {
        invocations.push_back({columns, ops::LinearPolicy::A16Only});
    }
    for (int columns : {31, 32, 33, 63, 64, 65, 127, 128, 129, 1024})
        invocations.push_back({columns, ops::LinearPolicy::A16Only});
    if (wide_range || cancellation) {
        std::erase_if(invocations, [](Invocation invocation) {
            return invocation.policy != ops::LinearPolicy::AllowA8 || invocation.tokens > 128;
        });
    }
    const std::int32_t kMaximumTokens = wide_range || cancellation ? 128 : 1024;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, n, k, seed);
    if (cancellation) {
        // Unit codes keep the Tensor Core dot exact, isolating the residual scaling FMA
        // from FP32 dot accumulation error under near-total cancellation.
        for (std::size_t i = 0; i < host_weight.code_plane_bytes; ++i) {
            host_weight.payload[i] = (host_weight.payload[i] & 0x80U) | 0x38U;
        }
        for (int row = 0; row < n; ++row) {
            const auto scale = f32_to_bf16(0.005F * (1.0F + (row % 127) / 128.0F));
            quantized_weight::detail::store_u16_le(host_weight.payload,
                                                   host_weight.scale_plane_offset + row * 2, scale);
        }
    }
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    if (wide_range) {
        // Narrow dyadic inputs hide reduction-order differences. Vary BF16 exponents while
        // retaining independently decoded stored weights and the FP64 mathematical oracle.
        for (std::size_t i = 0; i < activation.size(); ++i) {
            activation[i] = f32_to_bf16(
                std::ldexp(bf16_to_f32(activation[i]), static_cast<int>((i * 17U) % 25U) - 12));
        }
    }
    if (cancellation) {
        // Zero/one BF16 inputs have negligible A8 representation error; cancellation then
        // exposes a full-tile FMA differing from a partial-tile MUL+ADD at the final BF16 store.
        for (std::size_t i = 0; i < activation.size(); ++i) {
            activation[i] = activation[i] & 0x8000U ? f32_to_bf16(0) : f32_to_bf16(1);
        }
    }
    const std::vector<double> represented_a8    = represent_a8(activation, k);
    std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);
    if (cancellation) {
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            for (int token = 0; token < kMaximumTokens; ++token) {
                double sum = 0;
                for (int column = 0; column < k; ++column) {
                    sum += static_cast<double>(materialized_weight[sampled_row * k + column]) *
                           bf16_to_f32(activation[static_cast<std::size_t>(token) * k + column]);
                }
                initial_residual[static_cast<std::size_t>(token) * n + rows[sampled_row]] =
                    f32_to_bf16(static_cast<float>(-sum));
            }
        }
    }

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
            QType::FP8_E4M3FN_ROW_BF16, n, k, invocation.policy, invocation.tokens,
            invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize FP8 linear_add");

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

        const bool a8           = invocation.policy == ops::LinearPolicy::AllowA8;
        const std::string label = "FP8 linear_add [" + std::to_string(n) + "," + std::to_string(k) +
                                  "] " + (a8 ? "A8" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        std::vector<double> expected_a8;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum    = 0.0;
                double sum_a8 = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                    sum_a8 += static_cast<double>(weight_row[column]) *
                              represented_a8[static_cast<std::size_t>(token) * k + column];
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
                expected_a8.push_back(sum_a8 +
                                      static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a8 ? kA8Tolerance : kA16Tolerance);
        if (a8) {
            failures += verify_reduction(label + " independently represented A8", actual,
                                         expected_a8, kA16Tolerance);
        }
        if (a8 && (invocation.tokens == 32 || invocation.tokens == 128)) {
            // Greedy MTP verification must preserve each ordinary one-column residual update.
            output.copy_from_host(initial_residual.data(), output.bytes());
            for (int token = 0; token < invocation.tokens; ++token) {
                Tensor one_input  = x.slice(1, token, 1);
                Tensor one_output = residual.slice(1, token, 1);
                ops::linear_add(one_input, weight, one_output, invocation.policy, workspace,
                                nullptr);
            }
            cuda_check(cudaDeviceSynchronize(), "synchronize FP8 canonical residual updates");
            std::vector<std::uint16_t> singles(output_words);
            output.copy_to_host(singles.data(), output.bytes());
            if (singles != actual_bits) {
                std::cerr << label << ": batched residual differs from ordinary columns\n";
                ++failures;
            }
            failures += output.verify_guards(label + " canonical columns");
        }
    }

    failures += device_activation.verify_guards("FP8 linear_add activation");
    failures += device_weight.verify_guards("FP8 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "FP8 linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "FP8 linear_add weight");

    const std::size_t a16_interval = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::A16Only, 1, 2048);
    const std::size_t exact_one = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, 1, 1);
    const std::size_t hot_interval = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, 1, 48);
    const std::size_t exact_48 = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, 48, 48);
    const std::size_t through_1024 = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, 1, 1024);
    const std::size_t exact_1024 = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, 1024, 1024);
    if (a16_interval != 0 || exact_one == 0 || hot_interval != exact_48 ||
        through_1024 != exact_1024 || exact_1024 <= exact_48) {
        std::cerr << "FP8 linear_add [" << n << ',' << k
                  << "]: workspace interval contract mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_shape(5120, 6144, 861U);
    failures += run_shape(5120, 17408, 863U);
    failures += run_shape(5120, 6144, 867U, true);
    failures += run_shape(5120, 17408, 869U, true);
    failures += run_shape(5120, 6144, 877U, false, true);
    failures += run_shape(5120, 17408, 881U, false, true);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 linear_add\n";
    return failures == 0 ? 0 : 1;
}
