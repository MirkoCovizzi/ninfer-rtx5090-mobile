#include "core/weight.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    static_assert(ActiveTokens >= 1 && ActiveTokens <= 32);
    // Preserve the residual's contraction profile throughout compact decode/MTP batches.
    constexpr int rows = (ActiveTokens == 4 || ActiveTokens == 6) ? 2 : (ActiveTokens <= 8 ? 4 : 2);
    using Schedule =
        Bf16A16SimtSchedule<4, 1, rows, 16, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                            Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_simt<Bf16ScheduleInstance<Schedule, 6144, ActiveTokens, true>>(
        bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
        LinearResidualAddEpilogue{{data, weight.n}}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kBf16LinearAddSmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(
    std::make_index_sequence<kBf16LinearAddSmallTMaxTokens - kBf16LinearAddSmallTMinTokens + 1>{});

} // namespace

void bf16_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    kLaunchers[static_cast<std::size_t>(x.ne[1] - kBf16LinearAddSmallTMinTokens)](x, weight,
                                                                                  residual, stream);
}

} // namespace ninfer::ops::detail
