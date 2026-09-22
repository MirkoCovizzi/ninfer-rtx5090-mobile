#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N248320K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 =
    Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 =
    Q8A16SlicedKMmaSchedule<40, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
// Keep eight K splits across the full compact verification frontier.
using C48 = Q8A16SlicedKMmaSchedule<48, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::ActiveOnly, 0, 48, false, 24>;

void launch_decode_columns(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    // Bound per-CTA staging while retaining the ordinary-decode reduction for B8/W16.
    for (std::int32_t begin = 0; begin < x.ne[1]; begin += 32) {
        const std::int32_t count = std::min(32, x.ne[1] - begin);
        Tensor input             = x.slice(1, begin, count);
        Tensor output            = out.slice(1, begin, count);
        launch_q8_a16_sliced<Geometry, 32, C32>(input, weight, output, stream);
    }
}

} // namespace

Q8Launch select_q8_n248320_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 128) return launch_decode_columns;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
