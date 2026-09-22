#include "models/qwen3_5/program/planning/graph_profiles.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

bool dflash_target_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                        std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767, 122879});
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Verification and the next proposal chain use the same active K. Bound E+2K.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // KVarN verification changes route at 1K and width-one AR changes splits at 120K.
    add_shifted(1024U, draft_window + 1);
    add_shifted(122880U, 2 * draft_window);
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    auto profiles = graph_profiles_through(capacity - 1, ends);
    // KVarN H24 verification uses six-column chunks below 1K and a single packed block above
    // it. Wider verification therefore changes graph node count at this boundary, not just
    // launch parameters. Keep these definitions in distinct executable-update classes.
    for (auto& profile : profiles) {
        const auto visible = std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1);
        profile.topology_class = draft_window >= 6 && visible > 1024 ? 1U : 0U;
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window,
                                                         std::uint32_t batch_size) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

} // namespace ninfer::models::qwen3_5::detail
