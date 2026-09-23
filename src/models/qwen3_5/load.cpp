#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "core/layout.h"
#include "models/qwen3_5/load/bindings.h"
#include "ninfer/ops/nvfp4_scale_compression.h"

#include <map>
#include <limits>
#include <set>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::map<std::size_t, ops::Nvfp4CompressedScales> compressed_scales;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    if (options.nvfp4_scale_compression) {
        // Select by the mathematical consumer, then by actual parent representation. Only complete
        // NVFP4 dense-FFN parents enter the compressed execution path. Other artifact combinations
        // continue to bind normally, including FP8 layers in mixed-representation artifacts.
        std::set<std::size_t> selected_weights;
        for (const auto& block : out->weights.text.layers) {
            if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
                selected_weights.insert(dense->gate.index);
                selected_weights.insert(dense->up.index);
                selected_weights.insert(dense->down.index);
            }
        }
        if (out->weights.mtp) {
            if (const auto* dense = std::get_if<DenseWeights>(&out->weights.mtp->layer.ffn)) {
                selected_weights.insert(dense->gate.index);
                selected_weights.insert(dense->up.index);
                selected_weights.insert(dense->down.index);
            }
        }
        std::set<std::size_t> selected_objects;
        for (const auto id : selected_weights) {
            const auto& reference = bindings.weights[id].reference;
            for (const auto& part : reference.binding.parts) {
                const auto& g = reader.geometry(part.object);
                if (g.format != QType::NVFP4) { continue; }
                if (g.layout != QuantLayout::BlockScaleK16M128x4 || g.shape.size() != 2 ||
                    g.shape[0] > std::numeric_limits<std::int32_t>::max() ||
                    g.shape[1] > std::numeric_limits<std::int32_t>::max()) {
                    throw artifact::ArtifactError(reference.name +
                                                  ": unsupported compressed NVFP4 geometry");
                }
                (void)ops::nvfp4_scale_plane_bytes(static_cast<std::int32_t>(g.shape[0]),
                                                   static_cast<std::int32_t>(g.shape[1]));
                selected_objects.insert(part.object.index);
            }
        }
        if (selected_objects.empty()) {
            throw artifact::ArtifactError(
                "NVFP4 scale compression requires NVFP4 dense FFN weights");
        }
        for (std::size_t id = 0; id < bindings.weights.size(); ++id) {
            if (selected_weights.contains(id)) { continue; }
            for (const auto& part : bindings.weights[id].reference.binding.parts) {
                if (selected_objects.contains(part.object.index)) {
                    throw artifact::ArtifactError("compressed NVFP4 parent has another consumer");
                }
            }
        }
        for (const auto index : selected_objects) {
            const artifact::ObjectHandle handle{index};
            const auto& g     = reader.geometry(handle);
            const auto scales = binder.read_object_range(handle, g.scale_offset, g.scale_bytes);
            auto packed =
                ops::compress_nvfp4_scale_plane(scales, static_cast<std::int32_t>(g.shape[0]),
                                                static_cast<std::int32_t>(g.shape[1]));
            const auto compressed_bytes = ((packed.payload.size() + 255U) & ~std::size_t{255U}) +
                                          packed.offsets.size() * sizeof(std::uint32_t);
            // Some scale planes have nearly 256 distinct bytes per tile. Keep their original
            // representation rather than expanding the resident model under a memory-saving flag.
            if (compressed_bytes >= g.bytes - g.code_bytes) { continue; }
            out->compressed_scales.emplace(index, std::move(packed));
            binder.require_device_prefix(handle, g.code_bytes);
        }
        WorkspaceLayoutBuilder scale_storage;
        for (const auto& [index, scales] : out->compressed_scales) {
            (void)index;
            const auto aligned = (scales.payload.size() + 255U) & ~std::size_t{255U};
            (void)scale_storage.alloc_bytes(aligned + scales.offsets.size() * sizeof(std::uint32_t),
                                            256);
        }
        if (!out->compressed_scales.empty()) {
            binder.reserve_device_bytes(scale_storage.peak_bytes(1));
        }
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    for (const auto& [index, scales] : data->compressed_scales) {
        backing.attach_compressed_scales({index}, scales.payload, scales.offsets,
                                         scales.tiles_per_row);
    }
    auto bound = loading::resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
