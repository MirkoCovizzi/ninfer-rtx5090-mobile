#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity, std::uint32_t page_tokens) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / page_tokens;
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool kvarn = storage == KvCacheStorage::KvarnK4V2Group128;
    if (kvarn && head_dim != ops::kKvarnHeadDim) {
        throw std::invalid_argument("KVarN requires D256 heads");
    }
    const std::optional<PagedKVStorageLayout> layer_storage =
        kvarn ? std::nullopt : std::optional{paged_kv_storage_layout(storage, head_dim)};

    const std::uint32_t logical_pages = page_count(capacity, kv_page_tokens(storage));
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.page_tokens = kv_page_tokens(storage);
    geometry.planes.reserve(static_cast<std::size_t>(layers) *
                            (kvarn ? 1ULL : layer_storage->planes_per_layer()));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (kvarn) {
            geometry.planes.push_back(
                {DType::U8, ops::kKvarnRecordBytes / ops::kKvarnGroup, kv_heads, 256});
            continue;
        }
        geometry.planes.push_back(
            {layer_storage->key.data_dtype, layer_storage->key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back({layer_storage->value.data_dtype,
                                   layer_storage->value.data_leading_extent, kv_heads, 256});
        if (layer_storage->key.has_scale()) {
            geometry.planes.push_back({layer_storage->key.scale_dtype,
                                       layer_storage->key.scale_leading_extent, kv_heads, 256});
        }
        if (layer_storage->value.has_scale()) {
            geometry.planes.push_back({layer_storage->value.scale_dtype,
                                       layer_storage->value.scale_leading_extent, kv_heads, 256});
        }
    }
    PagedKVCacheLayout layout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers        = layers,
        .max_context   = capacity,
        .kv_heads      = kv_heads,
        .layer_storage = layer_storage,
    };
    if (kvarn) {
        const std::int32_t row_heads = table_rows * kv_heads * ops::kKvarnTailSlots;
        layout.kvarn_tail_k          = builder.add_tensor(
            DType::BF16, {head_dim, ops::kKvarnGroup, row_heads, static_cast<std::int32_t>(layers)},
            256, "KVarN rotated K sink/tail");
        layout.kvarn_tail_v = builder.add_tensor(
            DType::BF16, {head_dim, ops::kKvarnGroup, row_heads, static_cast<std::int32_t>(layers)},
            256, "KVarN rotated V sink/tail");
        layout.kvarn_tail_logical_pages = builder.add_tensor(
            DType::I32, {ops::kKvarnTailSlots, table_rows, static_cast<std::int32_t>(layers)}, 256,
            "KVarN sink/tail logical pages");
    }
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      layer_storage_(layout.layer_storage) {
    if (!layer_storage_) {
        kvarn_tail_k_             = layout.kvarn_tail_k.bind(backing);
        kvarn_tail_v_             = layout.kvarn_tail_v.bind(backing);
        kvarn_tail_logical_pages_ = layout.kvarn_tail_logical_pages.bind(backing);
        CUDA_CHECK(
            cudaMemset(kvarn_tail_logical_pages_.data, 0xff, kvarn_tail_logical_pages_.bytes()));
    }
    if (pages_.plane_count() != static_cast<std::size_t>(layers_) *
                                    (layer_storage_ ? layer_storage_->planes_per_layer() : 1ULL)) {
        throw std::invalid_argument("Paged KV layer plane inventory is inconsistent");
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table,
                                   std::int32_t table_row) noexcept
    : cache_(&cache), block_table_(block_table), table_row_(table_row) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

ops::KvarnPagedLayerView PagedKVCacheView::kvarn_layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->kvarn_layer_view(layer, block_table_, table_row_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()), row.row_index());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (!layer_storage_) { throw std::logic_error("KVarN cache requires kvarn_layer_view"); }
    const std::size_t stride        = layer_storage_->planes_per_layer();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_->key.has_scale());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = layer_storage_->key.has_scale() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = layer_storage_->value.has_scale() ? pages_.plane(v_scale_index) : Tensor(),
        .block_table   = block_table,
        .head_dim      = layer_storage_->head_dim,
        .num_kv_heads  = kv_heads_,
        .storage       = layer_storage_->storage,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor());
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
    };
}

ops::KvarnPagedLayerView PagedKVCache::kvarn_layer_view(std::uint32_t layer, Tensor block_table,
                                                        std::int32_t table_row) const {
    if (layer_storage_ || layer >= layers_ || table_row < 0 ||
        table_row >= kvarn_tail_logical_pages_.ne[1]) {
        throw std::invalid_argument("invalid KVarN layer view");
    }
    const std::int32_t row_heads = kv_heads_ * ops::kKvarnTailSlots;
    const std::int32_t begin     = table_row * row_heads;
    return {
        .records = pages_.plane(layer),
        .tail_k  = kvarn_tail_k_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .slice(2, begin, row_heads)
                      .view({ops::kKvarnHeadDim, ops::kKvarnGroup, row_heads}),
        .tail_v = kvarn_tail_v_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .slice(2, begin, row_heads)
                      .view({ops::kKvarnHeadDim, ops::kKvarnGroup, row_heads}),
        .tail_logical_pages =
            kvarn_tail_logical_pages_.slice(2, static_cast<std::int32_t>(layer), 1)
                .slice(1, table_row, 1)
                .view({ops::kKvarnTailSlots}),
        .block_table  = block_table,
        .num_kv_heads = kv_heads_,
    };
}

ops::KvarnPagedBatchLayerView PagedKVCache::kvarn_batch_layer_view(std::uint32_t layer) const {
    if (layer_storage_ || layer >= layers_) {
        throw std::invalid_argument("invalid KVarN batch layer view");
    }
    const std::int32_t rows      = kvarn_tail_logical_pages_.ne[1];
    const std::int32_t row_heads = kv_heads_ * ops::kKvarnTailSlots;
    return {
        .records = pages_.plane(layer),
        .tail_k  = kvarn_tail_k_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .view({ops::kKvarnHeadDim, ops::kKvarnGroup, row_heads, rows}),
        .tail_v = kvarn_tail_v_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .view({ops::kKvarnHeadDim, ops::kKvarnGroup, row_heads, rows}),
        .tail_logical_pages =
            kvarn_tail_logical_pages_.slice(2, static_cast<std::int32_t>(layer), 1)
                .view({ops::kKvarnTailSlots, rows}),
        .block_tables = execution_tables_.matrix(),
        .num_kv_heads = kv_heads_,
    };
}

void PagedKVCache::reset_kvarn_tail_row(std::int32_t table_row, cudaStream_t stream) const {
    if (layer_storage_ || table_row < 0 || table_row >= kvarn_tail_logical_pages_.ne[1]) { return; }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        Tensor markers = kvarn_tail_logical_pages_.slice(2, static_cast<std::int32_t>(layer), 1)
                             .slice(1, table_row, 1);
        CUDA_CHECK(cudaMemsetAsync(markers.data, 0xff, markers.bytes(), stream));
    }
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
