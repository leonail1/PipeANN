#include "ssd_index.h"

#include <cstring>
#include <limits>
#include <vector>

#include "filter/selector.h"
#include "utils/timer.h"

namespace pipeann {
  namespace {
    constexpr uint64_t kMetadataAllowPrefilterFlag = 1ULL << 2;

    AbstractSelector *selector_from_kind(HybridFilterKind filter_kind) {
      static LabelIntersectionSelector intersect_selector;
      static LabelSubsetSelector subset_selector;
      static RangeSelector range_selector;
      switch (filter_kind) {
        case HybridFilterKind::kIntersect:
          return &intersect_selector;
        case HybridFilterKind::kSubset:
          return &subset_selector;
        case HybridFilterKind::kRange:
          return &range_selector;
        case HybridFilterKind::kUnsupported:
        default:
          return nullptr;
      }
    }

    uint64_t selector_mask_for_kind(HybridFilterKind filter_kind) {
      switch (filter_kind) {
        case HybridFilterKind::kIntersect:
          return 1ULL;
        case HybridFilterKind::kSubset:
          return 2ULL;
        case HybridFilterKind::kRange:
          return 4ULL;
        case HybridFilterKind::kUnsupported:
        default:
          return 0ULL;
      }
    }

    struct DenseBitsetSelector : public AbstractSelector {
      const std::vector<uint64_t> *bitset_words = nullptr;
      uint64_t npoints = 0;

      bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
        (void) query_labels;
        (void) target_labels;
        if (bitset_words == nullptr || target_id >= npoints) {
          return false;
        }
        const uint64_t word_idx = static_cast<uint64_t>(target_id) >> 6U;
        if (word_idx >= bitset_words->size()) {
          return false;
        }
        const uint64_t bit_mask = 1ULL << (static_cast<uint64_t>(target_id) & 63ULL);
        return ((*bitset_words)[static_cast<size_t>(word_idx)] & bit_mask) != 0;
      }
    };

    std::vector<uint32_t> decode_label_filter_data(const void *filter_data) {
      if (filter_data == nullptr) {
        return {};
      }

      uint32_t label_count = 0;
      std::memcpy(&label_count, filter_data, sizeof(uint32_t));
      std::vector<uint32_t> labels(label_count);
      if (label_count > 0) {
        std::memcpy(labels.data(), static_cast<const char *>(filter_data) + sizeof(uint32_t),
                    static_cast<size_t>(label_count) * sizeof(uint32_t));
      }
      return labels;
    }

    template<typename TagT>
    void clear_result_buffers(uint64_t k_search, TagT *res_tags, float *res_dists) {
      for (uint64_t result_idx = 0; result_idx < k_search; ++result_idx) {
        if (res_tags != nullptr) {
          res_tags[result_idx] = std::numeric_limits<TagT>::max();
        }
        if (res_dists != nullptr) {
          res_dists[result_idx] = std::numeric_limits<float>::infinity();
        }
      }
    }
  }  // namespace

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::hybrid_search(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                          const uint64_t l_search, TagT *res_tags, float *res_dists,
                                          const uint64_t beam_width, HybridFilterKind filter_kind,
                                          const void *filter_data, QueryStats *stats,
                                          HybridQueryStats *hybrid_stats,
                                          HybridRouteOverride route_override) {
    Timer route_timer;
    if (hybrid_stats != nullptr) {
      *hybrid_stats = {};
    }

    AbstractSelector *selector = selector_from_kind(filter_kind);
    auto finish_graph_path = [&](HybridRouteDecision decision, AbstractSelector *graph_selector,
                                 const void *graph_filter_data) -> size_t {
      const uint64_t route_overhead_us = route_timer.elapsed();
      if (hybrid_stats != nullptr) {
        hybrid_stats->decision = decision;
        hybrid_stats->route_overhead_us = route_overhead_us;
      }
      const size_t result_count = pipe_search(query, k_search, mem_L, l_search, res_tags, res_dists, beam_width,
                                              stats, graph_selector, graph_filter_data, 0);
      if (stats != nullptr) {
        stats->total_us += static_cast<double>(route_overhead_us);
      }
      return result_count;
    };

    if (filter_kind == HybridFilterKind::kUnsupported || selector == nullptr) {
      return finish_graph_path(HybridRouteDecision::kAutoGraphFallback, selector, filter_data);
    }

    const uint64_t selector_mask = selector_mask_for_kind(filter_kind);
    if (densebit_index_ == nullptr) {
      return finish_graph_path(HybridRouteDecision::kAutoGraphFallback, selector, filter_data);
    }

    HybridQueryScratch scratch;
    const std::vector<uint32_t> labels = decode_label_filter_data(filter_data);
    const uint64_t candidate_count = densebit_index_->count_candidates(filter_kind, labels, &scratch);
    const uint64_t route_overhead_us = route_timer.elapsed();
    const bool routing_metadata_ready = hybrid_enabled() && hybrid_metadata_ != nullptr;
    const HybridMetadataHeaderV1 *meta_header = routing_metadata_ready ? &hybrid_metadata_->header() : nullptr;
    const bool calibrated_for_selector =
        meta_header != nullptr && (meta_header->flags & kMetadataAllowPrefilterFlag) != 0
        && (meta_header->route_selector_mask & selector_mask) != 0;
    const uint64_t tau_m = calibrated_for_selector ? meta_header->tau_m : 0;

    if (hybrid_stats != nullptr) {
      hybrid_stats->candidate_count = candidate_count;
      hybrid_stats->threshold = tau_m;
      hybrid_stats->threshold_version = meta_header == nullptr ? 0 : meta_header->threshold_version;
      hybrid_stats->route_overhead_us = route_overhead_us;
    }

    if (candidate_count == 0) {
      clear_result_buffers(k_search, res_tags, res_dists);
      if (stats != nullptr) {
        *stats = {};
        stats->total_us = static_cast<double>(route_overhead_us);
      }
      if (hybrid_stats != nullptr) {
        hybrid_stats->decision = HybridRouteDecision::kPrefilterFastReturn;
      }
      return 0;
    }

    const bool force_prefilter = route_override == HybridRouteOverride::kForcePrefilter;
    const bool force_graph_only = route_override == HybridRouteOverride::kForceGraphOnly;
    const bool choose_prefilter = force_prefilter || (!force_graph_only && calibrated_for_selector
                                                      && candidate_count <= tau_m);

    if (!choose_prefilter) {
      DenseBitsetSelector densebit_selector;
      densebit_selector.bitset_words = &scratch.bitset_words;
      densebit_selector.npoints = densebit_index_->header().npoints;
      if (hybrid_stats != nullptr) {
        hybrid_stats->decision = calibrated_for_selector || force_graph_only
                                      ? HybridRouteDecision::kGraphOnly
                                      : HybridRouteDecision::kAutoGraphFallback;
      }
      return finish_graph_path(calibrated_for_selector || force_graph_only ? HybridRouteDecision::kGraphOnly
                                                                           : HybridRouteDecision::kAutoGraphFallback,
                               &densebit_selector, nullptr);
    }

    std::vector<uint32_t> candidate_ids;
    densebit_index_->materialize_candidates(filter_kind, labels, &scratch, &candidate_ids);
    const size_t result_count = hybrid_prefilter_search(query, k_search, res_tags, res_dists, candidate_ids, stats);
    if (stats != nullptr) {
      stats->total_us += static_cast<double>(route_overhead_us);
    }
    if (hybrid_stats != nullptr) {
      hybrid_stats->decision = HybridRouteDecision::kPrefilter;
      hybrid_stats->route_overhead_us = route_overhead_us;
    }
    return result_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
