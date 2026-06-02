#include "ssd_index.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <queue>
#include <strings.h>
#include <utility>
#include <vector>

#include "utils/timer.h"

namespace pipeann {
  namespace {
    size_t prefilter_batch_size() {
      constexpr size_t kDefaultBatchSize = MAX_N_EDGES;
      const char *explicit_value = std::getenv("PIPEANN_PREFILTER_BATCH_SIZE");
      if (explicit_value != nullptr) {
        const unsigned long parsed = std::strtoul(explicit_value, nullptr, 10);
        if (parsed > 0) {
          return std::min<size_t>(kDefaultBatchSize, std::max<size_t>(1, static_cast<size_t>(parsed)));
        }
      }

      const char *drop_cache = std::getenv("PIPEANN_PQ_MMAP_DROP_CACHE");
      if (drop_cache != nullptr && std::strcmp(drop_cache, "0") != 0 && strcasecmp(drop_cache, "false") != 0
          && strcasecmp(drop_cache, "off") != 0) {
        return 64;
      }
      return kDefaultBatchSize;
    }
  }  // namespace

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::hybrid_prefilter_search(const T *query, const uint64_t k_search, TagT *res_tags,
                                                    float *res_dists, const std::vector<uint32_t> &candidate_ids,
                                                    QueryStats *stats, uint64_t total_points_override) {
    if (candidate_ids.empty()) {
      if (stats != nullptr) {
        *stats = {};
      }
      return 0;
    }

    std::shared_lock lk(merge_lock);
    Timer query_timer;
    QueryBuffer<T> *query_buf = pop_query_buf(query);
    nbr_handler->initialize_query(query_buf->aligned_query_T, query_buf);

    const uint64_t total_points = total_points_override == 0 ? meta_.npoints : total_points_override;
    const size_t rerank_l = compute_prefilter_rerank_l(k_search, candidate_ids.size(), total_points);
    using PQCandidate = std::pair<float, uint32_t>;
    std::priority_queue<PQCandidate> top_candidates;

    const size_t kBatchSize = prefilter_batch_size();
    for (size_t offset = 0; offset < candidate_ids.size(); offset += kBatchSize) {
      const size_t current_batch = std::min(kBatchSize, candidate_ids.size() - offset);
      nbr_handler->compute_dists(query_buf, candidate_ids.data() + offset, current_batch);

      for (size_t batch_idx = 0; batch_idx < current_batch; ++batch_idx) {
        const float approx_dist = query_buf->aligned_dist_scratch[batch_idx];
        const uint32_t point_id = candidate_ids[offset + batch_idx];
        if (top_candidates.size() < rerank_l) {
          top_candidates.emplace(approx_dist, point_id);
        } else if (approx_dist < top_candidates.top().first) {
          top_candidates.pop();
          top_candidates.emplace(approx_dist, point_id);
        }
      }
    }

    std::vector<PQCandidate> shortlist;
    shortlist.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
      shortlist.push_back(top_candidates.top());
      top_candidates.pop();
    }

    std::vector<std::pair<float, uint32_t>> exact_results;
    exact_results.reserve(shortlist.size());
    T *vector_buf = query_buf->coord_scratch;
    char *sector_buf = query_buf->sector_scratch;
    QueryStats rerank_io_stats{};
    for (const auto &candidate : shortlist) {
      const uint32_t point_id = candidate.second;
      if (get_vector_by_id(point_id, vector_buf, sector_buf, stats == nullptr ? nullptr : &rerank_io_stats) != 0) {
        continue;
      }
      const float exact_dist = dist_cmp->compare(query_buf->aligned_query_T, vector_buf,
                                                 static_cast<unsigned>(meta_.data_dim));
      exact_results.emplace_back(exact_dist, point_id);
    }

    const size_t result_count = std::min(static_cast<size_t>(k_search), exact_results.size());
    std::partial_sort(exact_results.begin(), exact_results.begin() + result_count, exact_results.end());
    for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
      res_tags[result_idx] = id2tag(exact_results[result_idx].second);
      if (res_dists != nullptr) {
        res_dists[result_idx] = exact_results[result_idx].first;
      }
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = static_cast<double>(query_timer.elapsed());
      stats->n_cmps = static_cast<double>(candidate_ids.size());
      stats->n_ios = rerank_io_stats.n_ios;
      stats->n_4k = rerank_io_stats.n_4k;
      stats->read_size = rerank_io_stats.read_size;
      stats->n_hops = 0.0;
    }
    return result_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
