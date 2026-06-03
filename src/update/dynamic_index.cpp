#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>
#include <utility>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <gperftools/malloc_extension.h>

#include "filter/label.h"
#include "ssd_index.h"
#include "utils/index_build_utils.h"

#include "linux_aligned_file_reader.h"

namespace pipeann {
  namespace {
    constexpr uint64_t kMetadataValidFlag = 1ULL << 0;
    constexpr uint64_t kCalibrationValidFlag = 1ULL << 1;
    constexpr uint64_t kMetadataAllowPrefilterFlag = 1ULL << 2;
    constexpr uint64_t kCalibrationSeed = 20260423ULL;

    bool exceeds_recalibration_drift(uint64_t n_calib, uint64_t live_count) {
      if (n_calib == 0) {
        return false;
      }
      const uint64_t diff = n_calib > live_count ? (n_calib - live_count) : (live_count - n_calib);
      return diff * 10ULL > n_calib;
    }

    uint64_t next_power_of_two(uint64_t value) {
      if (value <= 1) {
        return value == 0 ? 0 : 1;
      }
      --value;
      value |= value >> 1;
      value |= value >> 2;
      value |= value >> 4;
      value |= value >> 8;
      value |= value >> 16;
      value |= value >> 32;
      return value + 1;
    }

    uint64_t p50_as_uint64(std::vector<double> samples) {
      if (samples.empty()) {
        return 0;
      }
      std::sort(samples.begin(), samples.end());
      return static_cast<uint64_t>(std::llround(samples[samples.size() / 2]));
    }

    template<typename TagT>
    struct CalibrationBucketSamples {
      std::vector<double> prefilter_us;
      std::vector<double> graph_us;
    };

    const char *selector_name_for_kind(HybridFilterKind filter_kind) {
      switch (filter_kind) {
        case HybridFilterKind::kIntersect:
          return "intersect";
        case HybridFilterKind::kSubset:
          return "subset";
        case HybridFilterKind::kRange:
          return "range";
        case HybridFilterKind::kUnsupported:
        default:
          return nullptr;
      }
    }

    template<typename TagT>
    std::vector<TagT> load_tags_by_id_from_disk(const std::string &index_prefix, uint64_t npoints) {
      const std::string tag_file = index_prefix + "_disk.index.tags";
      std::vector<TagT> tags_by_id(static_cast<size_t>(npoints));
      if (!file_exists(tag_file)) {
        for (uint64_t id = 0; id < npoints; ++id) {
          tags_by_id[static_cast<size_t>(id)] = static_cast<TagT>(id);
        }
        return tags_by_id;
      }

      size_t tag_num = 0, tag_dim = 0;
      pipeann::load_bin<TagT>(tag_file, tags_by_id, tag_num, tag_dim, 0);
      if (tag_num != npoints) {
        LOG(ERROR) << "Tag file size mismatch for " << tag_file << ": expected " << npoints << ", got "
                   << tag_num;
        exit(-1);
      }
      return tags_by_id;
    }

    template<typename TagT>
    bool ids_are_compact(const std::unordered_map<TagT, uint32_t> &live_ids_by_tag, uint64_t npoints) {
      if (live_ids_by_tag.size() != static_cast<size_t>(npoints)) {
        return false;
      }

      std::vector<uint8_t> seen(static_cast<size_t>(npoints), 0);
      for (const auto &entry : live_ids_by_tag) {
        const uint32_t id = entry.second;
        if (id >= npoints || seen[static_cast<size_t>(id)] != 0) {
          return false;
        }
        seen[static_cast<size_t>(id)] = 1;
      }
      return true;
    }

    uint64_t dense_words_per_label(uint64_t npoints) {
      return (npoints + 63ULL) / 64ULL;
    }

    void normalize_labels(std::vector<uint32_t> *labels) {
      std::sort(labels->begin(), labels->end());
      labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    }

    std::vector<uint32_t> decode_label_filter_data(const void *filter_data) {
      if (filter_data == nullptr) {
        return {};
      }

      uint32_t label_count = 0;
      memcpy(&label_count, filter_data, sizeof(uint32_t));
      std::vector<uint32_t> labels(label_count);
      if (label_count > 0) {
        memcpy(labels.data(), static_cast<const char *>(filter_data) + sizeof(uint32_t),
               static_cast<size_t>(label_count) * sizeof(uint32_t));
      }
      return labels;
    }

    uint64_t popcount_words(const std::vector<uint64_t> &words) {
      uint64_t total = 0;
      for (const uint64_t word : words) {
        total += static_cast<uint64_t>(__builtin_popcountll(word));
      }
      return total;
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

    bool labels_match(HybridFilterKind filter_kind, const void *query_labels,
                      const std::vector<uint32_t> *target_labels) {
      uint32_t query_count = 0;
      if (query_labels != nullptr) {
        memcpy(&query_count, query_labels, sizeof(uint32_t));
      }
      const uint32_t target_count = target_labels == nullptr ? 0U : static_cast<uint32_t>(target_labels->size());

      if (filter_kind == HybridFilterKind::kIntersect) {
        if (query_count == 0 || target_count == 0) {
          return false;
        }
      } else if (filter_kind == HybridFilterKind::kSubset) {
        if (query_count == 0) {
          return true;
        }
        if (target_count == 0) {
          return false;
        }
      } else if (filter_kind == HybridFilterKind::kRange) {
        if (query_count == 0 || target_count == 0) {
          return false;
        }
      } else {
        return false;
      }

      std::vector<uint32_t> query_labels_vec(query_count);
      if (query_count > 0) {
        memcpy(query_labels_vec.data(), static_cast<const char *>(query_labels) + sizeof(uint32_t),
               static_cast<size_t>(query_count) * sizeof(uint32_t));
      }

      if (filter_kind == HybridFilterKind::kIntersect) {
        for (uint32_t query_label : query_labels_vec) {
          for (uint32_t target_label : *target_labels) {
            if (query_label == target_label) {
              return true;
            }
          }
        }
        return false;
      }

      if (filter_kind == HybridFilterKind::kRange) {
        std::sort(query_labels_vec.begin(), query_labels_vec.end());
        const uint32_t low = query_labels_vec.front();
        const uint32_t high = query_labels_vec.back();
        for (uint32_t target_label : *target_labels) {
          if (target_label >= low && target_label <= high) {
            return true;
          }
        }
        return false;
      }

      for (uint32_t query_label : query_labels_vec) {
        bool found = false;
        for (uint32_t target_label : *target_labels) {
          if (query_label == target_label) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    }

    struct LiveLabelSelector : public AbstractSelector {
      std::shared_timed_mutex *live_state_lock = nullptr;
      const std::unordered_map<uint32_t, std::vector<uint32_t>> *live_labels_by_id = nullptr;
      HybridFilterKind filter_kind = HybridFilterKind::kUnsupported;

      bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
        (void) target_labels;
        std::shared_lock<std::shared_timed_mutex> lock(*live_state_lock);
        auto iter = live_labels_by_id->find(target_id);
        const std::vector<uint32_t> *labels = iter == live_labels_by_id->end() ? nullptr : &iter->second;
        return labels_match(filter_kind, query_labels, labels);
      }
    };

    uint64_t compute_live_candidate_bitset(HybridFilterKind filter_kind, const std::vector<uint32_t> &labels,
                                           const std::unordered_map<uint32_t, std::vector<uint64_t>> &label_bitsets,
                                           const std::vector<uint64_t> &live_present_bitset,
                                           uint64_t words_per_label, HybridQueryScratch *scratch) {
      if (scratch == nullptr) {
        return 0;
      }

      scratch->normalized_labels = labels;
      normalize_labels(&scratch->normalized_labels);
      scratch->bitset_words.assign(static_cast<size_t>(words_per_label), 0ULL);

      if (filter_kind == HybridFilterKind::kUnsupported) {
        return 0;
      }

      if (filter_kind == HybridFilterKind::kIntersect) {
        if (scratch->normalized_labels.empty()) {
          return 0;
        }
        for (uint32_t label : scratch->normalized_labels) {
          auto iter = label_bitsets.find(label);
          if (iter == label_bitsets.end()) {
            continue;
          }
          for (size_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
            scratch->bitset_words[word_idx] |= iter->second[word_idx];
          }
        }
      } else if (filter_kind == HybridFilterKind::kSubset) {
        if (scratch->normalized_labels.empty()) {
          scratch->bitset_words = live_present_bitset;
        } else {
          bool first_label = true;
          for (uint32_t label : scratch->normalized_labels) {
            auto iter = label_bitsets.find(label);
            if (iter == label_bitsets.end()) {
              std::fill(scratch->bitset_words.begin(), scratch->bitset_words.end(), 0ULL);
              return 0;
            }
            if (first_label) {
              scratch->bitset_words = iter->second;
              first_label = false;
            } else {
              for (size_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
                scratch->bitset_words[word_idx] &= iter->second[word_idx];
              }
            }
          }
        }
      } else if (filter_kind == HybridFilterKind::kRange) {
        if (scratch->normalized_labels.empty()) {
          return 0;
        }
        const uint32_t low = scratch->normalized_labels.front();
        const uint32_t high = scratch->normalized_labels.back();
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1ULL;
        const uint64_t sparse_scan_cutoff = static_cast<uint64_t>(label_bitsets.size()) * 2ULL + 64ULL;
        if (span > sparse_scan_cutoff) {
          for (const auto &entry : label_bitsets) {
            if (entry.first < low || entry.first > high) {
              continue;
            }
            for (size_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
              scratch->bitset_words[word_idx] |= entry.second[word_idx];
            }
          }
        } else {
          for (uint32_t label = low; label <= high; ++label) {
            auto iter = label_bitsets.find(label);
            if (iter != label_bitsets.end()) {
              for (size_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
                scratch->bitset_words[word_idx] |= iter->second[word_idx];
              }
            }
            if (label == std::numeric_limits<uint32_t>::max()) {
              break;
            }
          }
        }
      }

      return popcount_words(scratch->bitset_words);
    }

    void materialize_candidate_ids(HybridQueryScratch *scratch, std::vector<uint32_t> *output) {
      output->clear();
      output->reserve(static_cast<size_t>(popcount_words(scratch->bitset_words)));
      for (size_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
        uint64_t word = scratch->bitset_words[word_idx];
        while (word != 0) {
          const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
          output->push_back(static_cast<uint32_t>(word_idx * 64ULL + bit));
          word &= (word - 1);
        }
      }
    }

    template<typename TagT>
    void clear_result_buffers(uint64_t k_search, TagT *res_tags, float *res_dists) {
      for (uint64_t idx = 0; idx < k_search; ++idx) {
        if (res_tags != nullptr) {
          res_tags[idx] = std::numeric_limits<TagT>::max();
        }
        if (res_dists != nullptr) {
          res_dists[idx] = std::numeric_limits<float>::infinity();
        }
      }
    }

    template<typename TagT>
    size_t filter_deleted_results(size_t raw_count, const std::vector<TagT> &raw_tags,
                                  const std::vector<float> &raw_distances, uint64_t k_search, TagT *tags,
                                  float *distances, std::shared_timed_mutex &delete_lock,
                                  tsl::robin_set<TagT> (&deletion_sets)[2]) {
      clear_result_buffers(k_search, tags, distances);

      std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
      size_t pos = 0;
      for (size_t idx = 0; idx < raw_count && pos < static_cast<size_t>(k_search); ++idx) {
        const TagT tag = raw_tags[idx];
        if (deletion_sets[0].find(tag) != deletion_sets[0].end()
            || deletion_sets[1].find(tag) != deletion_sets[1].end()) {
          continue;
        }
        tags[pos] = tag;
        if (distances != nullptr) {
          distances[pos] = raw_distances[idx];
        }
        ++pos;
      }
      return pos;
    }

    void write_spmat_labels(const std::string &path, uint64_t nrows, uint64_t nlabels,
                            const std::vector<std::vector<uint32_t>> &labels_by_row) {
      std::ofstream writer(path, std::ios::binary | std::ios::out | std::ios::trunc);
      if (!writer.is_open()) {
        LOG(ERROR) << "Failed to open flat labels spmat for write: " << path;
        crash();
      }

      int64_t nnz = 0;
      for (const auto &labels : labels_by_row) {
        nnz += static_cast<int64_t>(labels.size());
      }

      const int64_t nrow_i64 = static_cast<int64_t>(nrows);
      const int64_t ncol_i64 = static_cast<int64_t>(nlabels);
      writer.write(reinterpret_cast<const char *>(&nrow_i64), sizeof(int64_t));
      writer.write(reinterpret_cast<const char *>(&ncol_i64), sizeof(int64_t));
      writer.write(reinterpret_cast<const char *>(&nnz), sizeof(int64_t));

      std::vector<int64_t> indptr(static_cast<size_t>(nrows) + 1, 0);
      std::vector<int32_t> indices;
      std::vector<float> data;
      indices.reserve(static_cast<size_t>(nnz));
      data.reserve(static_cast<size_t>(nnz));
      for (uint64_t row = 0; row < nrows; ++row) {
        const auto &labels = labels_by_row[static_cast<size_t>(row)];
        indptr[static_cast<size_t>(row) + 1] = indptr[static_cast<size_t>(row)] + static_cast<int64_t>(labels.size());
        for (uint32_t label : labels) {
          indices.push_back(static_cast<int32_t>(label));
          data.push_back(1.0f);
        }
      }

      writer.write(reinterpret_cast<const char *>(indptr.data()),
                   static_cast<std::streamsize>(indptr.size() * sizeof(int64_t)));
      if (!indices.empty()) {
        writer.write(reinterpret_cast<const char *>(indices.data()),
                     static_cast<std::streamsize>(indices.size() * sizeof(int32_t)));
        writer.write(reinterpret_cast<const char *>(data.data()),
                     static_cast<std::streamsize>(data.size() * sizeof(float)));
      }
      writer.close();
    }
  }  // namespace

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            pipeann::Metric dist_metric, int search_mode, bool use_mem_index) {
    // check if file exists.
    if (!file_exists(disk_prefix_in + "_disk.index")) {
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    if (use_mem_index && !file_exists(disk_prefix_in + "_mem.index")) {
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in << "_mem.index";
      exit(-1);
    }

    this->_dist_metric = dist_metric;
    this->journal = new pipeann::Journal<TagT>(disk_prefix_out + "_journal");

    _paras_disk = parameters;
    _num_threads = parameters.num_threads;
    _beamwidth = parameters.beam_width;

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    reader.reset(new LinuxAlignedFileReader());
    AbstractNeighbor<T> *nbr_handler = new PQNeighbor<T>(this->_dist_metric);
    _disk_index = new pipeann::SSDIndex<T, TagT>(this->_dist_metric, reader, nbr_handler, true, &_paras_disk);

#ifndef NO_POLLUTE_ORIGINAL
    std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
    _disk_index->copy_index(_disk_index_prefix_in, disk_index_prefix_shadow);
    LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
    _disk_index_prefix_in = disk_index_prefix_shadow;
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH) {
      this->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, or PIPE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, use_page_search);
    if (res != 0) {
      LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
      exit(-1);
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + "_mem.index";  // use the original one.
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      _disk_index->load_mem_index(mem_index_path);
    }

    initialize_live_state_from_disk(_disk_index_prefix_in);
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_out,
                                            uint32_t data_dim, Distance<T> *dist, pipeann::Metric dist_metric,
                                            uint64_t flat_threshold, int search_mode, uint32_t flat_pq_bytes,
                                            const std::string &flat_pq_pivots_path,
                                            uint32_t flat_build_memory_gb) {
    this->_dist_metric = dist_metric;
    this->journal = new pipeann::Journal<TagT>(disk_prefix_out + "_journal");

    _paras_disk = parameters;
    _num_threads = parameters.num_threads == 0 ? 1 : parameters.num_threads;
    _beamwidth = parameters.beam_width;
    _disk_index_prefix_in = disk_prefix_out;
    _disk_index_prefix_out = disk_prefix_out + "_merge";
    _dist_comp = dist;

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH) {
      this->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, or PIPE_SEARCH.";
      exit(-1);
    }

    reader.reset(new LinuxAlignedFileReader());
    _disk_index = nullptr;
    flat_mode_ = true;
    flat_threshold_ = flat_threshold;
    flat_pq_bytes_ = std::max<uint32_t>(1, std::min<uint32_t>(flat_pq_bytes, data_dim));
    flat_build_memory_gb_ = std::max<uint32_t>(1, flat_build_memory_gb);
    flat_pq_pivots_path_ = flat_pq_pivots_path;
    flat_dim_ = data_dim;
    live_point_count_.store(0);
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    stop_hybrid_recalibration_worker();
    delete journal;
    journal = nullptr;
    delete _disk_index;
    _disk_index = nullptr;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::configure_hybrid_recalibration(HybridRecalibrationConfig config) {
    {
      std::lock_guard<std::mutex> lock(hybrid_recalibration_config_lock_);
      foreground_budget_ = config.foreground_budget;
      hybrid_recalibration_config_ = std::move(config);
    }
    ensure_hybrid_recalibration_worker_started();
    maybe_mark_hybrid_recalibration_pending();
    notify_hybrid_recalibration_worker();
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::set_background_recalibration_disabled(bool disabled) {
    foreground_counters_.background_recalibration_disabled.store(disabled);
    if (!disabled) {
      ensure_hybrid_recalibration_worker_started();
    }
    notify_hybrid_recalibration_worker();
  }

  template<typename T, typename TagT>
  size_t DynamicSSDIndex<T, TagT>::flat_exact_search_locked(const T *query, uint64_t k_search,
                                                            const std::vector<uint32_t> *candidate_ids, TagT *tags,
                                                            float *distances, QueryStats *stats) const {
    clear_result_buffers(k_search, tags, distances);
    if (k_search == 0 || live_point_count_.load() == 0) {
      if (stats != nullptr) {
        *stats = {};
      }
      return 0;
    }
    if (_dist_comp == nullptr) {
      LOG(ERROR) << "Flat exact search requires a distance comparator";
      crash();
    }

    std::vector<T> normalized_query(static_cast<size_t>(flat_dim_));
    if (_dist_metric == pipeann::Metric::COSINE) {
      pipeann::normalize_data(normalized_query.data(), query, flat_dim_);
      query = normalized_query.data();
    }

    Timer query_timer;
    std::vector<std::pair<float, uint32_t>> results;
    const size_t reserve_size =
        candidate_ids == nullptr ? static_cast<size_t>(live_point_count_.load()) : candidate_ids->size();
    results.reserve(reserve_size);

    auto consider_id = [&](uint32_t id) {
      if (id >= flat_tags_.size() || id >= flat_deleted_.size() || flat_deleted_[id] != 0) {
        return;
      }
      const TagT tag = flat_tags_[id];
      auto live_iter = live_ids_by_tag_.find(tag);
      if (live_iter == live_ids_by_tag_.end() || live_iter->second != id) {
        return;
      }
      const T *point = flat_data_.data() + static_cast<size_t>(id) * flat_dim_;
      const float dist = _dist_comp->compare(query, point, flat_dim_);
      results.emplace_back(dist, id);
    };

    if (candidate_ids != nullptr) {
      for (uint32_t id : *candidate_ids) {
        consider_id(id);
      }
    } else {
      for (uint32_t id = 0; id < flat_tags_.size(); ++id) {
        consider_id(id);
      }
    }

    const size_t result_count = std::min(static_cast<size_t>(k_search), results.size());
    if (result_count > 0) {
      std::partial_sort(results.begin(), results.begin() + result_count, results.end());
    }
    for (size_t idx = 0; idx < result_count; ++idx) {
      const uint32_t id = results[idx].second;
      tags[idx] = flat_tags_[id];
      if (distances != nullptr) {
        distances[idx] = results[idx].first;
      }
    }

    if (stats != nullptr) {
      *stats = {};
      stats->total_us = static_cast<double>(query_timer.elapsed());
      stats->n_cmps = static_cast<double>(results.size());
      stats->n_ios = 0.0;
      stats->n_hops = 0.0;
    }
    return result_count;
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::materialize_flat_to_disk() {
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);
    return materialize_flat_to_disk_locked();
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::materialize_flat_to_disk_locked() {
    if (!flat_mode_) {
      return true;
    }
    const uint64_t live_count = live_point_count_.load();
    if (live_count == 0) {
      return false;
    }

    std::vector<std::pair<uint32_t, TagT>> live_entries;
    live_entries.reserve(static_cast<size_t>(live_count));
    for (uint32_t id = 0; id < flat_tags_.size(); ++id) {
      if (id >= flat_deleted_.size() || flat_deleted_[id] != 0) {
        continue;
      }
      const TagT tag = flat_tags_[id];
      auto live_iter = live_ids_by_tag_.find(tag);
      if (live_iter != live_ids_by_tag_.end() && live_iter->second == id) {
        live_entries.emplace_back(id, tag);
      }
    }
    if (live_entries.empty()) {
      return false;
    }

    const std::string data_path = _disk_index_prefix_in + "_flat_build_data.bin";
    const std::string tag_path = _disk_index_prefix_in + "_flat_build_tags.bin";
    const std::string label_path = _disk_index_prefix_in + "_flat_build_labels.spmat";

    std::vector<T> build_data(live_entries.size() * static_cast<size_t>(flat_dim_));
    std::vector<TagT> build_tags(live_entries.size());
    std::vector<std::vector<uint32_t>> labels_by_row(live_entries.size());
    bool has_labels = false;
    uint64_t label_universe = 0;

    for (size_t row = 0; row < live_entries.size(); ++row) {
      const uint32_t old_id = live_entries[row].first;
      const TagT tag = live_entries[row].second;
      std::memcpy(build_data.data() + row * flat_dim_, flat_data_.data() + static_cast<size_t>(old_id) * flat_dim_,
                  static_cast<size_t>(flat_dim_) * sizeof(T));
      build_tags[row] = tag;
      auto label_iter = live_labels_by_tag_.find(tag);
      if (label_iter != live_labels_by_tag_.end()) {
        labels_by_row[row] = label_iter->second;
        if (!labels_by_row[row].empty()) {
          has_labels = true;
          label_universe = std::max<uint64_t>(label_universe, static_cast<uint64_t>(labels_by_row[row].back()) + 1ULL);
        }
      }
    }

    pipeann::save_bin<T>(data_path, build_data.data(), build_tags.size(), flat_dim_);
    pipeann::save_bin<TagT>(tag_path, build_tags.data(), build_tags.size(), 1);

    std::unique_ptr<pipeann::SpmatLabel> label;
    const char *label_source_file = nullptr;
    if (has_labels) {
      write_spmat_labels(label_path, build_tags.size(), label_universe, labels_by_row);
      label = std::make_unique<pipeann::SpmatLabel>(label_path);
      label_source_file = label_path.c_str();
    }

    PQNeighbor<T> *pq_nbr_handler = new PQNeighbor<T>(this->_dist_metric);
    if (!flat_pq_pivots_path_.empty()) {
      pq_nbr_handler->set_pretrained_pq_pivots(flat_pq_pivots_path_);
    }
    AbstractNeighbor<T> *nbr_handler = pq_nbr_handler;
    const uint32_t R = _paras_disk.R == 0 ? 64 : _paras_disk.R;
    const uint32_t L = _paras_disk.L == 0 ? R + 32 : _paras_disk.L;
    const uint32_t build_threads = _num_threads == 0 ? 1 : _num_threads;
    const uint32_t build_memory_gb = std::max<uint32_t>(1, flat_build_memory_gb_);
    const uint32_t pq_bytes = std::max<uint32_t>(1, std::min<uint32_t>(flat_pq_bytes_, flat_dim_));
    const bool build_ok = pipeann::build_disk_index<T, TagT>(
        data_path.c_str(), _disk_index_prefix_in.c_str(), R, L, build_memory_gb, build_threads, pq_bytes, _dist_metric,
        tag_path.c_str(), nbr_handler, nullptr, label_source_file);
    if (!build_ok) {
      delete nbr_handler;
      return false;
    }

    delete _disk_index;
    _disk_index = new pipeann::SSDIndex<T, TagT>(this->_dist_metric, reader, nbr_handler, true, &_paras_disk);
    if (_disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, search_mode == PAGE_SEARCH) != 0) {
      LOG(ERROR) << "Failed to load disk index after flat materialization";
      exit(-1);
    }

    initialize_live_state_from_disk(_disk_index_prefix_in);

    flat_data_.clear();
    flat_tags_.clear();
    flat_deleted_.clear();
    flat_mode_ = false;
    {
      std::unique_lock<std::shared_timed_mutex> delete_guard(delete_lock);
      deletion_sets[0].clear();
      deletion_sets[1].clear();
      deleted_tags[0].clear();
      deleted_tags[1].clear();
    }

    std::remove(data_path.c_str());
    std::remove(tag_path.c_str());
    if (has_labels) {
      std::remove(label_path.c_str());
    }
    return true;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::ensure_live_filter_capacity_locked(uint32_t id) {
    const uint64_t required_words = dense_words_per_label(static_cast<uint64_t>(id) + 1ULL);
    if (required_words <= live_densebit_words_per_label_) {
      return;
    }

    for (auto &entry : live_label_bitsets_) {
      entry.second.resize(static_cast<size_t>(required_words), 0ULL);
    }
    live_present_bitset_.resize(static_cast<size_t>(required_words), 0ULL);
    live_densebit_words_per_label_ = required_words;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::apply_live_labels_locked(uint32_t id, const std::vector<uint32_t> &labels,
                                                          bool add_labels) {
    if (labels.empty()) {
      return;
    }

    if (add_labels) {
      ensure_live_filter_capacity_locked(id);
      live_label_universe_ = std::max<uint64_t>(live_label_universe_, static_cast<uint64_t>(labels.back()) + 1ULL);
    }

    const size_t word_idx = static_cast<size_t>(id >> 6U);
    const uint64_t bit_mask = 1ULL << (id & 63U);
    for (uint32_t label : labels) {
      if (add_labels) {
        auto &bitset = live_label_bitsets_[label];
        if (bitset.size() < static_cast<size_t>(live_densebit_words_per_label_)) {
          bitset.resize(static_cast<size_t>(live_densebit_words_per_label_), 0ULL);
        }
        bitset[word_idx] |= bit_mask;
      } else {
        auto iter = live_label_bitsets_.find(label);
        if (iter != live_label_bitsets_.end() && word_idx < iter->second.size()) {
          iter->second[word_idx] &= ~bit_mask;
        }
      }
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::rebuild_live_filter_state_locked() {
    live_labels_by_id_.clear();
    live_label_bitsets_.clear();
    live_present_bitset_.clear();
    live_densebit_words_per_label_ = 0;
    live_label_universe_ = 0;

    uint32_t max_live_id = 0;
    bool has_live_points = false;
    for (const auto &entry : live_ids_by_tag_) {
      max_live_id = has_live_points ? std::max(max_live_id, entry.second) : entry.second;
      has_live_points = true;
    }

    if (!has_live_points) {
      return;
    }

    live_densebit_words_per_label_ = dense_words_per_label(static_cast<uint64_t>(max_live_id) + 1ULL);
    live_present_bitset_.assign(static_cast<size_t>(live_densebit_words_per_label_), 0ULL);

    for (const auto &entry : live_ids_by_tag_) {
      const TagT &tag = entry.first;
      const uint32_t id = entry.second;
      const size_t word_idx = static_cast<size_t>(id >> 6U);
      const uint64_t bit_mask = 1ULL << (id & 63U);
      live_present_bitset_[word_idx] |= bit_mask;

      auto label_iter = live_labels_by_tag_.find(tag);
      if (label_iter == live_labels_by_tag_.end()) {
        continue;
      }
      live_labels_by_id_[id] = label_iter->second;
      apply_live_labels_locked(id, label_iter->second, true);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::checkpoint() {
    // TODO(gh): checkpoint the index.
    journal->checkpoint();
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);
    if (flat_mode_) {
      return;
    }
    if (!persist_live_hybrid_state(_disk_index_prefix_in)) {
      LOG(INFO) << "Skip hybrid sidecar checkpoint because live IDs are not compact for " << _disk_index_prefix_in;
    }
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> recalibration_mutation_guard(hybrid_recalibration_mutation_lock_);
    if (flat_mode_) {
      std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);
      if (flat_mode_) {
        journal->append(pipeann::TxType::kInsert, tag);

        auto existing = live_ids_by_tag_.find(tag);
        uint32_t target_id = 0;
        if (existing != live_ids_by_tag_.end()) {
          target_id = existing->second;
        } else {
          target_id = static_cast<uint32_t>(flat_tags_.size());
          flat_tags_.push_back(tag);
          flat_deleted_.push_back(0);
          flat_data_.resize(static_cast<size_t>(target_id + 1) * flat_dim_);
        }

        T *dst = flat_data_.data() + static_cast<size_t>(target_id) * flat_dim_;
        if (_dist_metric == pipeann::Metric::COSINE) {
          pipeann::normalize_data(dst, point, flat_dim_);
        } else {
          std::memcpy(dst, point, static_cast<size_t>(flat_dim_) * sizeof(T));
        }

        {
          std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
          live_ids_by_tag_[tag] = target_id;
          flat_deleted_[target_id] = 0;
          ensure_live_filter_capacity_locked(target_id);
          live_present_bitset_[static_cast<size_t>(target_id >> 6U)] |= (1ULL << (target_id & 63U));
          live_labels_by_id_.erase(target_id);
          auto label_iter = live_labels_by_tag_.find(tag);
          if (label_iter != live_labels_by_tag_.end()) {
            live_labels_by_id_[target_id] = label_iter->second;
            apply_live_labels_locked(target_id, label_iter->second, true);
          }
          live_point_count_.store(live_ids_by_tag_.size());
        }

        deletion_sets[0].erase(tag);
        deletion_sets[1].erase(tag);

        if (flat_threshold_ != 0 && live_point_count_.load() > flat_threshold_) {
          materialize_flat_to_disk_locked();
        }
        return static_cast<int>(target_id);
      }
    }

    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    journal->append(pipeann::TxType::kInsert, tag);
    auto *deletion_set = &deletion_sets[active_delete_set];
    const int target_id = _disk_index->insert_in_place(point, tag, deletion_set);
    if (target_id >= 0) {
      std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
      live_ids_by_tag_[tag] = static_cast<uint32_t>(target_id);
      ensure_live_filter_capacity_locked(static_cast<uint32_t>(target_id));
      live_present_bitset_[static_cast<size_t>(target_id >> 6U)] |= (1ULL << (target_id & 63U));
      live_labels_by_id_.erase(static_cast<uint32_t>(target_id));
      live_point_count_.store(live_ids_by_tag_.size());
    }
    maybe_mark_hybrid_recalibration_pending();
    return target_id;
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::hybrid_recalibration_configured_locked() const {
    return !hybrid_recalibration_config_.datasets.empty()
        && hybrid_recalibration_config_.k != 0
        && hybrid_recalibration_config_.beamwidth != 0
        && hybrid_recalibration_config_.l_search >= hybrid_recalibration_config_.k;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::update_hybrid_recalibration_metadata(uint64_t live_count, bool pending, bool running) {
    const std::string meta_path = HybridMetadata::default_metadata_path(_disk_index_prefix_in);
    if (!file_exists(meta_path)) {
      return;
    }

    try {
      std::lock_guard<std::mutex> metadata_lock(hybrid_metadata_io_lock_);
      auto metadata = HybridMetadata::load(meta_path, false);
      metadata->set_n_live_snapshot(live_count);
      metadata->set_recalibration_flags(pending, running);
      metadata->write_atomically(meta_path);
    } catch (const std::exception &e) {
      LOG(WARNING) << "Failed to update hybrid recalibration metadata state: " << e.what();
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::maybe_mark_hybrid_recalibration_pending() {
    if (flat_mode_) {
      return;
    }
    if (hybrid_recalibration_state_.load() != HybridRecalibrationState::kIdle) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(hybrid_recalibration_config_lock_);
      if (!hybrid_recalibration_configured_locked()) {
        return;
      }
    }

    const std::string meta_path = HybridMetadata::default_metadata_path(_disk_index_prefix_in);
    if (!file_exists(meta_path)) {
      return;
    }

    try {
      const uint64_t live_count = live_point_count_.load();
      uint64_t n_calib = 0;
      {
        std::lock_guard<std::mutex> metadata_lock(hybrid_metadata_io_lock_);
        const auto metadata = HybridMetadata::load(meta_path, false);
        n_calib = metadata->header().n_calib;
      }

      if (!exceeds_recalibration_drift(n_calib, live_count)) {
        return;
      }

      HybridRecalibrationState expected = HybridRecalibrationState::kIdle;
      if (!hybrid_recalibration_state_.compare_exchange_strong(expected, HybridRecalibrationState::kPending)) {
        return;
      }
      update_hybrid_recalibration_metadata(live_count, true, false);
      ensure_hybrid_recalibration_worker_started();
      notify_hybrid_recalibration_worker();
    } catch (const std::exception &e) {
      LOG(WARNING) << "Failed to evaluate hybrid recalibration trigger: " << e.what();
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::notify_hybrid_recalibration_worker() {
    {
      std::lock_guard<std::mutex> lock(hybrid_recalibration_worker_lock_);
      ++hybrid_recalibration_signal_count_;
    }
    hybrid_recalibration_cv_.notify_all();
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::ensure_hybrid_recalibration_worker_started() {
    std::lock_guard<std::mutex> lock(hybrid_recalibration_worker_lock_);
    if (hybrid_recalibration_worker_started_) {
      return;
    }
    hybrid_recalibration_worker_stop_ = false;
    hybrid_recalibration_worker_started_ = true;
    hybrid_recalibration_worker_ = std::thread(&DynamicSSDIndex<T, TagT>::hybrid_recalibration_worker_loop, this);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::stop_hybrid_recalibration_worker() {
    {
      std::lock_guard<std::mutex> lock(hybrid_recalibration_worker_lock_);
      if (!hybrid_recalibration_worker_started_) {
        return;
      }
      hybrid_recalibration_worker_stop_ = true;
      ++hybrid_recalibration_signal_count_;
    }
    hybrid_recalibration_cv_.notify_all();
    if (hybrid_recalibration_worker_.joinable()) {
      hybrid_recalibration_worker_.join();
    }
    std::lock_guard<std::mutex> lock(hybrid_recalibration_worker_lock_);
    hybrid_recalibration_worker_started_ = false;
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::can_run_hybrid_recalibration_now() const {
    return !foreground_counters_.background_recalibration_disabled.load()
        && foreground_counters_.active_queries.load() <= foreground_budget_.active_queries_low_watermark
        && foreground_counters_.waiting_queries.load() <= foreground_budget_.waiting_queries_low_watermark
        && foreground_counters_.active_high_priority_tasks.load() == 0;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::hybrid_recalibration_worker_loop() {
    uint64_t observed_signal_count = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(hybrid_recalibration_worker_lock_);
        hybrid_recalibration_cv_.wait(lock, [&] {
          return hybrid_recalibration_worker_stop_ || hybrid_recalibration_signal_count_ != observed_signal_count;
        });
        if (hybrid_recalibration_worker_stop_) {
          return;
        }
        observed_signal_count = hybrid_recalibration_signal_count_;
      }

      if (hybrid_recalibration_state_.load() != HybridRecalibrationState::kPending) {
        continue;
      }
      if (!can_run_hybrid_recalibration_now()) {
        continue;
      }

      HybridRecalibrationState expected = HybridRecalibrationState::kPending;
      if (!hybrid_recalibration_state_.compare_exchange_strong(expected, HybridRecalibrationState::kRunning)) {
        continue;
      }

      update_hybrid_recalibration_metadata(live_point_count_.load(), false, true);
      const bool completed = run_hybrid_recalibration_once();
      if (completed) {
        hybrid_recalibration_state_.store(HybridRecalibrationState::kIdle);
        maybe_mark_hybrid_recalibration_pending();
      } else {
        hybrid_recalibration_state_.store(HybridRecalibrationState::kPending);
        update_hybrid_recalibration_metadata(live_point_count_.load(), true, false);
      }
    }
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::run_hybrid_recalibration_once() {
    HybridRecalibrationConfig config;
    {
      std::lock_guard<std::mutex> lock(hybrid_recalibration_config_lock_);
      if (!hybrid_recalibration_configured_locked()) {
        return false;
      }
      config = hybrid_recalibration_config_;
    }

    std::unique_lock<std::shared_timed_mutex> mutation_pause_lock(hybrid_recalibration_mutation_lock_);
    if (!can_run_hybrid_recalibration_now()) {
      return false;
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> label_bitsets_snapshot;
    std::unordered_map<uint32_t, std::vector<uint32_t>> labels_by_id_snapshot;
    std::vector<uint64_t> live_present_bitset_snapshot;
    uint64_t live_densebit_words_per_label_snapshot = 0;
    uint64_t live_point_count_snapshot = 0;
    {
      std::shared_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
      label_bitsets_snapshot = live_label_bitsets_;
      labels_by_id_snapshot = live_labels_by_id_;
      live_present_bitset_snapshot = live_present_bitset_;
      live_densebit_words_per_label_snapshot = live_densebit_words_per_label_;
      live_point_count_snapshot = live_point_count_.load();
    }

    const std::string meta_path = HybridMetadata::default_metadata_path(_disk_index_prefix_in);
    if (!file_exists(meta_path)) {
      return false;
    }

    std::map<uint64_t, CalibrationBucketSamples<TagT>> bucket_samples;
    uint64_t route_selector_mask = 0;
    uint64_t sampled_query_count = 0;

    std::shared_timed_mutex labels_snapshot_lock;
    LiveLabelSelector selector;
    selector.live_state_lock = &labels_snapshot_lock;
    selector.live_labels_by_id = &labels_by_id_snapshot;

    for (const auto &dataset : config.datasets) {
      if (!can_run_hybrid_recalibration_now()) {
        return false;
      }

      const char *selector_name = selector_name_for_kind(dataset.filter_kind);
      if (selector_name == nullptr) {
        continue;
      }
      selector.filter_kind = dataset.filter_kind;

      pipeann::SpmatLabel query_labels(dataset.query_label_file);
      std::unique_ptr<T[]> queries;
      size_t query_num = 0, query_dim = 0;
      pipeann::load_bin<T>(dataset.query_bin, queries, query_num, query_dim);
      if (query_labels.labels_.size() != query_num || query_dim != _disk_index->meta_.data_dim) {
        LOG(WARNING) << "Skipping invalid recalibration dataset: " << dataset.query_bin;
        return false;
      }

      const size_t max_filter_size = query_labels.label_size();
      std::vector<std::vector<char>> filter_buffers(query_num);
      for (size_t query_idx = 0; query_idx < query_num; ++query_idx) {
        filter_buffers[query_idx].resize(max_filter_size, 0);
        query_labels.write(query_idx, filter_buffers[query_idx].data());
      }

      std::vector<size_t> sampled_query_ids(query_num);
      std::iota(sampled_query_ids.begin(), sampled_query_ids.end(), 0);
      std::mt19937_64 rng(kCalibrationSeed);
      std::shuffle(sampled_query_ids.begin(), sampled_query_ids.end(), rng);
      if (dataset.sample_limit > 0 && dataset.sample_limit < sampled_query_ids.size()) {
        sampled_query_ids.resize(dataset.sample_limit);
      }
      std::sort(sampled_query_ids.begin(), sampled_query_ids.end());

      route_selector_mask |= selector_mask_for_kind(dataset.filter_kind);
      sampled_query_count += sampled_query_ids.size();

      std::vector<TagT> graph_tags(static_cast<size_t>(config.k), std::numeric_limits<TagT>::max());
      std::vector<float> graph_dists(static_cast<size_t>(config.k), std::numeric_limits<float>::infinity());
      std::vector<TagT> prefilter_tags(static_cast<size_t>(config.k), std::numeric_limits<TagT>::max());
      std::vector<float> prefilter_dists(static_cast<size_t>(config.k), std::numeric_limits<float>::infinity());

      for (size_t query_idx : sampled_query_ids) {
        if (!can_run_hybrid_recalibration_now()) {
          return false;
        }

        HybridQueryScratch scratch;
        const auto &labels = query_labels.labels_[query_idx];
        const uint64_t candidate_count = compute_live_candidate_bitset(dataset.filter_kind, labels,
                                                                       label_bitsets_snapshot,
                                                                       live_present_bitset_snapshot,
                                                                       live_densebit_words_per_label_snapshot,
                                                                       &scratch);
        if (candidate_count == 0) {
          continue;
        }

        std::vector<uint32_t> candidate_ids;
        materialize_candidate_ids(&scratch, &candidate_ids);

        std::fill(graph_tags.begin(), graph_tags.end(), std::numeric_limits<TagT>::max());
        std::fill(graph_dists.begin(), graph_dists.end(), std::numeric_limits<float>::infinity());
        std::fill(prefilter_tags.begin(), prefilter_tags.end(), std::numeric_limits<TagT>::max());
        std::fill(prefilter_dists.begin(), prefilter_dists.end(), std::numeric_limits<float>::infinity());

        QueryStats graph_stats{};
        QueryStats prefilter_stats{};
        _disk_index->pipe_search(queries.get() + (query_idx * query_dim), config.k, config.mem_L, config.l_search,
                                 graph_tags.data(), graph_dists.data(), config.beamwidth, &graph_stats, &selector,
                                 filter_buffers[query_idx].data(), 0);
        _disk_index->hybrid_prefilter_search(queries.get() + (query_idx * query_dim), config.k, prefilter_tags.data(),
                                             prefilter_dists.data(), candidate_ids, &prefilter_stats,
                                             live_point_count_snapshot);

        auto &bucket = bucket_samples[next_power_of_two(candidate_count)];
        bucket.prefilter_us.push_back(prefilter_stats.total_us);
        bucket.graph_us.push_back(graph_stats.total_us);
      }
    }

    std::vector<HybridCalibrationBucketV1> buckets;
    buckets.reserve(bucket_samples.size());
    uint64_t tau_m = 0;
    for (const auto &entry : bucket_samples) {
      HybridCalibrationBucketV1 bucket;
      bucket.candidate_upper_bound = entry.first;
      bucket.query_count = static_cast<uint64_t>(entry.second.prefilter_us.size());
      bucket.prefilter_p50_us = p50_as_uint64(entry.second.prefilter_us);
      bucket.graph_p50_us = p50_as_uint64(entry.second.graph_us);
      buckets.push_back(bucket);
      if (bucket.query_count >= 8 && bucket.prefilter_p50_us <= bucket.graph_p50_us) {
        tau_m = bucket.candidate_upper_bound;
      }
    }

    try {
      std::lock_guard<std::mutex> metadata_lock(hybrid_metadata_io_lock_);
      const auto existing_metadata = HybridMetadata::load(meta_path, false);
      HybridMetadataHeaderV1 header = existing_metadata->header();
      header.flags = kMetadataValidFlag | kCalibrationValidFlag | kMetadataAllowPrefilterFlag;
      header.route_selector_mask = route_selector_mask;
      header.tau_m = tau_m;
      header.n_calib = live_point_count_snapshot;
      header.n_live_snapshot = live_point_count_snapshot;
      header.threshold_version = existing_metadata->header().threshold_version + 1;
      header.calib_epoch_sec = static_cast<uint64_t>(std::time(nullptr));
      header.calib_query_count = sampled_query_count;
      header.calib_bucket_count = static_cast<uint64_t>(buckets.size());
      header.calib_k = config.k;
      header.calib_mem_L = config.mem_L;
      header.calib_beamwidth = config.beamwidth;
      header.calib_l_search = config.l_search;

      auto metadata = HybridMetadata::create(header, buckets);
      metadata->write_atomically(meta_path);
      _disk_index->load_hybrid_runtime(_disk_index_prefix_in.c_str());
    } catch (const std::exception &e) {
      LOG(WARNING) << "Failed to publish recalibrated hybrid metadata: " << e.what();
      return false;
    }

    return true;
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::update_labels(const TagT &tag, const uint32_t *labels, uint32_t label_count) {
    std::shared_lock<std::shared_timed_mutex> recalibration_mutation_guard(hybrid_recalibration_mutation_lock_);
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);

    std::vector<uint32_t> normalized_labels;
    normalized_labels.reserve(label_count);
    for (uint32_t idx = 0; idx < label_count; ++idx) {
      normalized_labels.push_back(labels[idx]);
    }
    std::sort(normalized_labels.begin(), normalized_labels.end());
    normalized_labels.erase(std::unique(normalized_labels.begin(), normalized_labels.end()), normalized_labels.end());

    std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
    auto id_iter = live_ids_by_tag_.find(tag);
    if (id_iter == live_ids_by_tag_.end()) {
      return -1;
    }

    const uint32_t id = id_iter->second;
    auto old_label_iter = live_labels_by_tag_.find(tag);
    if (old_label_iter != live_labels_by_tag_.end()) {
      apply_live_labels_locked(id, old_label_iter->second, false);
      live_labels_by_id_.erase(id);
    }

    if (normalized_labels.empty()) {
      live_labels_by_tag_.erase(tag);
      maybe_mark_hybrid_recalibration_pending();
      return 0;
    }
    live_labels_by_tag_[tag] = normalized_labels;
    live_labels_by_id_[id] = normalized_labels;
    apply_live_labels_locked(id, normalized_labels, true);
    maybe_mark_hybrid_recalibration_pending();
    return 0;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l, HybridFilterKind filter_kind, const void *filter_data,
                                        HybridQueryStats *hybrid_stats, HybridRouteOverride route_override) {
    foreground_counters_.waiting_queries.fetch_add(1);
    struct ForegroundQueryGuard {
      DynamicSSDIndex<T, TagT> *owner = nullptr;
      HybridForegroundCounters *counters = nullptr;

      ForegroundQueryGuard(DynamicSSDIndex<T, TagT> *dynamic_index, HybridForegroundCounters *foreground_counters)
          : owner(dynamic_index), counters(foreground_counters) {
        counters->waiting_queries.fetch_sub(1);
        counters->active_queries.fetch_add(1);
      }

      ~ForegroundQueryGuard() {
        counters->active_queries.fetch_sub(1);
        owner->notify_hybrid_recalibration_worker();
      }
    } foreground_query_guard(this, &foreground_counters_);

    const uint64_t raw_result_limit = std::max<uint64_t>(4096, search_L);
    std::vector<TagT> result_tags(static_cast<size_t>(raw_result_limit), std::numeric_limits<TagT>::max());
    std::vector<float> result_distances(static_cast<size_t>(raw_result_limit), std::numeric_limits<float>::infinity());

    if (stats != nullptr) {
      *stats = {};
    }
    if (hybrid_stats != nullptr) {
      *hybrid_stats = {};
    }

    size_t n = 0;
    const bool use_hybrid_filter = filter_kind != HybridFilterKind::kUnsupported && filter_data != nullptr;

    if (flat_mode_) {
      std::shared_lock<std::shared_timed_mutex> flat_lock(_merge_lock);
      if (flat_mode_) {
        if (use_hybrid_filter) {
          Timer route_timer;
          HybridQueryScratch scratch;
          const std::vector<uint32_t> query_labels = decode_label_filter_data(filter_data);
          uint64_t candidate_count = 0;
          std::vector<uint32_t> candidate_ids;
          {
            std::shared_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
            candidate_count = compute_live_candidate_bitset(filter_kind, query_labels, live_label_bitsets_,
                                                            live_present_bitset_, live_densebit_words_per_label_,
                                                            &scratch);
            materialize_candidate_ids(&scratch, &candidate_ids);
          }
          const uint64_t route_overhead_us = route_timer.elapsed();
          if (hybrid_stats != nullptr) {
            hybrid_stats->candidate_count = candidate_count;
            hybrid_stats->threshold = flat_threshold_;
            hybrid_stats->threshold_version = 0;
            hybrid_stats->route_overhead_us = route_overhead_us;
          }
          if (candidate_count == 0) {
            clear_result_buffers(K, tags, distances);
            if (stats != nullptr) {
              stats->total_us = static_cast<double>(route_overhead_us);
            }
            if (hybrid_stats != nullptr) {
              hybrid_stats->decision = HybridRouteDecision::kPrefilterFastReturn;
            }
            return;
          }
          n = flat_exact_search_locked(query, K, &candidate_ids, tags, distances, stats);
          if (stats != nullptr) {
            stats->total_us += static_cast<double>(route_overhead_us);
          }
          if (hybrid_stats != nullptr) {
            hybrid_stats->decision = HybridRouteDecision::kPrefilter;
            hybrid_stats->route_overhead_us = route_overhead_us;
          }
        } else {
          n = flat_exact_search_locked(query, K, nullptr, tags, distances, stats);
          (void) n;
        }
        return;
      }
    }

    if (use_hybrid_filter) {
      Timer route_timer;
      HybridQueryScratch scratch;
      const std::vector<uint32_t> query_labels = decode_label_filter_data(filter_data);
      uint64_t candidate_count = 0;
      {
        std::shared_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
        candidate_count = compute_live_candidate_bitset(filter_kind, query_labels, live_label_bitsets_,
                                                        live_present_bitset_, live_densebit_words_per_label_, &scratch);
      }

      const HybridMetadata *hybrid_metadata = _disk_index->hybrid_metadata();
      const uint64_t threshold = hybrid_metadata == nullptr ? 0 : hybrid_metadata->header().tau_m;
      const uint64_t threshold_version = hybrid_metadata == nullptr ? 0 : hybrid_metadata->header().threshold_version;
      const bool auto_routing_ready = hybrid_metadata != nullptr
          && (hybrid_metadata->header().flags & kMetadataAllowPrefilterFlag) != 0
          && (hybrid_metadata->header().route_selector_mask & selector_mask_for_kind(filter_kind)) != 0;

      if (hybrid_stats != nullptr) {
        hybrid_stats->candidate_count = candidate_count;
        hybrid_stats->threshold = threshold;
        hybrid_stats->threshold_version = threshold_version;
        hybrid_stats->route_overhead_us = route_timer.elapsed();
      }

      if (candidate_count == 0) {
        clear_result_buffers(K, tags, distances);
        if (stats != nullptr) {
          stats->total_us = static_cast<double>(route_timer.elapsed());
        }
        if (hybrid_stats != nullptr) {
          hybrid_stats->decision = HybridRouteDecision::kPrefilterFastReturn;
          hybrid_stats->route_overhead_us = route_timer.elapsed();
        }
        return;
      }

      const bool choose_prefilter = route_override == HybridRouteOverride::kForcePrefilter
          || (route_override != HybridRouteOverride::kForceGraphOnly
              && auto_routing_ready && candidate_count <= threshold);

      uint64_t search_overhead_us = route_timer.elapsed();
      if (choose_prefilter) {
        std::vector<uint32_t> candidate_ids;
        materialize_candidate_ids(&scratch, &candidate_ids);
        search_overhead_us = route_timer.elapsed();
        n = _disk_index->hybrid_prefilter_search(query, search_L, result_tags.data(), result_distances.data(),
                                                 candidate_ids, stats, live_point_count_.load());
        if (hybrid_stats != nullptr) {
          hybrid_stats->decision = HybridRouteDecision::kPrefilter;
          hybrid_stats->route_overhead_us = search_overhead_us;
        }
      } else {
        LiveLabelSelector selector;
        selector.live_state_lock = &live_state_lock_;
        selector.live_labels_by_id = &live_labels_by_id_;
        selector.filter_kind = filter_kind;
        n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                     beam_width, stats, &selector, filter_data, 0);
        if (hybrid_stats != nullptr) {
          hybrid_stats->decision = auto_routing_ready ? HybridRouteDecision::kGraphOnly
                                                      : HybridRouteDecision::kAutoGraphFallback;
          hybrid_stats->route_overhead_us = search_overhead_us;
        }
      }

      if (stats != nullptr) {
        stats->total_us += static_cast<double>(search_overhead_us);
      }
    } else {
      auto *deletion_set = &deletion_sets[active_delete_set];
      if (search_mode == BEAM_SEARCH) {
        n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                     beam_width, stats, deletion_set, dyn_search_l);
      } else if (search_mode == PAGE_SEARCH) {
        n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                     beam_width, stats);
      } else if (search_mode == PIPE_SEARCH) {
        n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                     beam_width, stats);
      }
    }

    filter_deleted_results(n, result_tags, result_distances, K, tags, distances, delete_lock, deletion_sets);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> recalibration_mutation_guard(hybrid_recalibration_mutation_lock_);
    if (flat_mode_) {
      std::unique_lock<std::shared_timed_mutex> merge_guard(_merge_lock);
      if (flat_mode_) {
        std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
        journal->append(pipeann::TxType::kDelete, tag);
        if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
          deletion_sets[active_delete_set].insert(tag);
          deleted_tags[active_delete_set].push_back(tag);
          std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
          auto id_iter = live_ids_by_tag_.find(tag);
          if (id_iter != live_ids_by_tag_.end()) {
            const uint32_t id = id_iter->second;
            if (id < flat_deleted_.size()) {
              flat_deleted_[id] = 1;
            }
            auto label_iter = live_labels_by_tag_.find(tag);
            if (label_iter != live_labels_by_tag_.end()) {
              apply_live_labels_locked(id, label_iter->second, false);
              live_labels_by_tag_.erase(label_iter);
            }
            live_labels_by_id_.erase(id);
            if (static_cast<size_t>(id >> 6U) < live_present_bitset_.size()) {
              live_present_bitset_[static_cast<size_t>(id >> 6U)] &= ~(1ULL << (id & 63U));
            }
            live_ids_by_tag_.erase(id_iter);
          }
          live_point_count_.store(live_ids_by_tag_.size());
        }
        maybe_mark_hybrid_recalibration_pending();
        return;
      }
    }

    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    journal->append(pipeann::TxType::kDelete, tag);

    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      deletion_sets[active_delete_set].insert(tag);
      deleted_tags[active_delete_set].push_back(tag);
      std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
      auto id_iter = live_ids_by_tag_.find(tag);
      if (id_iter != live_ids_by_tag_.end()) {
        const uint32_t id = id_iter->second;
        if (flat_mode_ && id < flat_deleted_.size()) {
          flat_deleted_[id] = 1;
        }
        auto label_iter = live_labels_by_tag_.find(tag);
        if (label_iter != live_labels_by_tag_.end()) {
          apply_live_labels_locked(id, label_iter->second, false);
          live_labels_by_tag_.erase(label_iter);
        }
        live_labels_by_id_.erase(id);
        if (static_cast<size_t>(id >> 6U) < live_present_bitset_.size()) {
          live_present_bitset_[static_cast<size_t>(id >> 6U)] &= ~(1ULL << (id & 63U));
        }
        live_ids_by_tag_.erase(id_iter);
      }
      live_point_count_.store(live_ids_by_tag_.size());
    }
    maybe_mark_hybrid_recalibration_pending();
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    std::shared_lock<std::shared_timed_mutex> recalibration_mutation_guard(hybrid_recalibration_mutation_lock_);
    foreground_counters_.active_high_priority_tasks.fetch_add(1);
    struct HighPriorityTaskGuard {
      DynamicSSDIndex<T, TagT> *owner = nullptr;
      HybridForegroundCounters *counters = nullptr;

      HighPriorityTaskGuard(DynamicSSDIndex<T, TagT> *dynamic_index, HybridForegroundCounters *foreground_counters)
          : owner(dynamic_index), counters(foreground_counters) {}

      ~HighPriorityTaskGuard() {
        counters->active_high_priority_tasks.fetch_sub(1);
        owner->notify_hybrid_recalibration_worker();
      }
    } high_priority_task_guard(this, &foreground_counters_);

    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    if (flat_mode_) {
      journal->checkpoint();
      return;
    }
    // _disk_index_in -> _disk_index_out
    // Before merge, only the active deletion set contains deletes.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      active_delete_set = !active_delete_set;
    }
    // During merge, both deletion_sets contain deletes.
    pipeann::Timer timer;
    merge(nthreads, n_sampled_nbrs);

    // After merge, clear the inactive deletion set (as they are already merged).
    // Only concurrent search & delete; no concurrent inserts as _merge_lock is held.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      deletion_sets[!active_delete_set].clear();
      deleted_tags[!active_delete_set].clear();
    }

    // TODO(gh): do we really need to reload disk index?
    std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
    // reload the disk index
    _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
    rebuild_live_tag_ids_from_disk(_disk_index_prefix_in);
    if (!persist_live_hybrid_state(_disk_index_prefix_in)) {
      LOG(INFO) << "Skip hybrid sidecar write after merge for " << _disk_index_prefix_in;
    }
    maybe_mark_hybrid_recalibration_pending();
    LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
    MallocExtension::instance()->ReleaseFreeMemory();  // Return free list to OS.
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                               deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::initialize_live_state_from_disk(const std::string &index_prefix) {
    const auto tags_by_id = load_tags_by_id_from_disk<TagT>(index_prefix, _disk_index->meta_.npoints);

    std::vector<std::vector<uint32_t>> labels_by_point(static_cast<size_t>(_disk_index->meta_.npoints));
    uint64_t label_universe = 0;
    const std::string sidecar_path = DenseBitsetIndex::default_sidecar_path(index_prefix);
    if (file_exists(sidecar_path)) {
      auto densebit_index = DenseBitsetIndex::load(sidecar_path, _disk_index->meta_.npoints);
      densebit_index->materialize_labels_by_point(&labels_by_point);
      label_universe = densebit_index->header().nlabels;
    }

    std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
    live_ids_by_tag_.clear();
    live_labels_by_tag_.clear();
    for (uint64_t id = 0; id < _disk_index->meta_.npoints; ++id) {
      const TagT tag = tags_by_id[static_cast<size_t>(id)];
      live_ids_by_tag_[tag] = static_cast<uint32_t>(id);
      if (!labels_by_point[static_cast<size_t>(id)].empty()) {
        live_labels_by_tag_[tag] = std::move(labels_by_point[static_cast<size_t>(id)]);
      }
    }
    rebuild_live_filter_state_locked();
    live_label_universe_ = std::max<uint64_t>(live_label_universe_, label_universe);
    live_point_count_.store(live_ids_by_tag_.size());
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::rebuild_live_tag_ids_from_disk(const std::string &index_prefix) {
    const auto tags_by_id = load_tags_by_id_from_disk<TagT>(index_prefix, _disk_index->meta_.npoints);

    std::unique_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
    std::unordered_map<TagT, uint32_t> next_live_ids_by_tag;
    std::unordered_map<TagT, std::vector<uint32_t>> next_live_labels_by_tag;
    next_live_ids_by_tag.reserve(tags_by_id.size());
    next_live_labels_by_tag.reserve(tags_by_id.size());
    for (uint64_t id = 0; id < tags_by_id.size(); ++id) {
      const TagT tag = tags_by_id[static_cast<size_t>(id)];
      next_live_ids_by_tag[tag] = static_cast<uint32_t>(id);
      auto label_iter = live_labels_by_tag_.find(tag);
      if (label_iter != live_labels_by_tag_.end()) {
        next_live_labels_by_tag.emplace(tag, std::move(label_iter->second));
      }
    }
    live_ids_by_tag_.swap(next_live_ids_by_tag);
    live_labels_by_tag_.swap(next_live_labels_by_tag);
    rebuild_live_filter_state_locked();
    live_point_count_.store(live_ids_by_tag_.size());
  }

  template<typename T, typename TagT>
  bool DynamicSSDIndex<T, TagT>::persist_live_hybrid_state(const std::string &index_prefix) {
    const uint64_t disk_npoints = _disk_index->meta_.npoints;
    std::vector<std::vector<uint32_t>> labels_by_point(static_cast<size_t>(disk_npoints));
    uint64_t live_count = 0;
    uint64_t label_universe = 0;

    {
      std::shared_lock<std::shared_timed_mutex> live_lock(live_state_lock_);
      if (!ids_are_compact(live_ids_by_tag_, disk_npoints)) {
        return false;
      }
      live_count = live_point_count_.load();
      label_universe = live_label_universe_;
      for (const auto &entry : live_ids_by_tag_) {
        const TagT &tag = entry.first;
        const uint32_t id = entry.second;
        auto label_iter = live_labels_by_tag_.find(tag);
        if (label_iter != live_labels_by_tag_.end()) {
          labels_by_point[static_cast<size_t>(id)] = label_iter->second;
          if (!label_iter->second.empty()) {
            label_universe = std::max<uint64_t>(label_universe,
                                                static_cast<uint64_t>(label_iter->second.back()) + 1ULL);
          }
        }
      }
    }

    try {
      std::lock_guard<std::mutex> metadata_lock(hybrid_metadata_io_lock_);
      const std::string sidecar_path = DenseBitsetIndex::default_sidecar_path(index_prefix);
      DenseBitsetIndex::write_atomically(sidecar_path, disk_npoints, label_universe, labels_by_point);

      const std::string meta_path = HybridMetadata::default_metadata_path(index_prefix);
      if (file_exists(meta_path)) {
        auto densebit_index = DenseBitsetIndex::load(sidecar_path, disk_npoints);
        auto hybrid_metadata = HybridMetadata::load(meta_path, false);
        hybrid_metadata->set_densebit_header(densebit_index->header());
        hybrid_metadata->set_n_live_snapshot(live_count);
        hybrid_metadata->write_atomically(meta_path);
      }

      _disk_index->load_hybrid_runtime(index_prefix.c_str());
    } catch (const std::exception &e) {
      LOG(WARNING) << "Failed to refresh live hybrid state after sidecar write: " << e.what();
    }
    return true;
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}  // namespace pipeann
