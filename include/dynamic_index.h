#pragma once

#include "utils/journal.h"
#include "utils/tsl/robin_set.h"
#include "ssd_index.h"
#include "index.h"
#include <atomic>
#include <condition_variable>
#include <limits>
#include <thread>
#include <vector>
#include <cassert>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace pipeann {

  struct HybridForegroundCounters {
    std::atomic<uint32_t> active_queries{0};
    std::atomic<uint32_t> waiting_queries{0};
    std::atomic<uint32_t> active_high_priority_tasks{0};
    std::atomic<bool> background_recalibration_disabled{false};
  };

  struct HybridForegroundBudget {
    uint32_t active_queries_low_watermark = 1;
    uint32_t waiting_queries_low_watermark = 0;
  };

  enum class HybridRecalibrationState : uint8_t {
    kIdle = 0,
    kPending = 1,
    kRunning = 2,
  };

  struct HybridRecalibrationDataset {
    HybridFilterKind filter_kind = HybridFilterKind::kUnsupported;
    std::string query_bin;
    std::string query_label_file;
    uint32_t sample_limit = 0;
  };

  struct HybridRecalibrationConfig {
    uint64_t k = 0;
    uint32_t mem_L = 0;
    uint64_t l_search = 0;
    uint32_t beamwidth = 0;
    HybridForegroundBudget foreground_budget;
    std::vector<HybridRecalibrationDataset> datasets;
  };

  template<typename T, typename TagT = uint32_t>
  class DynamicSSDIndex {
   public:
    /*
    Params:
    - parameters: IndexBuildParameters object with configuration of on-disk index.
    */
    DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_in,
                    const std::string disk_prefix_out, Distance<T> *dist, pipeann::Metric disk_metric,
                    int search_mode = BEAM_SEARCH, bool use_mem_index = false);

    ~DynamicSSDIndex();

    void checkpoint();
    pipeann::Journal<TagT> *journal;

    // in-place
    int insert(const T *point, const TagT &tag);

    int update_labels(const TagT &tag, const uint32_t *labels, uint32_t label_count);

    void configure_hybrid_recalibration(HybridRecalibrationConfig config);

    void set_background_recalibration_disabled(bool disabled);

    void search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                bool dyn_search_l = true, HybridFilterKind filter_kind = HybridFilterKind::kUnsupported,
                const void *filter_data = nullptr, HybridQueryStats *hybrid_stats = nullptr,
                HybridRouteOverride route_override = HybridRouteOverride::kAuto);

    void lazy_delete(const TagT &tag);

    void final_merge(const uint32_t &nthreads = 0,
                     const uint32_t &n_sampled_nbrs = std::numeric_limits<uint32_t>::max());

    uint64_t live_point_count() const {
      return live_point_count_.load();
    }

    HybridRecalibrationState hybrid_recalibration_state() const {
      return hybrid_recalibration_state_.load();
    }

    uint32_t active_query_count() const {
      return foreground_counters_.active_queries.load();
    }

    uint32_t waiting_query_count() const {
      return foreground_counters_.waiting_queries.load();
    }

   private:
    void merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs);
    void initialize_live_state_from_disk(const std::string &index_prefix);
    void rebuild_live_tag_ids_from_disk(const std::string &index_prefix);
    bool persist_live_hybrid_state(const std::string &index_prefix);
    void ensure_live_filter_capacity_locked(uint32_t id);
    void apply_live_labels_locked(uint32_t id, const std::vector<uint32_t> &labels, bool add_labels);
    void rebuild_live_filter_state_locked();
    bool hybrid_recalibration_configured_locked() const;
    void maybe_mark_hybrid_recalibration_pending();
    void update_hybrid_recalibration_metadata(uint64_t live_count, bool pending, bool running);
    void notify_hybrid_recalibration_worker();
    void ensure_hybrid_recalibration_worker_started();
    void stop_hybrid_recalibration_worker();
    void hybrid_recalibration_worker_loop();
    bool can_run_hybrid_recalibration_now() const;
    bool run_hybrid_recalibration_once();

   public:
    size_t _dim;
    uint32_t _num_threads;  // search + insert + delete
    uint64_t _beamwidth;

    std::shared_ptr<AlignedFileReader> reader = nullptr;
    SSDIndex<T, TagT> *_disk_index = nullptr;

    pipeann::Metric _dist_metric;
    Distance<T> *_dist_comp;

    pipeann::IndexBuildParameters _paras_mem;
    pipeann::IndexBuildParameters _paras_disk;

    int active_delete_set = 0;            // reflects active _deletion_set
    std::shared_timed_mutex delete_lock;  // lock to access _deletion_set
    tsl::robin_set<TagT> deletion_sets[2];
    std::vector<TagT> deleted_tags[2];

    std::shared_timed_mutex _merge_lock;

    std::string _disk_index_prefix_in;
    std::string _disk_index_prefix_out;

    bool _use_mem_index = false;
    int search_mode = BEAM_SEARCH;

   private:
    std::atomic<uint64_t> live_point_count_{0};
    mutable std::shared_timed_mutex live_state_lock_;
    std::unordered_map<TagT, uint32_t> live_ids_by_tag_;
    std::unordered_map<TagT, std::vector<uint32_t>> live_labels_by_tag_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> live_labels_by_id_;
    std::unordered_map<uint32_t, std::vector<uint64_t>> live_label_bitsets_;
    std::vector<uint64_t> live_present_bitset_;
    uint64_t live_densebit_words_per_label_ = 0;
    uint64_t live_label_universe_ = 0;
    HybridForegroundCounters foreground_counters_;
    HybridForegroundBudget foreground_budget_;
    std::atomic<HybridRecalibrationState> hybrid_recalibration_state_{HybridRecalibrationState::kIdle};
    mutable std::mutex hybrid_recalibration_config_lock_;
    HybridRecalibrationConfig hybrid_recalibration_config_;
    mutable std::mutex hybrid_metadata_io_lock_;
    mutable std::mutex hybrid_recalibration_worker_lock_;
    std::condition_variable hybrid_recalibration_cv_;
    std::thread hybrid_recalibration_worker_;
    bool hybrid_recalibration_worker_started_ = false;
    bool hybrid_recalibration_worker_stop_ = false;
    uint64_t hybrid_recalibration_signal_count_ = 0;
  };
}  // namespace pipeann
