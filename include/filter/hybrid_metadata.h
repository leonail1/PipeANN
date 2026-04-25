#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "filter/densebit_index.h"

namespace pipeann {
  constexpr uint64_t kHybridMetaMagic = 0x4859425249444d54ULL;
  constexpr uint32_t kHybridMetaVersion = 1U;

  struct HybridMetadataHeaderV1 {
    uint64_t magic = kHybridMetaMagic;
    uint32_t version = kHybridMetaVersion;
    uint32_t header_bytes = 256;
    uint64_t flags = 0;
    uint64_t route_selector_mask = 0;
    uint64_t tau_m = 0;
    uint64_t n_calib = 0;
    uint64_t n_live_snapshot = 0;
    uint64_t threshold_version = 0;
    uint64_t calib_epoch_sec = 0;
    uint64_t calib_query_count = 0;
    uint64_t calib_bucket_count = 0;
    uint64_t calib_k = 0;
    uint64_t calib_mem_L = 0;
    uint64_t calib_beamwidth = 0;
    uint64_t calib_l_search = 0;
    uint64_t densebit_npoints = 0;
    uint64_t densebit_nlabels = 0;
    uint64_t densebit_words_per_label = 0;
    uint64_t densebit_nnz = 0;
    uint64_t reserved[13]{};
  };

  struct HybridCalibrationBucketV1 {
    uint64_t candidate_upper_bound = 0;
    uint64_t query_count = 0;
    uint64_t prefilter_p50_us = 0;
    uint64_t graph_p50_us = 0;
    uint64_t reserved = 0;
  };

  class HybridMetadata {
   public:
    static std::string default_metadata_path(const std::string &index_prefix);
    static std::unique_ptr<HybridMetadata> create(HybridMetadataHeaderV1 header,
                            std::vector<HybridCalibrationBucketV1> buckets);
    static std::unique_ptr<HybridMetadata> load(const std::string &meta_path, bool require_routing_ready = true);

    const HybridMetadataHeaderV1 &header() const;
    const std::vector<HybridCalibrationBucketV1> &buckets() const;

    void validate_against_densebit(const DenseBitsetFileHeaderV1 &densebit_header) const;
    void validate_against_npoints(uint64_t npoints) const;
    void write_atomically(const std::string &meta_path) const;
    void set_n_live_snapshot(uint64_t n_live_snapshot);
    void set_densebit_header(const DenseBitsetFileHeaderV1 &densebit_header);
    void set_recalibration_flags(bool pending, bool running);
    void disable_routing();

   private:
    HybridMetadata(HybridMetadataHeaderV1 header, std::vector<HybridCalibrationBucketV1> buckets);

    HybridMetadataHeaderV1 header_{};
    std::vector<HybridCalibrationBucketV1> buckets_;
  };
}  // namespace pipeann