#include "filter/densebit_index.h"
#include "filter/hybrid_metadata.h"
#include "filter/selector.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdlib.h>

namespace {
  constexpr uint64_t kMetadataValidFlag = 1ULL << 0;
  constexpr uint64_t kCalibrationValidFlag = 1ULL << 1;
  constexpr uint64_t kAllowPrefilterFlag = 1ULL << 2;

  [[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
  }

  void require(bool condition, const std::string &message) {
    if (!condition) {
      fail(message);
    }
  }

  struct TempDir {
    std::filesystem::path path;

    ~TempDir() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  };

  TempDir make_temp_dir() {
    char dir_template[] = "/tmp/pipeann_densebit_range_testXXXXXX";
    char *created = ::mkdtemp(dir_template);
    if (created == nullptr) {
      fail("failed to create temporary directory");
    }
    return TempDir{std::filesystem::path(created)};
  }

  void expect_throws(const std::function<void()> &fn, const std::string &context) {
    try {
      fn();
    } catch (const std::exception &) {
      return;
    }
    fail(context + ": expected exception was not thrown");
  }

  std::vector<char> label_buffer(std::initializer_list<uint32_t> labels) {
    std::vector<char> buffer(sizeof(uint32_t) * (labels.size() + 1), 0);
    const uint32_t count = static_cast<uint32_t>(labels.size());
    std::memcpy(buffer.data(), &count, sizeof(uint32_t));
    if (count > 0) {
      std::memcpy(buffer.data() + sizeof(uint32_t), labels.begin(), sizeof(uint32_t) * labels.size());
    }
    return buffer;
  }
}  // namespace

int main() {
  try {
    const TempDir temp_dir = make_temp_dir();
    const std::filesystem::path sidecar_path = temp_dir.path / "sample_labels.densebit";

    std::vector<std::vector<uint32_t>> labels_by_point{
        {0}, {1}, {2}, {3}, {0}, {1}, {2}, {3},
    };
    pipeann::DenseBitsetIndex::write_atomically(sidecar_path.string(), labels_by_point.size(), 4, labels_by_point);
    auto densebit = pipeann::DenseBitsetIndex::load(sidecar_path.string(), labels_by_point.size());

    pipeann::HybridQueryScratch scratch;
    require(densebit->count_candidates(pipeann::HybridFilterKind::kRange, {1, 2}, &scratch) == 4,
            "range [1,2] candidate count mismatch");

    std::vector<uint32_t> candidate_ids;
    densebit->materialize_candidates(pipeann::HybridFilterKind::kRange, {1, 2}, &scratch, &candidate_ids);
    std::vector<uint32_t> expected_ids{1, 2, 5, 6};
    require(candidate_ids == expected_ids, "range [1,2] materialized ids mismatch");

    require(densebit->count_candidates(pipeann::HybridFilterKind::kRange, {3}, &scratch) == 2,
            "range equality candidate count mismatch");
    require(densebit->count_candidates(pipeann::HybridFilterKind::kRange, {7, 9}, &scratch) == 0,
            "out-of-universe range should be empty");

    pipeann::RangeSelector selector;
    const auto query_range = label_buffer({1, 2});
    const auto query_equal = label_buffer({3});
    const auto target_two = label_buffer({2});
    const auto target_zero = label_buffer({0});
    const auto target_multi = label_buffer({0, 2});
    require(selector.is_member(0, query_range.data(), target_two.data()), "range selector should match label 2");
    require(selector.is_member(0, query_range.data(), target_multi.data()), "range selector should match one label in a multi-label target");
    require(!selector.is_member(0, query_equal.data(), target_two.data()), "equality selector should reject label 2");
    require(!selector.is_member(0, query_range.data(), target_zero.data()), "range selector should reject label 0");

    const std::filesystem::path meta_path = temp_dir.path / "sample_hybrid.meta";
    pipeann::HybridMetadataHeaderV1 header;
    header.flags = kMetadataValidFlag | kCalibrationValidFlag | kAllowPrefilterFlag;
    header.route_selector_mask = 4ULL;
    header.tau_m = 8;
    header.n_calib = labels_by_point.size();
    header.n_live_snapshot = labels_by_point.size();
    header.threshold_version = 1;
    header.calib_bucket_count = 0;
    header.densebit_npoints = densebit->header().npoints;
    header.densebit_nlabels = densebit->header().nlabels;
    header.densebit_words_per_label = densebit->header().words_per_label;
    header.densebit_nnz = densebit->header().nnz;

    auto metadata = pipeann::HybridMetadata::create(header, {});
    metadata->write_atomically(meta_path.string());
    auto loaded_metadata = pipeann::HybridMetadata::load(meta_path.string());
    require(loaded_metadata->header().route_selector_mask == 4ULL, "range selector mask did not round-trip");

    header.route_selector_mask = 8ULL;
    pipeann::HybridMetadata::create(header, {})->write_atomically(meta_path.string());
    expect_throws([&]() { pipeann::HybridMetadata::load(meta_path.string()); }, "invalid selector mask");

    std::cout << "densebit_range: ok" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "densebit_range: " << e.what() << std::endl;
    return 1;
  }
}
