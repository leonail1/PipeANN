#include "filter/densebit_index.h"
#include "filter/hybrid_metadata.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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
    char dir_template[] = "/tmp/pipeann_hybrid_metadata_testXXXXXX";
    char *created = ::mkdtemp(dir_template);
    if (created == nullptr) {
      fail("failed to create temporary directory");
    }
    return TempDir{std::filesystem::path(created)};
  }

  void expect_throws(const std::function<void()> &fn, const std::string &needle, const std::string &context) {
    try {
      fn();
    } catch (const std::exception &e) {
      if (!needle.empty() && std::string(e.what()).find(needle) == std::string::npos) {
        fail(context + ": unexpected error: " + e.what());
      }
      return;
    }
    fail(context + ": expected exception was not thrown");
  }

  void overwrite_header(const std::filesystem::path &path, const pipeann::HybridMetadataHeaderV1 &header) {
    std::fstream writer(path, std::ios::binary | std::ios::in | std::ios::out);
    require(writer.is_open(), "failed to reopen metadata file for overwrite");
    writer.write(reinterpret_cast<const char *>(&header), sizeof(header));
    require(writer.good(), "failed to overwrite metadata header");
  }
}  // namespace

int main() {
  try {
    const TempDir temp_dir = make_temp_dir();
    const auto meta_path = temp_dir.path / "sample_hybrid.meta";

    pipeann::HybridMetadataHeaderV1 header;
    header.flags = kMetadataValidFlag | kCalibrationValidFlag | kAllowPrefilterFlag;
    header.route_selector_mask = 0x3ULL;
    header.tau_m = 64;
    header.n_calib = 1024;
    header.n_live_snapshot = 1024;
    header.threshold_version = 7;
    header.calib_epoch_sec = 1713916800;
    header.calib_query_count = 32;
    header.calib_bucket_count = 2;
    header.calib_k = 10;
    header.calib_mem_L = 0;
    header.calib_beamwidth = 4;
    header.calib_l_search = 64;
    header.densebit_npoints = 1024;
    header.densebit_nlabels = 17;
    header.densebit_words_per_label = 16;
    header.densebit_nnz = 300;

    std::vector<pipeann::HybridCalibrationBucketV1> buckets{
        {4, 10, 12, 18, 0},
        {16, 22, 24, 40, 0},
    };

    auto metadata = pipeann::HybridMetadata::create(header, buckets);
    metadata->write_atomically(meta_path.string());

    require(std::filesystem::exists(meta_path), "metadata file was not written");
    require(!std::filesystem::exists(meta_path.string() + ".tmp"), "temporary metadata file was not cleaned up");

    auto loaded = pipeann::HybridMetadata::load(meta_path.string());
    require(std::memcmp(&loaded->header(), &header, sizeof(header)) == 0, "metadata header changed after round-trip");
    require(loaded->buckets().size() == buckets.size(), "bucket count changed after round-trip");
    for (size_t bucket_idx = 0; bucket_idx < buckets.size(); ++bucket_idx) {
      require(std::memcmp(&loaded->buckets()[bucket_idx], &buckets[bucket_idx], sizeof(buckets[bucket_idx])) == 0,
              "bucket payload changed after round-trip");
    }

    pipeann::DenseBitsetFileHeaderV1 matching_densebit{};
    matching_densebit.npoints = header.densebit_npoints;
    matching_densebit.nlabels = header.densebit_nlabels;
    matching_densebit.words_per_label = header.densebit_words_per_label;
    matching_densebit.nnz = header.densebit_nnz;
    loaded->validate_against_densebit(matching_densebit);
    loaded->validate_against_npoints(header.n_calib);

    pipeann::DenseBitsetFileHeaderV1 mismatched_densebit = matching_densebit;
    mismatched_densebit.nlabels += 1;
    expect_throws([&loaded, &mismatched_densebit]() { loaded->validate_against_densebit(mismatched_densebit); },
                  "does not match densebit sidecar header", "densebit mismatch validation");
    expect_throws([&loaded, &header]() { loaded->validate_against_npoints(header.n_calib + 1); },
                  "calibration point count does not match current index", "npoints mismatch validation");

    pipeann::HybridMetadataHeaderV1 corrupted_header = header;
    corrupted_header.route_selector_mask = 0;
    overwrite_header(meta_path, corrupted_header);
    expect_throws([&meta_path]() { pipeann::HybridMetadata::load(meta_path.string()); },
                  "selector mask is invalid", "corrupted metadata load");

    std::cout << "hybrid_metadata_roundtrip: ok" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "hybrid_metadata_roundtrip: " << e.what() << std::endl;
    return 1;
  }
}