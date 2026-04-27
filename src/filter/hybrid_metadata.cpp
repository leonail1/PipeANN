#include "filter/hybrid_metadata.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace pipeann {
  namespace {
    constexpr uint64_t kMetadataValidFlag = 1ULL << 0;
    constexpr uint64_t kCalibrationValidFlag = 1ULL << 1;
    constexpr uint64_t kPendingRecalibrationFlag = 1ULL << 3;
    constexpr uint64_t kRunningRecalibrationFlag = 1ULL << 4;
    constexpr uint64_t kAllowedSelectorMask = 0x7ULL;

    static_assert(sizeof(HybridMetadataHeaderV1) == 256, "HybridMetadataHeaderV1 size must be 256 bytes");
    static_assert(sizeof(HybridCalibrationBucketV1) == 40, "HybridCalibrationBucketV1 size must be 40 bytes");

    void write_all_or_throw(int fd, const void *buffer, size_t bytes, const std::string &path) {
      const char *cursor = static_cast<const char *>(buffer);
      size_t remaining = bytes;
      while (remaining > 0) {
        const ssize_t written = ::write(fd, cursor, remaining);
        if (written < 0) {
          if (errno == EINTR) {
            continue;
          }
          throw std::runtime_error("failed to write hybrid metadata " + path + ": " + std::strerror(errno));
        }
        cursor += written;
        remaining -= static_cast<size_t>(written);
      }
    }
  }  // namespace

  std::string HybridMetadata::default_metadata_path(const std::string &index_prefix) {
    return index_prefix + "_hybrid.meta";
  }

  std::unique_ptr<HybridMetadata> HybridMetadata::create(HybridMetadataHeaderV1 header,
                                                         std::vector<HybridCalibrationBucketV1> buckets) {
    return std::unique_ptr<HybridMetadata>(new HybridMetadata(header, std::move(buckets)));
  }

  std::unique_ptr<HybridMetadata> HybridMetadata::load(const std::string &meta_path, bool require_routing_ready) {
    std::ifstream reader(meta_path, std::ios::binary);
    if (!reader.is_open()) {
      throw std::runtime_error("failed to open hybrid metadata: " + meta_path);
    }

    reader.seekg(0, std::ios::end);
    const std::streamoff file_bytes = reader.tellg();
    reader.seekg(0, std::ios::beg);
    if (file_bytes < static_cast<std::streamoff>(sizeof(HybridMetadataHeaderV1))) {
      throw std::runtime_error("hybrid metadata too small: " + meta_path);
    }

    HybridMetadataHeaderV1 header;
    reader.read(reinterpret_cast<char *>(&header), sizeof(HybridMetadataHeaderV1));
    if (!reader.good()) {
      throw std::runtime_error("failed to read hybrid metadata header: " + meta_path);
    }
    if (header.magic != kHybridMetaMagic || header.version != kHybridMetaVersion || header.header_bytes != 256) {
      throw std::runtime_error("hybrid metadata header mismatch: " + meta_path);
    }
    if ((header.route_selector_mask & ~kAllowedSelectorMask) != 0) {
      throw std::runtime_error("hybrid metadata selector mask is invalid: " + meta_path);
    }
    if (require_routing_ready) {
      if ((header.flags & (kMetadataValidFlag | kCalibrationValidFlag))
          != (kMetadataValidFlag | kCalibrationValidFlag)) {
        throw std::runtime_error("hybrid metadata flags are not valid for routing: " + meta_path);
      }
      if (header.route_selector_mask == 0) {
        throw std::runtime_error("hybrid metadata selector mask is invalid: " + meta_path);
      }
    }

    const uint64_t expected_bytes = sizeof(HybridMetadataHeaderV1)
        + header.calib_bucket_count * sizeof(HybridCalibrationBucketV1);
    if (static_cast<uint64_t>(file_bytes) != expected_bytes) {
      throw std::runtime_error("hybrid metadata size mismatch: " + meta_path);
    }

    std::vector<HybridCalibrationBucketV1> buckets(static_cast<size_t>(header.calib_bucket_count));
    if (!buckets.empty()) {
      reader.read(reinterpret_cast<char *>(buckets.data()),
                  static_cast<std::streamsize>(buckets.size() * sizeof(HybridCalibrationBucketV1)));
      if (!reader.good()) {
        throw std::runtime_error("failed to read hybrid metadata buckets: " + meta_path);
      }
      if (!std::is_sorted(buckets.begin(), buckets.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.candidate_upper_bound < rhs.candidate_upper_bound;
          })) {
        throw std::runtime_error("hybrid metadata buckets are not sorted: " + meta_path);
      }
    }

    return std::unique_ptr<HybridMetadata>(new HybridMetadata(header, std::move(buckets)));
  }

  HybridMetadata::HybridMetadata(HybridMetadataHeaderV1 header, std::vector<HybridCalibrationBucketV1> buckets)
      : header_(header), buckets_(std::move(buckets)) {
  }

  const HybridMetadataHeaderV1 &HybridMetadata::header() const {
    return header_;
  }

  const std::vector<HybridCalibrationBucketV1> &HybridMetadata::buckets() const {
    return buckets_;
  }

  void HybridMetadata::validate_against_densebit(const DenseBitsetFileHeaderV1 &densebit_header) const {
    if (header_.densebit_npoints != densebit_header.npoints || header_.densebit_nlabels != densebit_header.nlabels
        || header_.densebit_words_per_label != densebit_header.words_per_label
        || header_.densebit_nnz != densebit_header.nnz) {
      throw std::runtime_error("hybrid metadata does not match densebit sidecar header");
    }
  }

  void HybridMetadata::validate_against_npoints(uint64_t npoints) const {
    if (header_.n_calib != npoints) {
      throw std::runtime_error("hybrid metadata calibration point count does not match current index");
    }
  }

  void HybridMetadata::write_atomically(const std::string &meta_path) const {
    const std::string tmp_path = meta_path + ".tmp";
    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      throw std::runtime_error("failed to open hybrid metadata temp file: " + tmp_path);
    }

    try {
      write_all_or_throw(fd, &header_, sizeof(header_), tmp_path);
      if (!buckets_.empty()) {
        write_all_or_throw(fd, buckets_.data(), buckets_.size() * sizeof(HybridCalibrationBucketV1), tmp_path);
      }
      if (::fsync(fd) != 0) {
        throw std::runtime_error("failed to fsync hybrid metadata temp file: " + tmp_path);
      }
      if (::close(fd) != 0) {
        throw std::runtime_error("failed to close hybrid metadata temp file: " + tmp_path);
      }
      if (::rename(tmp_path.c_str(), meta_path.c_str()) != 0) {
        throw std::runtime_error("failed to publish hybrid metadata file: " + meta_path);
      }
    } catch (...) {
      ::close(fd);
      ::unlink(tmp_path.c_str());
      throw;
    }
  }

  void HybridMetadata::set_n_live_snapshot(uint64_t n_live_snapshot) {
    header_.n_live_snapshot = n_live_snapshot;
  }

  void HybridMetadata::set_densebit_header(const DenseBitsetFileHeaderV1 &densebit_header) {
    header_.densebit_npoints = densebit_header.npoints;
    header_.densebit_nlabels = densebit_header.nlabels;
    header_.densebit_words_per_label = densebit_header.words_per_label;
    header_.densebit_nnz = densebit_header.nnz;
  }

  void HybridMetadata::set_recalibration_flags(bool pending, bool running) {
    header_.flags &= ~(kPendingRecalibrationFlag | kRunningRecalibrationFlag);
    if (pending) {
      header_.flags |= kPendingRecalibrationFlag;
    }
    if (running) {
      header_.flags |= kRunningRecalibrationFlag;
    }
  }

  void HybridMetadata::disable_routing() {
    header_.flags &= ~(kCalibrationValidFlag | (1ULL << 2));
    header_.route_selector_mask = 0;
    header_.tau_m = 0;
  }
}  // namespace pipeann
