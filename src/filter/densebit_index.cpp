#include "filter/densebit_index.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "utils.h"

namespace pipeann {
  namespace {
    constexpr uint64_t kDenseBitsetMagic = 0x54494245534E4544ULL;
    constexpr uint64_t kDenseBitsetVersion = 1ULL;

    uint64_t dense_tail_mask(uint64_t npoints) {
      const uint64_t rem = npoints % 64ULL;
      return rem == 0 ? std::numeric_limits<uint64_t>::max() : ((1ULL << rem) - 1ULL);
    }

    uint64_t popcount_words(const std::vector<uint64_t> &words) {
      uint64_t total = 0;
      for (uint64_t word : words) {
        total += static_cast<uint64_t>(__builtin_popcountll(word));
      }
      return total;
    }

    void write_all_or_throw(int fd, const void *buffer, size_t bytes, const std::string &path) {
      const char *cursor = static_cast<const char *>(buffer);
      size_t remaining = bytes;
      while (remaining > 0) {
        const ssize_t written = ::write(fd, cursor, remaining);
        if (written < 0) {
          if (errno == EINTR) {
            continue;
          }
          throw std::runtime_error("failed to write densebit sidecar " + path + ": " + std::strerror(errno));
        }
        cursor += written;
        remaining -= static_cast<size_t>(written);
      }
    }
  }  // namespace

  std::string DenseBitsetIndex::default_sidecar_path(const std::string &index_prefix) {
    return index_prefix + "_labels.densebit";
  }

  std::unique_ptr<DenseBitsetIndex> DenseBitsetIndex::load(const std::string &sidecar_path, uint64_t expected_npoints) {
    const int fd = ::open(sidecar_path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("failed to open densebit sidecar: " + sidecar_path);
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("failed to stat densebit sidecar: " + sidecar_path);
    }
    if (static_cast<size_t>(st.st_size) < sizeof(DenseBitsetFileHeaderV1)) {
      ::close(fd);
      throw std::runtime_error("densebit sidecar too small: " + sidecar_path);
    }

    void *mmap_addr = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_addr == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("failed to mmap densebit sidecar: " + sidecar_path);
    }

    const auto *header = reinterpret_cast<const DenseBitsetFileHeaderV1 *>(mmap_addr);
    const uint64_t expected_bytes = sizeof(DenseBitsetFileHeaderV1)
        + header->nlabels * header->words_per_label * sizeof(uint64_t);
    if (header->magic != kDenseBitsetMagic || header->version != kDenseBitsetVersion) {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar magic/version mismatch: " + sidecar_path);
    }
    if (expected_npoints != 0 && header->npoints != expected_npoints) {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar point count mismatch: " + sidecar_path);
    }
    if (static_cast<uint64_t>(st.st_size) != expected_bytes) {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar size mismatch: " + sidecar_path);
    }

    const uint64_t *base_words = reinterpret_cast<const uint64_t *>(static_cast<const char *>(mmap_addr)
                                                                    + sizeof(DenseBitsetFileHeaderV1));
    return std::unique_ptr<DenseBitsetIndex>(new DenseBitsetIndex(sidecar_path, fd, mmap_addr,
                                                                  static_cast<size_t>(st.st_size), *header,
                                                                  base_words));
  }

  void DenseBitsetIndex::write_atomically(const std::string &sidecar_path, uint64_t npoints, uint64_t nlabels,
                                          const std::vector<std::vector<uint32_t>> &labels_by_point) {
    if (labels_by_point.size() != static_cast<size_t>(npoints)) {
      throw std::invalid_argument("labels_by_point size does not match densebit npoints");
    }

    const uint64_t words_per_label = (npoints + 63ULL) / 64ULL;
    if (nlabels > 0 && words_per_label > std::numeric_limits<size_t>::max() / nlabels) {
      throw std::runtime_error("densebit payload would overflow addressable memory");
    }

    std::vector<uint64_t> payload(static_cast<size_t>(nlabels * words_per_label), 0ULL);
    uint64_t nnz = 0;
    for (uint64_t point_id = 0; point_id < npoints; ++point_id) {
      const uint64_t word_idx = point_id >> 6;
      const uint64_t bit_mask = 1ULL << (point_id & 63ULL);
      for (uint32_t label : labels_by_point[static_cast<size_t>(point_id)]) {
        if (label >= nlabels) {
          throw std::runtime_error("label id exceeds densebit label universe");
        }
        payload[static_cast<size_t>(label) * static_cast<size_t>(words_per_label) + static_cast<size_t>(word_idx)] |= bit_mask;
        ++nnz;
      }
    }

    DenseBitsetFileHeaderV1 header;
    header.magic = kDenseBitsetMagic;
    header.version = kDenseBitsetVersion;
    header.npoints = npoints;
    header.nlabels = nlabels;
    header.words_per_label = words_per_label;
    header.nnz = nnz;

    const std::string tmp_path = sidecar_path + ".tmp";
    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      throw std::runtime_error("failed to open densebit temp file: " + tmp_path);
    }

    try {
      write_all_or_throw(fd, &header, sizeof(header), tmp_path);
      if (!payload.empty()) {
        write_all_or_throw(fd, payload.data(), payload.size() * sizeof(uint64_t), tmp_path);
      }
      if (::fsync(fd) != 0) {
        throw std::runtime_error("failed to fsync densebit temp file: " + tmp_path);
      }
      if (::close(fd) != 0) {
        throw std::runtime_error("failed to close densebit temp file: " + tmp_path);
      }
      if (::rename(tmp_path.c_str(), sidecar_path.c_str()) != 0) {
        throw std::runtime_error("failed to publish densebit sidecar: " + sidecar_path);
      }
    } catch (...) {
      ::close(fd);
      ::unlink(tmp_path.c_str());
      throw;
    }
  }

  DenseBitsetIndex::DenseBitsetIndex(std::string sidecar_path, int fd, void *mmap_addr, size_t mmap_len,
                                     DenseBitsetFileHeaderV1 header, const uint64_t *base_words)
      : sidecar_path_(std::move(sidecar_path)), fd_(fd), mmap_addr_(mmap_addr), mmap_len_(mmap_len), header_(header),
        base_words_(base_words) {
  }

  DenseBitsetIndex::~DenseBitsetIndex() {
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
      ::munmap(mmap_addr_, mmap_len_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  const DenseBitsetFileHeaderV1 &DenseBitsetIndex::header() const {
    return header_;
  }

  void DenseBitsetIndex::materialize_labels_by_point(std::vector<std::vector<uint32_t>> *output) const {
    if (output == nullptr) {
      throw std::invalid_argument("materialize_labels_by_point requires output");
    }

    output->assign(static_cast<size_t>(header_.npoints), {});
    for (uint64_t label = 0; label < header_.nlabels; ++label) {
      const uint64_t *label_words = base_words_ + static_cast<size_t>(label) * static_cast<size_t>(header_.words_per_label);
      for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
        uint64_t word = label_words[static_cast<size_t>(word_idx)];
        if (word_idx + 1 == header_.words_per_label) {
          word &= dense_tail_mask(header_.npoints);
        }
        while (word != 0) {
          const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
          const uint64_t point_id = word_idx * 64ULL + bit;
          if (point_id < header_.npoints) {
            (*output)[static_cast<size_t>(point_id)].push_back(static_cast<uint32_t>(label));
          }
          word &= (word - 1);
        }
      }
    }
  }

  uint64_t DenseBitsetIndex::count_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                              HybridQueryScratch *scratch) const {
    return compute_bitset(kind, labels, scratch);
  }

  void DenseBitsetIndex::materialize_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                                HybridQueryScratch *scratch, std::vector<uint32_t> *output) const {
    compute_bitset(kind, labels, scratch);
    materialize_from_scratch(scratch, output);
  }

  uint64_t DenseBitsetIndex::compute_bitset(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                            HybridQueryScratch *scratch) const {
    if (scratch == nullptr) {
      throw std::invalid_argument("HybridQueryScratch must not be null");
    }

    normalize_labels(labels, scratch);
    scratch->bitset_words.assign(static_cast<size_t>(header_.words_per_label), 0ULL);

    if (kind == HybridFilterKind::kUnsupported) {
      return 0;
    }

    if (kind == HybridFilterKind::kIntersect) {
      if (scratch->normalized_labels.empty()) {
        return 0;
      }
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          continue;
        }
        const uint64_t *label_words = base_words_ + static_cast<size_t>(label) * static_cast<size_t>(header_.words_per_label);
        for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
          scratch->bitset_words[static_cast<size_t>(word_idx)] |= label_words[static_cast<size_t>(word_idx)];
        }
      }
    } else if (kind == HybridFilterKind::kSubset) {
      if (scratch->normalized_labels.empty()) {
        scratch->bitset_words.assign(static_cast<size_t>(header_.words_per_label), std::numeric_limits<uint64_t>::max());
      } else {
        bool first_label = true;
        for (uint32_t label : scratch->normalized_labels) {
          if (label >= header_.nlabels) {
            std::fill(scratch->bitset_words.begin(), scratch->bitset_words.end(), 0ULL);
            return 0;
          }
          const uint64_t *label_words = base_words_ + static_cast<size_t>(label) * static_cast<size_t>(header_.words_per_label);
          if (first_label) {
            for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
              scratch->bitset_words[static_cast<size_t>(word_idx)] = label_words[static_cast<size_t>(word_idx)];
            }
            first_label = false;
          } else {
            for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
              scratch->bitset_words[static_cast<size_t>(word_idx)] &= label_words[static_cast<size_t>(word_idx)];
            }
          }
        }
      }
    }

    if (!scratch->bitset_words.empty()) {
      scratch->bitset_words.back() &= dense_tail_mask(header_.npoints);
    }
    return popcount_words(scratch->bitset_words);
  }

  void DenseBitsetIndex::normalize_labels(const std::vector<uint32_t> &labels, HybridQueryScratch *scratch) const {
    scratch->normalized_labels = labels;
    std::sort(scratch->normalized_labels.begin(), scratch->normalized_labels.end());
    scratch->normalized_labels.erase(
        std::unique(scratch->normalized_labels.begin(), scratch->normalized_labels.end()),
        scratch->normalized_labels.end());
  }

  void DenseBitsetIndex::materialize_from_scratch(HybridQueryScratch *scratch, std::vector<uint32_t> *output) const {
    if (scratch == nullptr || output == nullptr) {
      throw std::invalid_argument("materialize_candidates requires scratch and output");
    }

    scratch->candidate_ids.clear();
    scratch->candidate_ids.reserve(static_cast<size_t>(popcount_words(scratch->bitset_words)));
    for (uint64_t word_idx = 0; word_idx < scratch->bitset_words.size(); ++word_idx) {
      uint64_t word = scratch->bitset_words[static_cast<size_t>(word_idx)];
      while (word != 0) {
        const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
        scratch->candidate_ids.push_back(static_cast<uint32_t>(word_idx * 64ULL + bit));
        word &= (word - 1);
      }
    }
    *output = scratch->candidate_ids;
  }
}  // namespace pipeann