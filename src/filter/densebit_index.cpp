#include "filter/densebit_index.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace pipeann {
  namespace {
    constexpr uint64_t kDenseBitsetMagic = 0x54494245534E4544ULL;
    constexpr uint64_t kDenseBitsetVersionV1 = 1ULL;
    constexpr uint64_t kDenseBitsetVersionV2 = 2ULL;

    enum class DenseBitsetEncoding : uint16_t {
      kEmpty = 0,
      kBitmap = 1,
      kPosting = 2,
    };

    struct DenseBitsetFileHeaderRawV1 {
      uint64_t magic = 0;
      uint64_t version = 0;
      uint64_t npoints = 0;
      uint64_t nlabels = 0;
      uint64_t words_per_label = 0;
      uint64_t nnz = 0;
    };

    struct DenseBitsetFileHeaderRawV2 {
      uint64_t magic = 0;
      uint64_t version = 0;
      uint64_t npoints = 0;
      uint64_t nlabels = 0;
      uint64_t words_per_label = 0;
      uint64_t nnz = 0;
      uint64_t label_entry_offset = 0;
      uint64_t payload_offset = 0;
      uint64_t posting_threshold = 0;
      uint64_t dense_label_count = 0;
      uint64_t sparse_label_count = 0;
      uint64_t reserved0 = 0;
    };

    struct DenseBitsetLabelEntryV2 {
      uint64_t payload_offset = 0;
      uint64_t payload_size = 0;
      uint64_t candidate_count = 0;
      uint16_t encoding = static_cast<uint16_t>(DenseBitsetEncoding::kEmpty);
      uint16_t reserved16 = 0;
      uint32_t reserved32 = 0;
    };

    uint64_t dense_tail_mask(uint64_t npoints) {
      const uint64_t rem = npoints % 64ULL;
      return rem == 0 ? std::numeric_limits<uint64_t>::max() : ((1ULL << rem) - 1ULL);
    }

    void apply_tail_mask(std::vector<uint64_t> *words, uint64_t npoints) {
      if (words == nullptr || words->empty()) {
        return;
      }
      words->back() &= dense_tail_mask(npoints);
    }

    uint64_t align_up(uint64_t value, uint64_t alignment) {
      if (alignment <= 1) {
        return value;
      }
      const uint64_t rem = value % alignment;
      return rem == 0 ? value : value + (alignment - rem);
    }

    uint64_t posting_threshold_for_words(uint64_t words_per_label) {
      return words_per_label > (std::numeric_limits<uint64_t>::max() / 2ULL) ? std::numeric_limits<uint64_t>::max()
                                                                             : 2ULL * words_per_label;
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

    void write_padding_or_throw(int fd, uint64_t bytes, const std::string &path) {
      static constexpr uint64_t kZero = 0ULL;
      uint64_t remaining = bytes;
      while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, sizeof(kZero)));
        write_all_or_throw(fd, &kZero, chunk, path);
        remaining -= static_cast<uint64_t>(chunk);
      }
    }

    const DenseBitsetLabelEntryV2 &entry_at(const uint8_t *label_directory, uint32_t label) {
      return reinterpret_cast<const DenseBitsetLabelEntryV2 *>(label_directory)[static_cast<size_t>(label)];
    }

    DenseBitsetEncoding label_encoding(uint64_t file_version, const uint8_t *label_directory, uint32_t label) {
      if (file_version == kDenseBitsetVersionV1) {
        return DenseBitsetEncoding::kBitmap;
      }
      return static_cast<DenseBitsetEncoding>(entry_at(label_directory, label).encoding);
    }

    const uint64_t *bitmap_words(uint64_t file_version, const uint64_t *base_words, const uint8_t *label_directory,
                                 const uint8_t *payload_base, uint64_t words_per_label, uint32_t label) {
      if (file_version == kDenseBitsetVersionV1) {
        return base_words + static_cast<size_t>(label) * static_cast<size_t>(words_per_label);
      }
      const auto &entry = entry_at(label_directory, label);
      return reinterpret_cast<const uint64_t *>(payload_base + entry.payload_offset);
    }

    const uint32_t *posting_ids(const uint8_t *label_directory, const uint8_t *payload_base, uint32_t label) {
      const auto &entry = entry_at(label_directory, label);
      return reinterpret_cast<const uint32_t *>(payload_base + entry.payload_offset);
    }

    uint64_t posting_count(const uint8_t *label_directory, uint32_t label) {
      return entry_at(label_directory, label).candidate_count;
    }

    void or_posting_into_bitset(const uint32_t *ids, uint64_t count, std::vector<uint64_t> *bitset_words) {
      for (uint64_t idx = 0; idx < count; ++idx) {
        const uint32_t point_id = ids[idx];
        const uint64_t word_idx = static_cast<uint64_t>(point_id) >> 6U;
        if (word_idx >= bitset_words->size()) {
          continue;
        }
        (*bitset_words)[static_cast<size_t>(word_idx)] |= 1ULL << (static_cast<uint64_t>(point_id) & 63ULL);
      }
    }

    void candidate_ids_to_bitset(const std::vector<uint32_t> &candidate_ids, uint64_t words_per_label, uint64_t npoints,
                                 std::vector<uint64_t> *bitset_words) {
      bitset_words->assign(static_cast<size_t>(words_per_label), 0ULL);
      for (uint32_t point_id : candidate_ids) {
        if (point_id >= npoints) {
          continue;
        }
        const uint64_t word_idx = static_cast<uint64_t>(point_id) >> 6U;
        (*bitset_words)[static_cast<size_t>(word_idx)] |= 1ULL << (static_cast<uint64_t>(point_id) & 63ULL);
      }
      apply_tail_mask(bitset_words, npoints);
    }

    void merge_union_sorted(const uint32_t *ids, uint64_t count, std::vector<uint32_t> *candidate_ids) {
      if (count == 0) {
        return;
      }
      if (candidate_ids->empty()) {
        candidate_ids->assign(ids, ids + count);
        return;
      }
      std::vector<uint32_t> merged;
      merged.reserve(candidate_ids->size() + static_cast<size_t>(count));
      std::set_union(candidate_ids->begin(), candidate_ids->end(), ids, ids + count, std::back_inserter(merged));
      candidate_ids->swap(merged);
    }

    void intersect_sorted(const uint32_t *ids, uint64_t count, std::vector<uint32_t> *candidate_ids) {
      if (candidate_ids->empty() || count == 0) {
        candidate_ids->clear();
        return;
      }
      std::vector<uint32_t> merged;
      merged.reserve(std::min(candidate_ids->size(), static_cast<size_t>(count)));
      std::set_intersection(candidate_ids->begin(), candidate_ids->end(), ids, ids + count, std::back_inserter(merged));
      candidate_ids->swap(merged);
    }

    void copy_bitmap_words(const uint64_t *src_words, uint64_t words_per_label, std::vector<uint64_t> *dst_words) {
      dst_words->assign(src_words, src_words + static_cast<size_t>(words_per_label));
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

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("failed to stat densebit sidecar: " + sidecar_path);
    }
    if (static_cast<size_t>(st.st_size) < sizeof(DenseBitsetFileHeaderRawV1)) {
      ::close(fd);
      throw std::runtime_error("densebit sidecar too small: " + sidecar_path);
    }

    void *mmap_addr = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_addr == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("failed to mmap densebit sidecar: " + sidecar_path);
    }

    const auto *header_v1 = reinterpret_cast<const DenseBitsetFileHeaderRawV1 *>(mmap_addr);
    if (header_v1->magic != kDenseBitsetMagic) {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar magic/version mismatch: " + sidecar_path);
    }

    DenseBitsetFileHeaderV1 summary_header;
    summary_header.magic = header_v1->magic;
    summary_header.version = header_v1->version;
    summary_header.npoints = header_v1->npoints;
    summary_header.nlabels = header_v1->nlabels;
    summary_header.words_per_label = header_v1->words_per_label;
    summary_header.nnz = header_v1->nnz;

    if (expected_npoints != 0 && summary_header.npoints != expected_npoints) {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar point count mismatch: " + sidecar_path);
    }

    const uint8_t *mapped_bytes = static_cast<const uint8_t *>(mmap_addr);
    const uint64_t *base_words = nullptr;
    const uint8_t *label_directory = nullptr;
    const uint8_t *payload_base = nullptr;

    if (header_v1->version == kDenseBitsetVersionV1) {
      const uint64_t expected_bytes = sizeof(DenseBitsetFileHeaderRawV1) +
                                      summary_header.nlabels * summary_header.words_per_label * sizeof(uint64_t);
      if (static_cast<uint64_t>(st.st_size) != expected_bytes) {
        ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
        ::close(fd);
        throw std::runtime_error("densebit sidecar size mismatch: " + sidecar_path);
      }
      base_words = reinterpret_cast<const uint64_t *>(mapped_bytes + sizeof(DenseBitsetFileHeaderRawV1));
    } else if (header_v1->version == kDenseBitsetVersionV2) {
      if (static_cast<size_t>(st.st_size) < sizeof(DenseBitsetFileHeaderRawV2)) {
        ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
        ::close(fd);
        throw std::runtime_error("densebit v2 sidecar too small: " + sidecar_path);
      }

      const auto *header_v2 = reinterpret_cast<const DenseBitsetFileHeaderRawV2 *>(mmap_addr);
      const uint64_t entry_bytes = header_v2->nlabels * sizeof(DenseBitsetLabelEntryV2);
      if (header_v2->label_entry_offset < sizeof(DenseBitsetFileHeaderRawV2) ||
          header_v2->label_entry_offset > static_cast<uint64_t>(st.st_size) ||
          entry_bytes > static_cast<uint64_t>(st.st_size) ||
          header_v2->payload_offset > static_cast<uint64_t>(st.st_size) ||
          header_v2->label_entry_offset + entry_bytes > header_v2->payload_offset) {
        ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
        ::close(fd);
        throw std::runtime_error("densebit v2 directory layout mismatch: " + sidecar_path);
      }

      label_directory = mapped_bytes + header_v2->label_entry_offset;
      payload_base = mapped_bytes + header_v2->payload_offset;
      const uint64_t payload_bytes = static_cast<uint64_t>(st.st_size) - header_v2->payload_offset;
      uint64_t dense_labels = 0;
      uint64_t sparse_labels = 0;
      for (uint32_t label = 0; label < header_v2->nlabels; ++label) {
        const auto &entry = entry_at(label_directory, label);
        const auto encoding = static_cast<DenseBitsetEncoding>(entry.encoding);
        if (entry.payload_offset > payload_bytes || entry.payload_size > payload_bytes - entry.payload_offset) {
          ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
          ::close(fd);
          throw std::runtime_error("densebit v2 payload range mismatch: " + sidecar_path);
        }

        if (encoding == DenseBitsetEncoding::kEmpty) {
          if (entry.payload_size != 0 || entry.candidate_count != 0) {
            ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
            ::close(fd);
            throw std::runtime_error("densebit v2 empty entry mismatch: " + sidecar_path);
          }
          continue;
        }
        if (encoding == DenseBitsetEncoding::kBitmap) {
          if (entry.payload_size != summary_header.words_per_label * sizeof(uint64_t)) {
            ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
            ::close(fd);
            throw std::runtime_error("densebit v2 bitmap payload size mismatch: " + sidecar_path);
          }
          ++dense_labels;
          continue;
        }
        if (encoding == DenseBitsetEncoding::kPosting) {
          if (entry.payload_size != entry.candidate_count * sizeof(uint32_t)) {
            ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
            ::close(fd);
            throw std::runtime_error("densebit v2 posting payload size mismatch: " + sidecar_path);
          }
          ++sparse_labels;
          continue;
        }
        ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
        ::close(fd);
        throw std::runtime_error("densebit v2 encoding mismatch: " + sidecar_path);
      }

      if (dense_labels != header_v2->dense_label_count || sparse_labels != header_v2->sparse_label_count) {
        ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
        ::close(fd);
        throw std::runtime_error("densebit v2 encoding counts mismatch: " + sidecar_path);
      }
    } else {
      ::munmap(mmap_addr, static_cast<size_t>(st.st_size));
      ::close(fd);
      throw std::runtime_error("densebit sidecar version mismatch: " + sidecar_path);
    }

    return std::unique_ptr<DenseBitsetIndex>(new DenseBitsetIndex(sidecar_path, fd, mmap_addr,
                                                                  static_cast<size_t>(st.st_size), summary_header,
                                                                  base_words, label_directory, payload_base));
  }

  void DenseBitsetIndex::write_atomically(const std::string &sidecar_path, uint64_t npoints, uint64_t nlabels,
                                          const std::vector<std::vector<uint32_t>> &labels_by_point) {
    if (labels_by_point.size() != static_cast<size_t>(npoints)) {
      throw std::invalid_argument("labels_by_point size does not match densebit npoints");
    }

    const uint64_t words_per_label = (npoints + 63ULL) / 64ULL;
    const uint64_t posting_threshold = posting_threshold_for_words(words_per_label);
    std::vector<uint64_t> label_counts(static_cast<size_t>(nlabels), 0ULL);
    uint64_t nnz = 0;
    for (const auto &point_labels : labels_by_point) {
      for (uint32_t label : point_labels) {
        if (label >= nlabels) {
          throw std::runtime_error("label id exceeds densebit label universe");
        }
        ++label_counts[static_cast<size_t>(label)];
        ++nnz;
      }
    }

    std::vector<DenseBitsetLabelEntryV2> entries(static_cast<size_t>(nlabels));
    std::vector<uint64_t> label_storage_offsets(static_cast<size_t>(nlabels), 0ULL);
    uint64_t total_payload_bytes = 0;
    uint64_t dense_label_count = 0;
    uint64_t sparse_label_count = 0;
    uint64_t bitmap_payload_words = 0;
    uint64_t posting_payload_ids = 0;

    for (uint32_t label = 0; label < nlabels; ++label) {
      auto &entry = entries[static_cast<size_t>(label)];
      entry.candidate_count = label_counts[static_cast<size_t>(label)];
      if (entry.candidate_count == 0) {
        entry.encoding = static_cast<uint16_t>(DenseBitsetEncoding::kEmpty);
        continue;
      }

      if (entry.candidate_count < posting_threshold) {
        entry.encoding = static_cast<uint16_t>(DenseBitsetEncoding::kPosting);
        entry.payload_offset = align_up(total_payload_bytes, alignof(uint32_t));
        entry.payload_size = entry.candidate_count * sizeof(uint32_t);
        label_storage_offsets[static_cast<size_t>(label)] = posting_payload_ids;
        posting_payload_ids += entry.candidate_count;
        ++sparse_label_count;
      } else {
        entry.encoding = static_cast<uint16_t>(DenseBitsetEncoding::kBitmap);
        entry.payload_offset = align_up(total_payload_bytes, alignof(uint64_t));
        entry.payload_size = words_per_label * sizeof(uint64_t);
        label_storage_offsets[static_cast<size_t>(label)] = bitmap_payload_words;
        bitmap_payload_words += words_per_label;
        ++dense_label_count;
      }
      total_payload_bytes = entry.payload_offset + entry.payload_size;
    }

    if (bitmap_payload_words > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        posting_payload_ids > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      throw std::runtime_error("densebit payload would overflow addressable memory");
    }

    std::vector<uint64_t> bitmap_payload(static_cast<size_t>(bitmap_payload_words), 0ULL);
    std::vector<uint32_t> posting_payload(static_cast<size_t>(posting_payload_ids), 0U);
    std::vector<uint64_t> posting_write_offsets(static_cast<size_t>(nlabels), 0ULL);
    for (uint64_t point_id = 0; point_id < npoints; ++point_id) {
      const uint64_t word_idx = point_id >> 6U;
      const uint64_t bit_mask = 1ULL << (point_id & 63ULL);
      for (uint32_t label : labels_by_point[static_cast<size_t>(point_id)]) {
        const auto &entry = entries[static_cast<size_t>(label)];
        const uint64_t storage_offset = label_storage_offsets[static_cast<size_t>(label)];
        if (entry.encoding == static_cast<uint16_t>(DenseBitsetEncoding::kPosting)) {
          posting_payload[static_cast<size_t>(storage_offset + posting_write_offsets[static_cast<size_t>(label)]++)] =
              static_cast<uint32_t>(point_id);
        } else if (entry.encoding == static_cast<uint16_t>(DenseBitsetEncoding::kBitmap)) {
          bitmap_payload[static_cast<size_t>(storage_offset + word_idx)] |= bit_mask;
        }
      }
    }

    DenseBitsetFileHeaderRawV2 header;
    header.magic = kDenseBitsetMagic;
    header.version = kDenseBitsetVersionV2;
    header.npoints = npoints;
    header.nlabels = nlabels;
    header.words_per_label = words_per_label;
    header.nnz = nnz;
    header.label_entry_offset = sizeof(DenseBitsetFileHeaderRawV2);
    header.payload_offset =
        align_up(header.label_entry_offset + entries.size() * sizeof(DenseBitsetLabelEntryV2), alignof(uint64_t));
    header.posting_threshold = posting_threshold;
    header.dense_label_count = dense_label_count;
    header.sparse_label_count = sparse_label_count;

    const std::string tmp_path = sidecar_path + ".tmp";
    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      throw std::runtime_error("failed to open densebit temp file: " + tmp_path);
    }

    bool fd_finalized = false;
    try {
      write_all_or_throw(fd, &header, sizeof(header), tmp_path);
      if (!entries.empty()) {
        write_all_or_throw(fd, entries.data(), entries.size() * sizeof(DenseBitsetLabelEntryV2), tmp_path);
      }

      const uint64_t directory_bytes = header.label_entry_offset + entries.size() * sizeof(DenseBitsetLabelEntryV2);
      if (header.payload_offset > directory_bytes) {
        write_padding_or_throw(fd, header.payload_offset - directory_bytes, tmp_path);
      }

      uint64_t written_payload_bytes = 0;
      for (uint32_t label = 0; label < nlabels; ++label) {
        const auto &entry = entries[static_cast<size_t>(label)];
        if (entry.payload_offset > written_payload_bytes) {
          write_padding_or_throw(fd, entry.payload_offset - written_payload_bytes, tmp_path);
          written_payload_bytes = entry.payload_offset;
        }
        if (entry.payload_size == 0) {
          continue;
        }

        const uint64_t storage_offset = label_storage_offsets[static_cast<size_t>(label)];
        if (entry.encoding == static_cast<uint16_t>(DenseBitsetEncoding::kPosting)) {
          write_all_or_throw(fd, posting_payload.data() + static_cast<size_t>(storage_offset),
                             static_cast<size_t>(entry.payload_size), tmp_path);
        } else if (entry.encoding == static_cast<uint16_t>(DenseBitsetEncoding::kBitmap)) {
          write_all_or_throw(fd, bitmap_payload.data() + static_cast<size_t>(storage_offset),
                             static_cast<size_t>(entry.payload_size), tmp_path);
        }
        written_payload_bytes = entry.payload_offset + entry.payload_size;
      }

      if (::fsync(fd) != 0) {
        throw std::runtime_error("failed to fsync densebit temp file: " + tmp_path);
      }
      if (::close(fd) != 0) {
        fd_finalized = true;
        throw std::runtime_error("failed to close densebit temp file: " + tmp_path);
      }
      fd_finalized = true;
      if (::rename(tmp_path.c_str(), sidecar_path.c_str()) != 0) {
        throw std::runtime_error("failed to publish densebit sidecar: " + sidecar_path);
      }
    } catch (...) {
      if (!fd_finalized) {
        ::close(fd);
      }
      ::unlink(tmp_path.c_str());
      throw;
    }
  }

  DenseBitsetIndex::DenseBitsetIndex(std::string sidecar_path, int fd, void *mmap_addr, size_t mmap_len,
                                     DenseBitsetFileHeaderV1 header, const uint64_t *base_words,
                                     const uint8_t *label_directory, const uint8_t *payload_base)
      : sidecar_path_(std::move(sidecar_path)), fd_(fd), mmap_addr_(mmap_addr), mmap_len_(mmap_len), header_(header),
        file_version_(header.version), base_words_(base_words), label_directory_(label_directory),
        payload_base_(payload_base) {
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
    for (uint32_t label = 0; label < header_.nlabels; ++label) {
      const auto encoding = label_encoding(file_version_, label_directory_, label);
      if (encoding == DenseBitsetEncoding::kEmpty) {
        continue;
      }
      if (encoding == DenseBitsetEncoding::kPosting) {
        const uint32_t *ids = posting_ids(label_directory_, payload_base_, label);
        const uint64_t count = posting_count(label_directory_, label);
        for (uint64_t idx = 0; idx < count; ++idx) {
          const uint32_t point_id = ids[idx];
          if (point_id < header_.npoints) {
            (*output)[static_cast<size_t>(point_id)].push_back(label);
          }
        }
        continue;
      }

      const uint64_t *label_words =
          bitmap_words(file_version_, base_words_, label_directory_, payload_base_, header_.words_per_label, label);
      for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
        uint64_t word = label_words[static_cast<size_t>(word_idx)];
        if (word_idx + 1 == header_.words_per_label) {
          word &= dense_tail_mask(header_.npoints);
        }
        while (word != 0) {
          const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
          const uint64_t point_id = word_idx * 64ULL + bit;
          if (point_id < header_.npoints) {
            (*output)[static_cast<size_t>(point_id)].push_back(label);
          }
          word &= (word - 1);
        }
      }
    }
  }

  uint64_t DenseBitsetIndex::count_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                              HybridQueryScratch *scratch) const {
    return compute_candidates(kind, labels, scratch);
  }

  void DenseBitsetIndex::materialize_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                                HybridQueryScratch *scratch, std::vector<uint32_t> *output) const {
    compute_candidates(kind, labels, scratch);
    if (!scratch->candidate_ids.empty()) {
      *output = scratch->candidate_ids;
      return;
    }
    materialize_from_scratch(scratch, output);
  }

  uint64_t DenseBitsetIndex::compute_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                                HybridQueryScratch *scratch) const {
    if (scratch == nullptr) {
      throw std::invalid_argument("HybridQueryScratch must not be null");
    }

    normalize_labels(labels, scratch);
    scratch->bitset_words.clear();
    scratch->candidate_ids.clear();

    if (kind == HybridFilterKind::kUnsupported) {
      return 0;
    }

    if (kind == HybridFilterKind::kSubset && scratch->normalized_labels.empty()) {
      scratch->bitset_words.assign(static_cast<size_t>(header_.words_per_label), std::numeric_limits<uint64_t>::max());
      apply_tail_mask(&scratch->bitset_words, header_.npoints);
      return header_.npoints;
    }

    const bool is_range = kind == HybridFilterKind::kRange;
    uint32_t range_low = 0;
    uint32_t range_high = 0;
    if (is_range) {
      if (scratch->normalized_labels.empty() || header_.nlabels == 0) {
        return 0;
      }
      range_low = scratch->normalized_labels.front();
      if (range_low >= header_.nlabels) {
        return 0;
      }
      range_high = std::min<uint32_t>(scratch->normalized_labels.back(), static_cast<uint32_t>(header_.nlabels - 1));
    }

    bool all_postings = !is_range;
    if (!is_range) {
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          if (kind == HybridFilterKind::kSubset) {
            return 0;
          }
          continue;
        }
        const auto encoding = label_encoding(file_version_, label_directory_, label);
        if (encoding != DenseBitsetEncoding::kPosting && encoding != DenseBitsetEncoding::kEmpty) {
          all_postings = false;
          break;
        }
      }
    }

    if (all_postings && kind == HybridFilterKind::kIntersect) {
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          continue;
        }
        if (label_encoding(file_version_, label_directory_, label) != DenseBitsetEncoding::kPosting) {
          continue;
        }
        merge_union_sorted(posting_ids(label_directory_, payload_base_, label), posting_count(label_directory_, label),
                           &scratch->candidate_ids);
      }
      candidate_ids_to_bitset(scratch->candidate_ids, header_.words_per_label, header_.npoints, &scratch->bitset_words);
      return static_cast<uint64_t>(scratch->candidate_ids.size());
    }

    if (all_postings && kind == HybridFilterKind::kSubset) {
      bool first_posting = true;
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          return 0;
        }
        const auto encoding = label_encoding(file_version_, label_directory_, label);
        if (encoding == DenseBitsetEncoding::kEmpty) {
          return 0;
        }
        if (encoding != DenseBitsetEncoding::kPosting) {
          continue;
        }
        const uint32_t *ids = posting_ids(label_directory_, payload_base_, label);
        const uint64_t count = posting_count(label_directory_, label);
        if (first_posting) {
          scratch->candidate_ids.assign(ids, ids + count);
          first_posting = false;
        } else {
          intersect_sorted(ids, count, &scratch->candidate_ids);
        }
        if (scratch->candidate_ids.empty()) {
          return 0;
        }
      }
      candidate_ids_to_bitset(scratch->candidate_ids, header_.words_per_label, header_.npoints, &scratch->bitset_words);
      return static_cast<uint64_t>(scratch->candidate_ids.size());
    }

    scratch->bitset_words.assign(static_cast<size_t>(header_.words_per_label), 0ULL);
    if (kind == HybridFilterKind::kSubset) {
      bool initialized = false;
      std::vector<uint64_t> posting_bitset(static_cast<size_t>(header_.words_per_label), 0ULL);
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          std::fill(scratch->bitset_words.begin(), scratch->bitset_words.end(), 0ULL);
          return 0;
        }
        const auto encoding = label_encoding(file_version_, label_directory_, label);
        if (encoding == DenseBitsetEncoding::kEmpty) {
          std::fill(scratch->bitset_words.begin(), scratch->bitset_words.end(), 0ULL);
          return 0;
        }
        if (encoding == DenseBitsetEncoding::kBitmap) {
          const uint64_t *label_words =
              bitmap_words(file_version_, base_words_, label_directory_, payload_base_, header_.words_per_label, label);
          if (!initialized) {
            copy_bitmap_words(label_words, header_.words_per_label, &scratch->bitset_words);
            initialized = true;
          } else {
            for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
              scratch->bitset_words[static_cast<size_t>(word_idx)] &= label_words[static_cast<size_t>(word_idx)];
            }
          }
          continue;
        }

        std::fill(posting_bitset.begin(), posting_bitset.end(), 0ULL);
        or_posting_into_bitset(posting_ids(label_directory_, payload_base_, label),
                               posting_count(label_directory_, label), &posting_bitset);
        if (!initialized) {
          scratch->bitset_words.swap(posting_bitset);
          initialized = true;
        } else {
          for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
            scratch->bitset_words[static_cast<size_t>(word_idx)] &= posting_bitset[static_cast<size_t>(word_idx)];
          }
        }
      }
    } else if (kind == HybridFilterKind::kIntersect) {
      for (uint32_t label : scratch->normalized_labels) {
        if (label >= header_.nlabels) {
          continue;
        }
        const auto encoding = label_encoding(file_version_, label_directory_, label);
        if (encoding == DenseBitsetEncoding::kEmpty) {
          continue;
        }
        if (encoding == DenseBitsetEncoding::kPosting) {
          or_posting_into_bitset(posting_ids(label_directory_, payload_base_, label),
                                 posting_count(label_directory_, label), &scratch->bitset_words);
          continue;
        }
        const uint64_t *label_words =
            bitmap_words(file_version_, base_words_, label_directory_, payload_base_, header_.words_per_label, label);
        for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
          scratch->bitset_words[static_cast<size_t>(word_idx)] |= label_words[static_cast<size_t>(word_idx)];
        }
      }
    } else if (kind == HybridFilterKind::kRange) {
      for (uint32_t label = range_low; label <= range_high; ++label) {
        const auto encoding = label_encoding(file_version_, label_directory_, label);
        if (encoding == DenseBitsetEncoding::kEmpty) {
          if (label == std::numeric_limits<uint32_t>::max()) {
            break;
          }
          continue;
        }
        if (encoding == DenseBitsetEncoding::kPosting) {
          or_posting_into_bitset(posting_ids(label_directory_, payload_base_, label),
                                 posting_count(label_directory_, label), &scratch->bitset_words);
        } else {
          const uint64_t *label_words =
              bitmap_words(file_version_, base_words_, label_directory_, payload_base_, header_.words_per_label, label);
          for (uint64_t word_idx = 0; word_idx < header_.words_per_label; ++word_idx) {
            scratch->bitset_words[static_cast<size_t>(word_idx)] |= label_words[static_cast<size_t>(word_idx)];
          }
        }
        if (label == std::numeric_limits<uint32_t>::max()) {
          break;
        }
      }
    }

    apply_tail_mask(&scratch->bitset_words, header_.npoints);
    return popcount_words(scratch->bitset_words);
  }

  void DenseBitsetIndex::normalize_labels(const std::vector<uint32_t> &labels, HybridQueryScratch *scratch) const {
    scratch->normalized_labels = labels;
    std::sort(scratch->normalized_labels.begin(), scratch->normalized_labels.end());
    scratch->normalized_labels.erase(std::unique(scratch->normalized_labels.begin(), scratch->normalized_labels.end()),
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