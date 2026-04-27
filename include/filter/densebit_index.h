#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pipeann {
  enum class HybridFilterKind : uint8_t {
    kUnsupported = 0,
    kIntersect = 1,
    kSubset = 2,
    kRange = 3,
  };

  struct DenseBitsetFileHeaderV1 {
    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t npoints = 0;
    uint64_t nlabels = 0;
    uint64_t words_per_label = 0;
    uint64_t nnz = 0;
  };

  struct HybridQueryScratch {
    std::vector<uint64_t> bitset_words;
    std::vector<uint32_t> candidate_ids;
    std::vector<uint32_t> normalized_labels;
  };

  class DenseBitsetIndex {
   public:
    static std::string default_sidecar_path(const std::string &index_prefix);
    static std::unique_ptr<DenseBitsetIndex> load(const std::string &sidecar_path, uint64_t expected_npoints = 0);
    static void write_atomically(const std::string &sidecar_path, uint64_t npoints, uint64_t nlabels,
                   const std::vector<std::vector<uint32_t>> &labels_by_point);

    ~DenseBitsetIndex();

    DenseBitsetIndex(const DenseBitsetIndex &) = delete;
    DenseBitsetIndex &operator=(const DenseBitsetIndex &) = delete;

    const DenseBitsetFileHeaderV1 &header() const;
  void materialize_labels_by_point(std::vector<std::vector<uint32_t>> *output) const;

    uint64_t count_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                              HybridQueryScratch *scratch) const;
    void materialize_candidates(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                                HybridQueryScratch *scratch, std::vector<uint32_t> *output) const;

   private:
    DenseBitsetIndex(std::string sidecar_path, int fd, void *mmap_addr, size_t mmap_len,
                     DenseBitsetFileHeaderV1 header, const uint64_t *base_words);

    uint64_t compute_bitset(HybridFilterKind kind, const std::vector<uint32_t> &labels,
                            HybridQueryScratch *scratch) const;
    void normalize_labels(const std::vector<uint32_t> &labels, HybridQueryScratch *scratch) const;
    void materialize_from_scratch(HybridQueryScratch *scratch, std::vector<uint32_t> *output) const;

    std::string sidecar_path_;
    int fd_ = -1;
    void *mmap_addr_ = nullptr;
    size_t mmap_len_ = 0;
    DenseBitsetFileHeaderV1 header_{};
    const uint64_t *base_words_ = nullptr;
  };
}  // namespace pipeann
