#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <omp.h>
#include <ssd_index.h>

#include "distance.h"
#include "filter/label.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "utils/log.h"
#include "utils/timer.h"

namespace {
constexpr uint64_t kDenseBitsetMagic = 0x54494245534E4544ULL;
constexpr uint64_t kDenseBitsetVersion = 1;

struct DenseBitsetFileHeader {
  uint64_t magic = kDenseBitsetMagic;
  uint64_t version = kDenseBitsetVersion;
  uint64_t npoints = 0;
  uint64_t nlabels = 0;
  uint64_t words_per_label = 0;
  uint64_t nnz = 0;
};

bool load_spmat_header(const std::string &label_file, int64_t &nrow, int64_t &ncol, int64_t &nnz) {
  std::ifstream reader(label_file, std::ios::binary);
  if (!reader.is_open()) {
    LOG(ERROR) << "Cannot open label file: " << label_file;
    return false;
  }

  reader.read(reinterpret_cast<char *>(&nrow), sizeof(int64_t));
  reader.read(reinterpret_cast<char *>(&ncol), sizeof(int64_t));
  reader.read(reinterpret_cast<char *>(&nnz), sizeof(int64_t));
  return reader.good();
}

uint64_t dense_words_per_label(uint64_t npoints) {
  return DIV_ROUND_UP(npoints, 64ULL);
}

uint64_t dense_tail_mask(uint64_t npoints) {
  uint64_t rem = npoints % 64ULL;
  return rem == 0 ? std::numeric_limits<uint64_t>::max() : ((1ULL << rem) - 1ULL);
}

std::vector<uint32_t> normalize_query_labels(const std::vector<uint32_t> &labels) {
  std::vector<uint32_t> normalized = labels;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
  return normalized;
}

class DenseBitsetIndex {
 public:
  explicit DenseBitsetIndex(const std::string &label_file) : label_file_(label_file), sidecar_file_(label_file + ".densebit") {
    int64_t nrow = 0, ncol = 0, nnz = 0;
    if (!load_spmat_header(label_file_, nrow, ncol, nnz)) {
      LOG(ERROR) << "Failed to read spmat header from " << label_file_;
      crash();
    }

    fd_ = open(sidecar_file_.c_str(), O_RDONLY);
    if (fd_ < 0) {
      LOG(ERROR) << "Densebit sidecar missing: " << sidecar_file_;
      crash();
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0) {
      LOG(ERROR) << "Failed to stat densebit sidecar: " << sidecar_file_;
      crash();
    }
    if (static_cast<size_t>(st.st_size) < sizeof(DenseBitsetFileHeader)) {
      LOG(ERROR) << "Densebit sidecar too small: " << sidecar_file_;
      crash();
    }

    mmap_len_ = static_cast<size_t>(st.st_size);
    mmap_addr_ = mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
      LOG(ERROR) << "Failed to mmap densebit sidecar: " << sidecar_file_;
      crash();
    }

    const auto *header = reinterpret_cast<const DenseBitsetFileHeader *>(mmap_addr_);
    const uint64_t expected_words = dense_words_per_label(static_cast<uint64_t>(nrow));
    const uint64_t expected_size = sizeof(DenseBitsetFileHeader)
                                 + static_cast<uint64_t>(ncol) * expected_words * sizeof(uint64_t);
    if (header->magic != kDenseBitsetMagic || header->version != kDenseBitsetVersion ||
        header->npoints != static_cast<uint64_t>(nrow) || header->nlabels != static_cast<uint64_t>(ncol) ||
        header->words_per_label != expected_words || header->nnz != static_cast<uint64_t>(nnz) ||
        static_cast<uint64_t>(mmap_len_) != expected_size) {
      LOG(ERROR) << "Densebit sidecar metadata mismatch for " << sidecar_file_;
      crash();
    }

    total_points_ = header->npoints;
    nlabels_ = header->nlabels;
    words_per_label_ = header->words_per_label;
    base_words_ = reinterpret_cast<const uint64_t *>(static_cast<const char *>(mmap_addr_) + sizeof(DenseBitsetFileHeader));
  }

  ~DenseBitsetIndex() {
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
      munmap(mmap_addr_, mmap_len_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  size_t total_points() const {
    return total_points_;
  }

  std::vector<uint32_t> build_candidates(const std::vector<uint32_t> &raw_query_labels,
                                         const std::string &selector_type) {
    const std::vector<uint32_t> query_labels = normalize_query_labels(raw_query_labels);

    if (selector_type == "intersect") {
      return union_labels(query_labels);
    }
    if (selector_type == "subset") {
      return intersect_labels(query_labels);
    }

    LOG(ERROR) << "Unsupported selector_type for densebit prefilter search: " << selector_type;
    return {};
  }

 private:
  std::vector<uint32_t> union_labels(const std::vector<uint32_t> &query_labels) const {
    if (query_labels.empty()) {
      return {};
    }

    std::vector<uint64_t> accum(words_per_label_, 0ULL);
    for (uint32_t label : query_labels) {
      if (label >= nlabels_) {
        continue;
      }
      const uint64_t *label_words = base_words_ + static_cast<size_t>(label) * words_per_label_;
      for (uint64_t i = 0; i < words_per_label_; ++i) {
        accum[i] |= label_words[i];
      }
    }
    return materialize_candidates(accum);
  }

  std::vector<uint32_t> intersect_labels(const std::vector<uint32_t> &query_labels) const {
    if (query_labels.empty()) {
      std::vector<uint32_t> all_points(total_points_);
      for (uint32_t point_id = 0; point_id < total_points_; ++point_id) {
        all_points[point_id] = point_id;
      }
      return all_points;
    }

    std::vector<uint64_t> accum(words_per_label_, std::numeric_limits<uint64_t>::max());
    bool first = true;
    for (uint32_t label : query_labels) {
      if (label >= nlabels_) {
        return {};
      }
      const uint64_t *label_words = base_words_ + static_cast<size_t>(label) * words_per_label_;
      if (first) {
        for (uint64_t i = 0; i < words_per_label_; ++i) {
          accum[i] = label_words[i];
        }
        first = false;
        continue;
      }
      for (uint64_t i = 0; i < words_per_label_; ++i) {
        accum[i] &= label_words[i];
      }
    }
    return materialize_candidates(accum);
  }

  std::vector<uint32_t> materialize_candidates(std::vector<uint64_t> &accum) const {
    if (!accum.empty()) {
      accum.back() &= dense_tail_mask(total_points_);
    }
    size_t reserve = 0;
    for (uint64_t word : accum) {
      reserve += static_cast<size_t>(__builtin_popcountll(word));
    }
    std::vector<uint32_t> candidates;
    candidates.reserve(reserve);
    for (uint64_t word_idx = 0; word_idx < accum.size(); ++word_idx) {
      uint64_t word = accum[word_idx];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
        candidates.push_back(static_cast<uint32_t>(word_idx * 64 + bit));
        word &= (word - 1);
      }
    }
    return candidates;
  }

  size_t total_points_ = 0;
  size_t nlabels_ = 0;
  size_t words_per_label_ = 0;
  std::string label_file_;
  std::string sidecar_file_;
  int fd_ = -1;
  void *mmap_addr_ = nullptr;
  size_t mmap_len_ = 0;
  const uint64_t *base_words_ = nullptr;
};

size_t compute_prefilter_rerank_l(uint64_t k_search, size_t candidate_count, size_t total_points) {
  if (candidate_count == 0) {
    return 0;
  }

  double selectivity = total_points > 0 ? static_cast<double>(candidate_count) / static_cast<double>(total_points) : 1.0;
  size_t target = 192;
  if (selectivity <= 0.005) {
    target = 96;
  } else if (selectivity <= 0.02) {
    target = 128;
  } else if (selectivity <= 0.1) {
    target = 160;
  }

  target = std::max<size_t>(target, static_cast<size_t>(k_search));
  return std::min(target, candidate_count);
}

template<typename T>
size_t prefilter_search(pipeann::SSDIndex<T> &index, pipeann::AbstractNeighbor<T> &nbr_handler,
                        const pipeann::Distance<T> &distance, const T *query, uint64_t k_search,
                        const std::vector<uint32_t> &candidate_ids, size_t total_points, uint32_t *res_tags,
                        float *res_dists, pipeann::QueryStats *stats) {
  if (candidate_ids.empty()) {
    if (stats != nullptr) {
      *stats = {};
    }
    return 0;
  }

  pipeann::Timer timer;
  pipeann::QueryBuffer<T> *query_buf = index.pop_query_buf(query);
  nbr_handler.initialize_query(query_buf->aligned_query_T, query_buf);

  const size_t rerank_l = compute_prefilter_rerank_l(k_search, candidate_ids.size(), total_points);
  using PQCandidate = std::pair<float, uint32_t>;
  std::priority_queue<PQCandidate> top_candidates;

  constexpr size_t kBatchSize = MAX_N_EDGES;
  for (size_t offset = 0; offset < candidate_ids.size(); offset += kBatchSize) {
    const size_t current_batch = std::min(kBatchSize, candidate_ids.size() - offset);
    nbr_handler.compute_dists(query_buf, candidate_ids.data() + offset, current_batch);

    for (size_t i = 0; i < current_batch; ++i) {
      const float approx_dist = query_buf->aligned_dist_scratch[i];
      const uint32_t point_id = candidate_ids[offset + i];
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
  for (const auto &[approx_dist, point_id] : shortlist) {
    (void) approx_dist;
    if (index.get_vector_by_id(point_id, vector_buf) != 0) {
      continue;
    }
    const float exact_dist =
        distance.compare(query_buf->aligned_query_T, vector_buf, static_cast<unsigned>(index.meta_.data_dim));
    exact_results.emplace_back(exact_dist, point_id);
  }

  const size_t result_count = std::min(static_cast<size_t>(k_search), exact_results.size());
  std::partial_sort(exact_results.begin(), exact_results.begin() + result_count, exact_results.end());
  for (size_t i = 0; i < result_count; ++i) {
    res_tags[i] = index.id2tag(exact_results[i].second);
    if (res_dists != nullptr) {
      res_dists[i] = exact_results[i].first;
    }
  }

  index.push_query_buf(query_buf);

  if (stats != nullptr) {
    stats->total_us = static_cast<double>(timer.elapsed());
    stats->n_cmps = static_cast<double>(candidate_ids.size());
    stats->n_ios = static_cast<double>(shortlist.size());
    stats->n_hops = 0.0;
  }
  return result_count;
}

template<typename T>
int search_disk_index(int argc, char **argv) {
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *gt_tags = nullptr;
  size_t query_num = 0, query_dim = 0, gt_num = 0, gt_dim = 0;
  std::vector<uint64_t> Lvec;

  int arg_index = 2;
  const std::string index_prefix_path(argv[arg_index++]);
  const uint32_t num_threads = std::atoi(argv[arg_index++]);
  const uint32_t beamwidth = std::atoi(argv[arg_index++]);
  const std::string query_bin(argv[arg_index++]);
  const std::string truthset_bin(argv[arg_index++]);
  const uint64_t recall_at = std::atoi(argv[arg_index++]);
  const std::string dist_metric(argv[arg_index++]);
  const std::string nbr_type(argv[arg_index++]);
  const std::string selector_type(argv[arg_index++]);
  const std::string query_label_file(argv[arg_index++]);
  const std::string data_label_file(argv[arg_index++]);

  const pipeann::Metric metric = pipeann::get_metric(dist_metric);

  for (int ctr = arg_index; ctr < argc; ++ctr) {
    const uint64_t cur_l = std::atoi(argv[ctr]);
    if (cur_l >= recall_at) {
      Lvec.push_back(cur_l);
    }
  }

  if (Lvec.empty()) {
    LOG(ERROR) << "No valid Lsearch found. Lsearch must be >= recall_at";
    return -1;
  }

  if (selector_type != "intersect" && selector_type != "subset") {
    LOG(ERROR) << "Standalone prefilter search only supports intersect/subset selectors";
    return -1;
  }

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  bool calc_recall_flag = false;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    if (gt_num != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and ground truth data";
      return -1;
    }
    calc_recall_flag = true;
  }

  pipeann::SpmatLabel query_labels(query_label_file);
  if (query_labels.labels_.size() != query_num) {
    LOG(ERROR) << "Mismatch in number of queries and query labels";
    return -1;
  }

  DenseBitsetIndex posting_index(data_label_file);
  std::vector<std::vector<uint32_t>> per_query_candidates(query_num);
  size_t total_candidates = 0;
  for (size_t i = 0; i < query_num; ++i) {
    per_query_candidates[i] = posting_index.build_candidates(query_labels.labels_[i], selector_type);
    total_candidates += per_query_candidates[i].size();
  }
  LOG(INFO) << "Prepared " << query_num << " prefilter candidate sets. Mean candidates/query="
            << (query_num == 0 ? 0.0 : static_cast<double>(total_candidates) / static_cast<double>(query_num));

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  if (nbr_handler == nullptr) {
    LOG(ERROR) << "Unknown neighbor type: " << nbr_type;
    return -1;
  }

  std::unique_ptr<pipeann::SSDIndex<T>> index(new pipeann::SSDIndex<T>(metric, reader, nbr_handler, true));
  index->enable_low_memory_search_mode(true);
  if (index->load(index_prefix_path.c_str(), num_threads, false) != 0) {
    return -1;
  }

  std::unique_ptr<pipeann::Distance<T>> distance(pipeann::get_distance_function<T>(metric));
  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    auto *stats = new pipeann::QueryStats[query_num];
    const uint64_t L = Lvec[test_id];

    query_result_tags[test_id].assign(
        static_cast<size_t>(recall_at) * query_num,
        std::numeric_limits<uint32_t>::max());
    query_result_dists[test_id].assign(
        static_cast<size_t>(recall_at) * query_num,
        std::numeric_limits<float>::infinity());

    const auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
      (void) L;
      prefilter_search(
          *index,
          *nbr_handler,
          *distance,
          query + (i * query_dim),
          recall_at,
          per_query_candidates[i],
          posting_index.total_points(),
          query_result_tags[test_id].data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at),
          stats + i);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    const float qps = diff.count() > 0.0 ? static_cast<float>(query_num / diff.count()) : 0.0f;
    const float mean_latency =
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.total_us; });
    const float latency_999 =
        pipeann::get_percentile_stats(stats, query_num, 0.999f, [](const pipeann::QueryStats &s) { return s.total_us; });
    const float mean_hops =
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_hops; });
    const float mean_ios =
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_ios; });

    float recall = 0.0f;
    if (calc_recall_flag) {
      recall = pipeann::calculate_recall(
          static_cast<uint32_t>(query_num),
          gt_ids,
          gt_dists,
          static_cast<uint32_t>(gt_dim),
          query_result_tags[test_id].data(),
          static_cast<uint32_t>(recall_at),
          static_cast<uint32_t>(recall_at));
    }

    if (output) {
      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall;
      }
      std::cout << std::endl;
    }

    delete[] stats;
  };

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  const std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12)
            << "Mean IOs";
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string;
  }
  std::cout << std::endl;
  std::cout << std::string(90, '=') << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); ++test_id) {
    run_tests(test_id, true);
  }

  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 13) {
    std::cout << "Usage: " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <query_file.bin>"
              << " <truthset.bin (use \"null\" for none)>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/dummy)>"
              << " <selector_type (intersect/subset)>"
              << " <query_label.spmat>"
              << " <data_label.spmat>"
              << " <L1> [L2] ..." << std::endl;
    return -1;
  }

  const std::string index_type = argv[1];
  if (index_type == "float") {
    return search_disk_index<float>(argc, argv);
  }
  if (index_type == "int8") {
    return search_disk_index<int8_t>(argc, argv);
  }
  if (index_type == "uint8") {
    return search_disk_index<uint8_t>(argc, argv);
  }

  std::cout << "Unsupported index type: " << index_type << ". Use float/int8/uint8" << std::endl;
  return -1;
}
