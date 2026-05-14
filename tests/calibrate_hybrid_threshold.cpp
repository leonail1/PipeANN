#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include <ssd_index.h>

#include "filter/densebit_index.h"
#include "filter/hybrid_metadata.h"
#include "filter/label.h"
#include "filter/selector.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "utils/log.h"

namespace {
constexpr uint64_t kMetadataValidFlag = 1ULL << 0;
constexpr uint64_t kCalibrationValidFlag = 1ULL << 1;
constexpr uint64_t kAllowPrefilterFlag = 1ULL << 2;
constexpr uint64_t kCalibrationSeed = 20260423ULL;
constexpr uint32_t kDefaultAutoCalibrationQueriesPerBucket = 100;
constexpr std::array<double, 8> kDefaultAutoSelectivities = {0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00};

struct HybridCalibrationInput {
  pipeann::HybridFilterKind filter_kind = pipeann::HybridFilterKind::kUnsupported;
  std::string selector_type;
  std::string query_bin;
  std::string query_label_file;
  uint32_t sample_limit = 0;
};

struct HybridCalibrationConfig {
  uint64_t k = 0;
  uint32_t mem_L = 0;
  uint64_t l_search = 0;
  uint32_t beamwidth = 0;
  std::vector<HybridCalibrationInput> datasets;
};

struct ParsedCalibrationCli {
  std::string index_prefix_path;
  uint32_t num_threads = 0;
  std::string dist_metric;
  std::string nbr_type;
  HybridCalibrationConfig config;
  bool auto_selectivity = false;
  std::string auto_base_bin;
  std::string auto_selector_type;
  pipeann::HybridFilterKind auto_filter_kind = pipeann::HybridFilterKind::kUnsupported;
  uint32_t auto_queries_per_bucket = kDefaultAutoCalibrationQueriesPerBucket;
  uint64_t auto_seed = kCalibrationSeed;
  std::vector<double> auto_selectivities;
};

struct DatasetSummary {
  std::string selector_type;
  std::string query_bin;
  std::string query_label_file;
  uint64_t total_queries = 0;
  uint64_t sampled_queries = 0;
  uint64_t empty_queries = 0;
};

struct BucketSamples {
  std::vector<double> prefilter_us;
  std::vector<double> graph_us;
};

struct SyntheticBitsetSelector : public pipeann::AbstractSelector {
  const std::vector<uint64_t> *bitset_words = nullptr;
  uint64_t npoints = 0;

  bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
    (void) query_labels;
    (void) target_labels;
    if (bitset_words == nullptr || target_id >= npoints) {
      return false;
    }
    const uint64_t word_idx = static_cast<uint64_t>(target_id) >> 6U;
    if (word_idx >= bitset_words->size()) {
      return false;
    }
    const uint64_t bit_mask = 1ULL << (static_cast<uint64_t>(target_id) & 63ULL);
    return ((*bitset_words)[static_cast<size_t>(word_idx)] & bit_mask) != 0;
  }
};

pipeann::HybridFilterKind parse_filter_kind(const std::string &selector_type) {
  if (selector_type == "intersect") {
    return pipeann::HybridFilterKind::kIntersect;
  }
  if (selector_type == "subset") {
    return pipeann::HybridFilterKind::kSubset;
  }
  if (selector_type == "range") {
    return pipeann::HybridFilterKind::kRange;
  }
  return pipeann::HybridFilterKind::kUnsupported;
}

uint64_t selector_mask_for_kind(pipeann::HybridFilterKind kind) {
  switch (kind) {
    case pipeann::HybridFilterKind::kIntersect:
      return 1ULL;
    case pipeann::HybridFilterKind::kSubset:
      return 2ULL;
    case pipeann::HybridFilterKind::kRange:
      return 4ULL;
    case pipeann::HybridFilterKind::kUnsupported:
    default:
      return 0ULL;
  }
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
  const size_t idx = samples.size() / 2;
  return static_cast<uint64_t>(std::llround(samples[idx]));
}

std::vector<double> default_auto_selectivities() {
  return std::vector<double>(kDefaultAutoSelectivities.begin(), kDefaultAutoSelectivities.end());
}

std::vector<double> parse_selectivity_csv(const std::string &csv) {
  std::vector<double> values;
  std::stringstream stream(csv);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const double value = std::stod(token);
    if (!(value > 0.0 && value <= 1.0)) {
      throw std::invalid_argument("auto calibration selectivity must be in (0, 1]");
    }
    values.push_back(value);
  }
  if (values.empty()) {
    throw std::invalid_argument("auto calibration selectivity list is empty");
  }
  return values;
}

uint64_t candidate_count_for_selectivity(double selectivity, uint64_t npoints) {
  if (npoints == 0) {
    return 0;
  }
  uint64_t candidate_count = static_cast<uint64_t>(std::llround(selectivity * static_cast<double>(npoints)));
  candidate_count = std::max<uint64_t>(1, candidate_count);
  return std::min<uint64_t>(candidate_count, npoints);
}

std::vector<uint32_t> synthetic_candidate_ids(uint64_t npoints, uint64_t candidate_count, uint64_t seed) {
  if (npoints > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("auto calibration supports at most uint32_t point ids");
  }
  std::vector<uint32_t> ids(static_cast<size_t>(npoints));
  std::iota(ids.begin(), ids.end(), 0U);
  if (candidate_count < npoints) {
    std::mt19937_64 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(static_cast<size_t>(candidate_count));
    std::sort(ids.begin(), ids.end());
  }
  return ids;
}

std::vector<uint64_t> bitset_from_candidate_ids(const std::vector<uint32_t> &candidate_ids, uint64_t npoints) {
  std::vector<uint64_t> bitset(static_cast<size_t>((npoints + 63ULL) / 64ULL), 0ULL);
  for (uint32_t point_id : candidate_ids) {
    if (point_id >= npoints) {
      continue;
    }
    const uint64_t word_idx = static_cast<uint64_t>(point_id) >> 6U;
    bitset[static_cast<size_t>(word_idx)] |= 1ULL << (static_cast<uint64_t>(point_id) & 63ULL);
  }
  return bitset;
}

template<typename T>
struct SampledQueries {
  std::vector<T> values;
  uint64_t query_count = 0;
  uint64_t dim = 0;
};

template<typename T>
SampledQueries<T> load_sampled_base_queries(const std::string &base_bin, uint32_t query_count, uint64_t seed) {
  if (query_count == 0) {
    throw std::invalid_argument("auto calibration query count must be positive");
  }

  std::ifstream reader(base_bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("failed to open auto calibration base bin: " + base_bin);
  }

  int32_t npoints_i = 0;
  int32_t dim_i = 0;
  reader.read(reinterpret_cast<char *>(&npoints_i), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&dim_i), sizeof(int32_t));
  if (!reader.good() || npoints_i <= 0 || dim_i <= 0) {
    throw std::runtime_error("invalid auto calibration base bin header: " + base_bin);
  }

  const uint64_t npoints = static_cast<uint64_t>(npoints_i);
  const uint64_t dim = static_cast<uint64_t>(dim_i);
  const uint64_t row_bytes = dim * sizeof(T);
  SampledQueries<T> sampled;
  sampled.query_count = query_count;
  sampled.dim = dim;
  sampled.values.resize(static_cast<size_t>(sampled.query_count * sampled.dim));

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> pick_row(0, npoints - 1);
  for (uint64_t query_idx = 0; query_idx < sampled.query_count; ++query_idx) {
    const uint64_t source_id = pick_row(rng);
    const std::streamoff offset = static_cast<std::streamoff>(sizeof(int32_t) * 2ULL + source_id * row_bytes);
    reader.seekg(offset, std::ios::beg);
    reader.read(reinterpret_cast<char *>(sampled.values.data() + static_cast<size_t>(query_idx * dim)),
                static_cast<std::streamsize>(row_bytes));
    if (!reader.good()) {
      throw std::runtime_error("failed to read sampled calibration row from: " + base_bin);
    }
  }
  return sampled;
}

bool is_unsigned_integer(const char *value) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  for (const unsigned char *cursor = reinterpret_cast<const unsigned char *>(value); *cursor != '\0'; ++cursor) {
    if (!std::isdigit(*cursor)) {
      return false;
    }
  }
  return true;
}

bool append_dataset(HybridCalibrationConfig *config, const std::string &selector_type, const std::string &query_bin,
                    const std::string &query_label_file, const uint32_t sample_limit) {
  const pipeann::HybridFilterKind filter_kind = parse_filter_kind(selector_type);
  if (filter_kind == pipeann::HybridFilterKind::kUnsupported) {
    LOG(ERROR) << "Calibration only supports intersect/subset/range selectors. Got: " << selector_type;
    return false;
  }

  HybridCalibrationInput input;
  input.filter_kind = filter_kind;
  input.selector_type = selector_type;
  input.query_bin = query_bin;
  input.query_label_file = query_label_file;
  input.sample_limit = sample_limit;
  config->datasets.push_back(std::move(input));
  return true;
}

bool parse_calibration_cli(int argc, char **argv, ParsedCalibrationCli *parsed_cli) {
  parsed_cli->index_prefix_path = argv[2];
  parsed_cli->num_threads = static_cast<uint32_t>(std::atoi(argv[3]));
  parsed_cli->config.beamwidth = static_cast<uint32_t>(std::atoi(argv[4]));

  const bool legacy_mode = file_exists(argv[5]);
  int index = 5;
  if (std::string(argv[index]) == "--auto-selectivity") {
    if (argc < 13) {
      LOG(ERROR) << "Auto-selectivity calibration mode expects:"
                 << " --auto-selectivity <base_file.bin> <K> <similarity> <nbr_type> <mem_L> <l_search>"
                 << " <selector_type> [queries_per_bucket] [seed] [selectivity_csv]";
      return false;
    }

    parsed_cli->auto_selectivity = true;
    ++index;
    parsed_cli->auto_base_bin = argv[index++];
    parsed_cli->config.k = std::strtoull(argv[index++], nullptr, 10);
    parsed_cli->dist_metric = argv[index++];
    parsed_cli->nbr_type = argv[index++];
    parsed_cli->config.mem_L = static_cast<uint32_t>(std::strtoul(argv[index++], nullptr, 10));
    parsed_cli->config.l_search = std::strtoull(argv[index++], nullptr, 10);
    parsed_cli->auto_selector_type = argv[index++];
    parsed_cli->auto_filter_kind = parse_filter_kind(parsed_cli->auto_selector_type);
    if (parsed_cli->auto_filter_kind == pipeann::HybridFilterKind::kUnsupported) {
      LOG(ERROR) << "Auto calibration only supports intersect/subset/range selectors. Got: "
                 << parsed_cli->auto_selector_type;
      return false;
    }
    if (index < argc) {
      parsed_cli->auto_queries_per_bucket = static_cast<uint32_t>(std::strtoul(argv[index++], nullptr, 10));
    }
    if (index < argc) {
      parsed_cli->auto_seed = std::strtoull(argv[index++], nullptr, 10);
    }
    try {
      parsed_cli->auto_selectivities = index < argc ? parse_selectivity_csv(argv[index++])
                                                    : default_auto_selectivities();
    } catch (const std::exception &e) {
      LOG(ERROR) << "Invalid auto calibration selectivity list: " << e.what();
      return false;
    }
    if (index != argc) {
      LOG(ERROR) << "Unexpected extra auto calibration arguments.";
      return false;
    }
    return true;
  }

  if (legacy_mode) {
    if (argc != 14) {
      LOG(ERROR) << "Legacy calibration mode expects exactly one dataset.";
      return false;
    }

    const std::string query_bin(argv[index++]);
    parsed_cli->config.k = std::strtoull(argv[index++], nullptr, 10);
    parsed_cli->dist_metric = argv[index++];
    parsed_cli->nbr_type = argv[index++];
    const std::string selector_type(argv[index++]);
    const std::string query_label_file(argv[index++]);
    const uint32_t sample_limit = static_cast<uint32_t>(std::strtoul(argv[index++], nullptr, 10));
    parsed_cli->config.mem_L = static_cast<uint32_t>(std::strtoul(argv[index++], nullptr, 10));
    parsed_cli->config.l_search = std::strtoull(argv[index++], nullptr, 10);
    return append_dataset(&parsed_cli->config, selector_type, query_bin, query_label_file, sample_limit);
  }

  if (!is_unsigned_integer(argv[index])) {
    LOG(ERROR) << "Unable to parse calibration K from argument: " << argv[index];
    return false;
  }

  parsed_cli->config.k = std::strtoull(argv[index++], nullptr, 10);
  parsed_cli->dist_metric = argv[index++];
  parsed_cli->nbr_type = argv[index++];
  parsed_cli->config.mem_L = static_cast<uint32_t>(std::strtoul(argv[index++], nullptr, 10));
  parsed_cli->config.l_search = std::strtoull(argv[index++], nullptr, 10);

  const int remaining = argc - index;
  if (remaining <= 0 || (remaining % 4) != 0) {
    LOG(ERROR) << "Grouped calibration mode expects one or more dataset groups:"
               << " <selector_type> <query_file.bin> <query_label.spmat> <sample_limit>";
    return false;
  }

  for (; index < argc; index += 4) {
    const std::string selector_type(argv[index]);
    const std::string query_bin(argv[index + 1]);
    const std::string query_label_file(argv[index + 2]);
    const uint32_t sample_limit = static_cast<uint32_t>(std::strtoul(argv[index + 3], nullptr, 10));
    if (!append_dataset(&parsed_cli->config, selector_type, query_bin, query_label_file, sample_limit)) {
      return false;
    }
  }
  return true;
}
}  // namespace

template<typename T>
int calibrate_auto_selectivity_threshold(const ParsedCalibrationCli &parsed_cli) {
  if (parsed_cli.config.l_search < parsed_cli.config.k) {
    LOG(ERROR) << "l_search must be >= recall_at";
    return -1;
  }

  if (parsed_cli.auto_queries_per_bucket == 0) {
    LOG(ERROR) << "auto calibration queries_per_bucket must be positive";
    return -1;
  }

  const pipeann::Metric metric = pipeann::get_metric(parsed_cli.dist_metric);

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, parsed_cli.nbr_type);
  if (nbr_handler == nullptr) {
    LOG(ERROR) << "Unknown neighbor type: " << parsed_cli.nbr_type;
    return -1;
  }

  std::unique_ptr<pipeann::SSDIndex<T>> index_ptr(new pipeann::SSDIndex<T>(metric, reader, nbr_handler, true));
  index_ptr->enable_low_memory_search_mode(true);
  if (index_ptr->load(parsed_cli.index_prefix_path.c_str(), parsed_cli.num_threads, false) != 0) {
    return -1;
  }

  if (parsed_cli.config.mem_L != 0) {
    const auto mem_index_path = parsed_cli.index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    index_ptr->load_mem_index(mem_index_path);
  }

  const std::string sidecar_path = pipeann::DenseBitsetIndex::default_sidecar_path(parsed_cli.index_prefix_path);
  std::unique_ptr<pipeann::DenseBitsetIndex> runtime_densebit_index;
  try {
    runtime_densebit_index = pipeann::DenseBitsetIndex::load(sidecar_path, index_ptr->meta_.npoints);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to load runtime densebit sidecar for auto calibration: " << e.what();
    return -1;
  }

  SampledQueries<T> sampled_queries;
  try {
    sampled_queries = load_sampled_base_queries<T>(parsed_cli.auto_base_bin, parsed_cli.auto_queries_per_bucket,
                                                   parsed_cli.auto_seed);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to load auto calibration queries: " << e.what();
    return -1;
  }
  if (sampled_queries.dim != index_ptr->meta_.data_dim) {
    LOG(ERROR) << "Auto calibration base dimension mismatch: expected " << index_ptr->meta_.data_dim
               << ", got " << sampled_queries.dim;
    return -1;
  }

  omp_set_num_threads(1);

  std::map<uint64_t, BucketSamples> bucket_samples;
  uint64_t sampled_query_count = 0;
  uint64_t empty_query_count = 0;
  const uint64_t npoints = index_ptr->meta_.npoints;
  const uint64_t route_selector_mask = selector_mask_for_kind(parsed_cli.auto_filter_kind);

  std::vector<uint32_t> graph_tags(static_cast<size_t>(parsed_cli.config.k), std::numeric_limits<uint32_t>::max());
  std::vector<float> graph_dists(static_cast<size_t>(parsed_cli.config.k), std::numeric_limits<float>::infinity());
  std::vector<uint32_t> prefilter_tags(static_cast<size_t>(parsed_cli.config.k),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<float> prefilter_dists(static_cast<size_t>(parsed_cli.config.k),
                                     std::numeric_limits<float>::infinity());

  for (size_t selectivity_idx = 0; selectivity_idx < parsed_cli.auto_selectivities.size(); ++selectivity_idx) {
    const double selectivity = parsed_cli.auto_selectivities[selectivity_idx];
    const uint64_t candidate_count = candidate_count_for_selectivity(selectivity, npoints);
    if (candidate_count == 0) {
      ++empty_query_count;
      continue;
    }

    std::vector<uint32_t> candidate_ids;
    std::vector<uint64_t> candidate_bitset;
    try {
      candidate_ids = synthetic_candidate_ids(npoints, candidate_count,
                                              parsed_cli.auto_seed + 104729ULL * (selectivity_idx + 1ULL));
      candidate_bitset = bitset_from_candidate_ids(candidate_ids, npoints);
    } catch (const std::exception &e) {
      LOG(ERROR) << "Failed to build synthetic calibration candidates: " << e.what();
      return -1;
    }

    SyntheticBitsetSelector synthetic_selector;
    synthetic_selector.bitset_words = &candidate_bitset;
    synthetic_selector.npoints = npoints;

    for (uint64_t query_idx = 0; query_idx < sampled_queries.query_count; ++query_idx) {
      std::fill(graph_tags.begin(), graph_tags.end(), std::numeric_limits<uint32_t>::max());
      std::fill(graph_dists.begin(), graph_dists.end(), std::numeric_limits<float>::infinity());
      std::fill(prefilter_tags.begin(), prefilter_tags.end(), std::numeric_limits<uint32_t>::max());
      std::fill(prefilter_dists.begin(), prefilter_dists.end(), std::numeric_limits<float>::infinity());

      pipeann::QueryStats graph_stats{};
      pipeann::QueryStats prefilter_stats{};
      const T *query = sampled_queries.values.data() + static_cast<size_t>(query_idx * sampled_queries.dim);
      index_ptr->pipe_search(query, parsed_cli.config.k, parsed_cli.config.mem_L, parsed_cli.config.l_search,
                             graph_tags.data(), graph_dists.data(), parsed_cli.config.beamwidth, &graph_stats,
                             &synthetic_selector, nullptr, 0);
      index_ptr->hybrid_prefilter_search(query, parsed_cli.config.k, prefilter_tags.data(), prefilter_dists.data(),
                                         candidate_ids, &prefilter_stats);

      auto &bucket = bucket_samples[candidate_count];
      bucket.prefilter_us.push_back(prefilter_stats.total_us);
      bucket.graph_us.push_back(graph_stats.total_us);
      ++sampled_query_count;
    }
  }

  std::vector<pipeann::HybridCalibrationBucketV1> buckets;
  buckets.reserve(bucket_samples.size());
  uint64_t tau_m = 0;
  for (const auto &entry : bucket_samples) {
    pipeann::HybridCalibrationBucketV1 bucket;
    bucket.candidate_upper_bound = entry.first;
    bucket.query_count = static_cast<uint64_t>(entry.second.prefilter_us.size());
    bucket.prefilter_p50_us = p50_as_uint64(entry.second.prefilter_us);
    bucket.graph_p50_us = p50_as_uint64(entry.second.graph_us);
    buckets.push_back(bucket);

    if (bucket.query_count >= 8 && bucket.prefilter_p50_us <= bucket.graph_p50_us) {
      tau_m = bucket.candidate_upper_bound;
    }
  }

  if (buckets.empty() || sampled_query_count == 0) {
    LOG(ERROR) << "Auto calibration produced no usable samples";
    return -1;
  }

  const std::string meta_path = pipeann::HybridMetadata::default_metadata_path(parsed_cli.index_prefix_path);
  uint64_t next_threshold_version = 1;
  if (file_exists(meta_path)) {
    try {
      const auto existing_metadata = pipeann::HybridMetadata::load(meta_path, false);
      next_threshold_version = existing_metadata->header().threshold_version + 1;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Existing hybrid metadata is unreadable; restarting threshold_version at 1: " << e.what();
    }
  }

  pipeann::HybridMetadataHeaderV1 header;
  header.flags = kMetadataValidFlag | kCalibrationValidFlag | kAllowPrefilterFlag;
  header.route_selector_mask = route_selector_mask;
  header.tau_m = tau_m;
  header.n_calib = npoints;
  header.n_live_snapshot = npoints;
  header.threshold_version = next_threshold_version;
  header.calib_epoch_sec = static_cast<uint64_t>(std::time(nullptr));
  header.calib_query_count = sampled_query_count;
  header.calib_bucket_count = static_cast<uint64_t>(buckets.size());
  header.calib_k = parsed_cli.config.k;
  header.calib_mem_L = parsed_cli.config.mem_L;
  header.calib_beamwidth = parsed_cli.config.beamwidth;
  header.calib_l_search = parsed_cli.config.l_search;
  header.densebit_npoints = runtime_densebit_index->header().npoints;
  header.densebit_nlabels = runtime_densebit_index->header().nlabels;
  header.densebit_words_per_label = runtime_densebit_index->header().words_per_label;
  header.densebit_nnz = runtime_densebit_index->header().nnz;

  try {
    auto metadata = pipeann::HybridMetadata::create(header, buckets);
    metadata->write_atomically(meta_path);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to write auto-calibrated hybrid metadata: " << e.what();
    return -1;
  }

  try {
    auto metadata = pipeann::HybridMetadata::load(meta_path);
    metadata->validate_against_densebit(runtime_densebit_index->header());
    metadata->validate_against_npoints(npoints);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Auto-calibrated hybrid metadata validation failed after write: " << e.what();
    return -1;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::cout << "mode=auto-selectivity, selectivity_buckets=" << parsed_cli.auto_selectivities.size()
            << ", queries_per_bucket=" << parsed_cli.auto_queries_per_bucket
            << ", sampled_queries=" << sampled_query_count << ", empty_queries=" << empty_query_count
            << ", tau_m=" << tau_m << ", threshold_version=" << next_threshold_version << std::endl;
  std::cout << std::setw(16) << "BucketUpper" << std::setw(12) << "Count" << std::setw(18)
            << "PrefilterP50(us)" << std::setw(18) << "GraphP50(us)" << std::endl;
  std::cout << std::string(64, '=') << std::endl;
  for (const auto &bucket : buckets) {
    std::cout << std::setw(16) << bucket.candidate_upper_bound << std::setw(12) << bucket.query_count
              << std::setw(18) << bucket.prefilter_p50_us << std::setw(18) << bucket.graph_p50_us << std::endl;
  }
  return 0;
}

template<typename T>
int calibrate_threshold(int argc, char **argv) {
  ParsedCalibrationCli parsed_cli;
  if (!parse_calibration_cli(argc, argv, &parsed_cli)) {
    return -1;
  }

  if (parsed_cli.auto_selectivity) {
    return calibrate_auto_selectivity_threshold<T>(parsed_cli);
  }

  if (parsed_cli.config.datasets.empty()) {
    LOG(ERROR) << "No calibration datasets provided";
    return -1;
  }

  if (parsed_cli.config.l_search < parsed_cli.config.k) {
    LOG(ERROR) << "l_search must be >= recall_at";
    return -1;
  }

  const pipeann::Metric metric = pipeann::get_metric(parsed_cli.dist_metric);

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, parsed_cli.nbr_type);
  if (nbr_handler == nullptr) {
    LOG(ERROR) << "Unknown neighbor type: " << parsed_cli.nbr_type;
    return -1;
  }

  std::unique_ptr<pipeann::SSDIndex<T>> index_ptr(new pipeann::SSDIndex<T>(metric, reader, nbr_handler, true));
  index_ptr->enable_low_memory_search_mode(true);
  if (index_ptr->load(parsed_cli.index_prefix_path.c_str(), parsed_cli.num_threads, false) != 0) {
    return -1;
  }

  if (parsed_cli.config.mem_L != 0) {
    const auto mem_index_path = parsed_cli.index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    index_ptr->load_mem_index(mem_index_path);
  }

  const std::string sidecar_path = pipeann::DenseBitsetIndex::default_sidecar_path(parsed_cli.index_prefix_path);
  std::unique_ptr<pipeann::DenseBitsetIndex> densebit_index;
  try {
    densebit_index = pipeann::DenseBitsetIndex::load(sidecar_path, index_ptr->meta_.npoints);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to load densebit sidecar for calibration: " << e.what();
    return -1;
  }

  omp_set_num_threads(1);

  std::map<uint64_t, BucketSamples> bucket_samples;
  uint64_t empty_query_count = 0;

  uint64_t route_selector_mask = 0;
  uint64_t sampled_query_count = 0;
  std::vector<DatasetSummary> dataset_summaries;
  dataset_summaries.reserve(parsed_cli.config.datasets.size());

  for (const auto &dataset : parsed_cli.config.datasets) {
    pipeann::SpmatLabel query_labels(dataset.query_label_file);
    std::unique_ptr<T[]> queries;
    size_t query_num = 0, query_dim = 0;
    pipeann::load_bin<T>(dataset.query_bin, queries, query_num, query_dim);
    if (query_labels.labels_.size() != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and query labels for " << dataset.query_label_file;
      return -1;
    }
    if (query_dim != index_ptr->meta_.data_dim) {
      LOG(ERROR) << "Query dimension mismatch for " << dataset.query_bin << ": expected "
                 << index_ptr->meta_.data_dim << ", got " << query_dim;
      return -1;
    }

    std::unique_ptr<pipeann::AbstractSelector> selector(pipeann::get_selector<T>(dataset.selector_type));
    if (!selector) {
      LOG(ERROR) << "Unknown selector type: " << dataset.selector_type;
      return -1;
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

    DatasetSummary summary;
    summary.selector_type = dataset.selector_type;
    summary.query_bin = dataset.query_bin;
    summary.query_label_file = dataset.query_label_file;
    summary.total_queries = static_cast<uint64_t>(query_num);
    summary.sampled_queries = static_cast<uint64_t>(sampled_query_ids.size());

    std::vector<uint32_t> graph_tags(static_cast<size_t>(parsed_cli.config.k), std::numeric_limits<uint32_t>::max());
    std::vector<float> graph_dists(static_cast<size_t>(parsed_cli.config.k), std::numeric_limits<float>::infinity());
    std::vector<uint32_t> prefilter_tags(static_cast<size_t>(parsed_cli.config.k),
                                         std::numeric_limits<uint32_t>::max());
    std::vector<float> prefilter_dists(static_cast<size_t>(parsed_cli.config.k),
                                       std::numeric_limits<float>::infinity());

    route_selector_mask |= selector_mask_for_kind(dataset.filter_kind);
    sampled_query_count += summary.sampled_queries;

    for (size_t query_idx : sampled_query_ids) {
      pipeann::HybridQueryScratch scratch;
      const auto &labels = query_labels.labels_[query_idx];
      const uint64_t candidate_count = densebit_index->count_candidates(dataset.filter_kind, labels, &scratch);
      if (candidate_count == 0) {
        ++empty_query_count;
        ++summary.empty_queries;
        continue;
      }

      std::vector<uint32_t> candidate_ids;
      densebit_index->materialize_candidates(dataset.filter_kind, labels, &scratch, &candidate_ids);

      std::fill(graph_tags.begin(), graph_tags.end(), std::numeric_limits<uint32_t>::max());
      std::fill(graph_dists.begin(), graph_dists.end(), std::numeric_limits<float>::infinity());
      std::fill(prefilter_tags.begin(), prefilter_tags.end(), std::numeric_limits<uint32_t>::max());
      std::fill(prefilter_dists.begin(), prefilter_dists.end(), std::numeric_limits<float>::infinity());

      pipeann::QueryStats graph_stats{};
      pipeann::QueryStats prefilter_stats{};

      index_ptr->pipe_search(queries.get() + (query_idx * query_dim), parsed_cli.config.k, parsed_cli.config.mem_L,
                             parsed_cli.config.l_search, graph_tags.data(), graph_dists.data(),
                             parsed_cli.config.beamwidth, &graph_stats, selector.get(),
                             filter_buffers[query_idx].data(), 0);
      index_ptr->hybrid_prefilter_search(queries.get() + (query_idx * query_dim), parsed_cli.config.k,
                                         prefilter_tags.data(), prefilter_dists.data(), candidate_ids,
                                         &prefilter_stats);

      const uint64_t bucket_upper_bound = next_power_of_two(candidate_count);
      BucketSamples &samples = bucket_samples[bucket_upper_bound];
      samples.prefilter_us.push_back(prefilter_stats.total_us);
      samples.graph_us.push_back(graph_stats.total_us);
    }

    dataset_summaries.push_back(std::move(summary));
  }

  std::vector<pipeann::HybridCalibrationBucketV1> buckets;
  buckets.reserve(bucket_samples.size());
  uint64_t tau_m = 0;
  for (const auto &entry : bucket_samples) {
    pipeann::HybridCalibrationBucketV1 bucket;
    bucket.candidate_upper_bound = entry.first;
    bucket.query_count = static_cast<uint64_t>(entry.second.prefilter_us.size());
    bucket.prefilter_p50_us = p50_as_uint64(entry.second.prefilter_us);
    bucket.graph_p50_us = p50_as_uint64(entry.second.graph_us);
    buckets.push_back(bucket);

    if (bucket.query_count >= 8 && bucket.prefilter_p50_us <= bucket.graph_p50_us) {
      tau_m = bucket.candidate_upper_bound;
    }
  }

  const std::string meta_path = pipeann::HybridMetadata::default_metadata_path(parsed_cli.index_prefix_path);
  uint64_t next_threshold_version = 1;
  if (file_exists(meta_path)) {
    try {
      const auto existing_metadata = pipeann::HybridMetadata::load(meta_path);
      next_threshold_version = existing_metadata->header().threshold_version + 1;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Existing hybrid metadata is unreadable; restarting threshold_version at 1: " << e.what();
    }
  }

  pipeann::HybridMetadataHeaderV1 header;
  header.flags = kMetadataValidFlag | kCalibrationValidFlag | kAllowPrefilterFlag;
  header.route_selector_mask = route_selector_mask;
  header.tau_m = tau_m;
  header.n_calib = index_ptr->meta_.npoints;
  header.n_live_snapshot = index_ptr->meta_.npoints;
  header.threshold_version = next_threshold_version;
  header.calib_epoch_sec = static_cast<uint64_t>(std::time(nullptr));
  header.calib_query_count = sampled_query_count;
  header.calib_bucket_count = static_cast<uint64_t>(buckets.size());
  header.calib_k = parsed_cli.config.k;
  header.calib_mem_L = parsed_cli.config.mem_L;
  header.calib_beamwidth = parsed_cli.config.beamwidth;
  header.calib_l_search = parsed_cli.config.l_search;
  header.densebit_npoints = densebit_index->header().npoints;
  header.densebit_nlabels = densebit_index->header().nlabels;
  header.densebit_words_per_label = densebit_index->header().words_per_label;
  header.densebit_nnz = densebit_index->header().nnz;

  try {
    auto metadata = pipeann::HybridMetadata::create(header, buckets);
    metadata->write_atomically(meta_path);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to write hybrid metadata: " << e.what();
    return -1;
  }

  try {
    auto metadata = pipeann::HybridMetadata::load(meta_path);
    metadata->validate_against_densebit(densebit_index->header());
    metadata->validate_against_npoints(index_ptr->meta_.npoints);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Hybrid metadata smoke validation failed after write: " << e.what();
    return -1;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::cout << "datasets=" << parsed_cli.config.datasets.size() << ", sampled_queries=" << sampled_query_count
            << ", empty_queries=" << empty_query_count << ", tau_m=" << tau_m
            << ", threshold_version=" << next_threshold_version << std::endl;
  for (const auto &summary : dataset_summaries) {
    std::cout << "  - selector=" << summary.selector_type << ", query_bin=" << summary.query_bin
              << ", query_labels=" << summary.query_label_file << ", total_queries=" << summary.total_queries
              << ", sampled_queries=" << summary.sampled_queries << ", empty_queries=" << summary.empty_queries
              << std::endl;
  }
  std::cout << std::setw(16) << "BucketUpper" << std::setw(12) << "Count" << std::setw(18)
            << "PrefilterP50(us)" << std::setw(18) << "GraphP50(us)" << std::endl;
  std::cout << std::string(64, '=') << std::endl;
  for (const auto &bucket : buckets) {
    std::cout << std::setw(16) << bucket.candidate_upper_bound << std::setw(12) << bucket.query_count
              << std::setw(18) << bucket.prefilter_p50_us << std::setw(18) << bucket.graph_p50_us << std::endl;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 13) {
    std::cout << "Usage (legacy single dataset): " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <query_file.bin>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <selector_type (intersect/subset)>"
              << " <query_label.spmat>"
              << " <sample_limit (0 means all)>"
              << " <mem_L (0 means no mem index)>"
              << " <l_search>" << std::endl;
    std::cout << "Usage (grouped multi-dataset): " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <mem_L (0 means no mem index)>"
              << " <l_search>"
              << " <selector_type_1> <query_file_1.bin> <query_label_1.spmat> <sample_limit_1>"
              << " [<selector_type_2> <query_file_2.bin> <query_label_2.spmat> <sample_limit_2> ...]"
              << std::endl;
    std::cout << "Usage (auto-selectivity post-build): " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " --auto-selectivity"
              << " <base_file.bin>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <mem_L (0 means no mem index)>"
              << " <l_search>"
              << " <selector_type (intersect/subset/range)>"
              << " [queries_per_bucket=100] [seed=20260423] [selectivity_csv]" << std::endl;
    return -1;
  }

  const std::string index_type = argv[1];
  if (index_type == "float") {
    return calibrate_threshold<float>(argc, argv);
  }
  if (index_type == "int8") {
    return calibrate_threshold<int8_t>(argc, argv);
  }
  if (index_type == "uint8") {
    return calibrate_threshold<uint8_t>(argc, argv);
  }

  std::cout << "Unsupported index type: " << index_type << ". Use float/int8/uint8" << std::endl;
  return -1;
}
