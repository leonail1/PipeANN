#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include "distance.h"
#include "dynamic_index.h"
#include "filter/label.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include "utils.h"

namespace {

struct Config {
  std::string mode;
  std::string source_prefix;
  std::string dest_prefix;
  std::string data_bin;
  std::string query_bin;
  std::string query_vector_csv;
  std::string truthset_bin;
  std::string base_label_file;
  std::string query_label_file;
  std::string query_label_csv;
  std::string delete_id_file;
  std::string insert_tag_file;
  std::string flat_pq_pivots;
  std::string selector_type = "none";
  std::string route = "auto";
  std::string jsonl_output;
  std::string metric = "l2";
  std::string raw_command;
  uint32_t cpu_cap = 0;
  uint64_t insert_start = 0;
  uint64_t insert_count = 0;
  uint64_t delete_start = 0;
  uint64_t delete_count = 0;
  uint32_t insert_threads = 1;
  uint32_t search_threads = 1;
  uint32_t merge_threads = 1;
  uint32_t build_l = 96;
  uint32_t build_r = 64;
  uint32_t beamwidth = 4;
  uint32_t k = 10;
  uint32_t search_l = 60;
  uint32_t mem_l = 0;
  uint32_t pq_bytes = 16;
  uint64_t flat_threshold = 10000;
  uint64_t query_limit = 0;
  bool final_merge = true;
  bool single_query_static_rss = false;
};

struct LoadedQueries {
  std::unique_ptr<float[]> data;
  size_t count = 0;
  size_t dim = 0;
};

struct Truthset {
  std::vector<uint32_t> tags;
  size_t count = 0;
  size_t dim = 0;
};

struct SearchMetrics {
  double elapsed_s = 0.0;
  double qps = 0.0;
  double avg_latency_us = 0.0;
  double p50_latency_us = 0.0;
  double p95_latency_us = 0.0;
  double p99_latency_us = 0.0;
  double recall = 0.0;
  uint64_t process_max_rss_kb = 0;
  uint64_t rss_before_query_kb = 0;
  uint64_t rss_after_query_kb = 0;
  uint64_t query_peak_rss_kb = 0;
  uint64_t query_peak_delta_kb = 0;
  double mean_candidate_count = 0.0;
  double mean_route_overhead_us = 0.0;
  uint64_t prefilter_count = 0;
  uint64_t graph_count = 0;
  uint64_t fallback_count = 0;
  uint64_t empty_count = 0;
  uint64_t min_threshold = 0;
  uint64_t max_threshold = 0;
};

std::string require_value(int argc, char **argv, int *index, const std::string &flag) {
  if (*index + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  return argv[++(*index)];
}

Config parse_args(int argc, char **argv) {
  Config config;
  std::unordered_set<std::string> seen_scalar_args;
  std::ostringstream raw_command;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) raw_command << " ";
    raw_command << argv[i];
  }
  config.raw_command = raw_command.str();
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    auto mark_scalar = [&](const std::string &name) {
      if (!seen_scalar_args.insert(name).second) {
        throw std::runtime_error("duplicate scalar argument: " + name);
      }
    };
    if (arg == "--mode") {
      mark_scalar(arg);
      config.mode = require_value(argc, argv, &index, arg);
    } else if (arg == "--source-prefix") {
      mark_scalar(arg);
      config.source_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--dest-prefix") {
      mark_scalar(arg);
      config.dest_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--data-bin") {
      mark_scalar(arg);
      config.data_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-bin") {
      mark_scalar(arg);
      config.query_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-vector-csv") {
      mark_scalar(arg);
      config.query_vector_csv = require_value(argc, argv, &index, arg);
    } else if (arg == "--truthset-bin") {
      mark_scalar(arg);
      config.truthset_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--base-label-file") {
      mark_scalar(arg);
      config.base_label_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-label-file") {
      mark_scalar(arg);
      config.query_label_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-label-csv") {
      mark_scalar(arg);
      config.query_label_csv = require_value(argc, argv, &index, arg);
    } else if (arg == "--delete-id-file") {
      mark_scalar(arg);
      config.delete_id_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--insert-tag-file") {
      mark_scalar(arg);
      config.insert_tag_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--flat-pq-pivots") {
      mark_scalar(arg);
      config.flat_pq_pivots = require_value(argc, argv, &index, arg);
    } else if (arg == "--selector-type") {
      mark_scalar(arg);
      config.selector_type = require_value(argc, argv, &index, arg);
    } else if (arg == "--route") {
      mark_scalar(arg);
      config.route = require_value(argc, argv, &index, arg);
    } else if (arg == "--jsonl-output") {
      mark_scalar(arg);
      config.jsonl_output = require_value(argc, argv, &index, arg);
    } else if (arg == "--metric") {
      mark_scalar(arg);
      config.metric = require_value(argc, argv, &index, arg);
    } else if (arg == "--cpu-cap") {
      mark_scalar(arg);
      config.cpu_cap = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--insert-start") {
      mark_scalar(arg);
      config.insert_start = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-count") {
      mark_scalar(arg);
      config.insert_count = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--delete-start") {
      mark_scalar(arg);
      config.delete_start = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--delete-count") {
      mark_scalar(arg);
      config.delete_count = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-threads") {
      mark_scalar(arg);
      config.insert_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--search-threads") {
      mark_scalar(arg);
      config.search_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--merge-threads") {
      mark_scalar(arg);
      config.merge_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--build-l") {
      mark_scalar(arg);
      config.build_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--build-r") {
      mark_scalar(arg);
      config.build_r = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--beamwidth") {
      mark_scalar(arg);
      config.beamwidth = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--k") {
      mark_scalar(arg);
      config.k = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--search-l") {
      mark_scalar(arg);
      config.search_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--mem-l") {
      mark_scalar(arg);
      config.mem_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--pq-bytes") {
      mark_scalar(arg);
      config.pq_bytes = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--flat-threshold") {
      mark_scalar(arg);
      config.flat_threshold = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--query-limit") {
      mark_scalar(arg);
      config.query_limit = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--skip-final-merge") {
      mark_scalar(arg);
      config.final_merge = false;
    } else if (arg == "--single-query-static-rss") {
      mark_scalar(arg);
      config.single_query_static_rss = true;
    } else if (arg == "--help") {
      throw std::runtime_error(
          "usage: dynamic_update_suite_driver --mode insert-only|zero-insert-only|search-during-insert|delete-batch|reinsert-batch|measure-dynamic-search|measure-delete-only|measure-delete-then-merge ... [--flat-pq-pivots <path> for zero-insert-only]");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (config.mode.empty() || config.source_prefix.empty()) {
    throw std::runtime_error("--mode and --source-prefix are required");
  }
  if ((config.mode == "insert-only" || config.mode == "reinsert-batch" || config.mode == "search-during-insert"
       || config.mode == "zero-insert-only")
      && config.data_bin.empty()) {
    throw std::runtime_error("--data-bin is required for insert modes");
  }
  if (!config.insert_tag_file.empty() &&
      !(config.mode == "insert-only" || config.mode == "reinsert-batch" || config.mode == "search-during-insert"
        || config.mode == "zero-insert-only")) {
      throw std::runtime_error("--insert-tag-file is only valid for insert modes");
    }
  if (!config.flat_pq_pivots.empty() && config.mode != "zero-insert-only") {
    throw std::runtime_error("--flat-pq-pivots is only valid for zero-insert-only");
  }
  if (!config.flat_pq_pivots.empty() && !file_exists(config.flat_pq_pivots)) {
    throw std::runtime_error("--flat-pq-pivots file does not exist: " + config.flat_pq_pivots);
  }
  if (config.mode == "zero-insert-only" && config.insert_count <= config.flat_threshold) {
    throw std::runtime_error("--insert-count must exceed --flat-threshold so PQ materialization occurs");
  }
  if (config.mode == "measure-delete-then-merge" && config.dest_prefix.empty()) {
    throw std::runtime_error("--dest-prefix is required for merge-producing modes");
  }
  if ((config.mode == "measure-dynamic-search" || config.mode == "search-during-insert")
      && config.query_bin.empty() && config.query_vector_csv.empty()) {
    throw std::runtime_error("--query-bin or --query-vector-csv is required for search modes");
  }
  return config;
}

std::string json_escape(const std::string &input) {
  std::string out;
  for (char ch : input) {
    if (ch == '\\') out += "\\\\";
    else if (ch == '"') out += "\\\"";
    else if (ch == '\n') out += "\\n";
    else out += ch;
  }
  return out;
}

void append_jsonl(const Config &config, const std::string &payload) {
  if (config.jsonl_output.empty()) {
    std::cout << payload << std::endl;
    return;
  }
  std::ofstream writer(config.jsonl_output, std::ios::app);
  if (!writer.is_open()) {
    throw std::runtime_error("failed to open JSONL output: " + config.jsonl_output);
  }
  writer << payload << "\n";
}

uint64_t max_rss_kb() {
  rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<uint64_t>(usage.ru_maxrss);
}

uint64_t current_rss_kb() {
  std::ifstream reader("/proc/self/statm");
  uint64_t total_pages = 0;
  uint64_t resident_pages = 0;
  reader >> total_pages >> resident_pages;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (!reader.good() || page_size <= 0) {
    return 0;
  }
  return resident_pages * static_cast<uint64_t>(page_size) / 1024ULL;
}

std::string cpus_allowed_list() {
  std::ifstream reader("/proc/self/status");
  std::string line;
  while (std::getline(reader, line)) {
    const std::string key = "Cpus_allowed_list:";
    if (line.rfind(key, 0) == 0) {
      const auto pos = line.find_first_not_of(" \t", key.size());
      return pos == std::string::npos ? "" : line.substr(pos);
    }
  }
  return "";
}

bool cpu_cap_enforced(uint32_t cpu_cap) {
  if (cpu_cap == 0) {
    return true;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    return false;
  }
  return static_cast<uint32_t>(CPU_COUNT(&mask)) <= cpu_cap;
}

std::vector<uint32_t> load_delete_tags(const Config &config) {
  std::vector<uint32_t> tags;
  if (!config.delete_id_file.empty()) {
    std::ifstream reader(config.delete_id_file);
    if (!reader.is_open()) {
      throw std::runtime_error("failed to open delete id file: " + config.delete_id_file);
    }
    uint64_t value = 0;
    while (reader >> value) {
      if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("delete tag exceeds uint32 range");
      }
      tags.push_back(static_cast<uint32_t>(value));
    }
    if (config.delete_count != 0 && tags.size() != config.delete_count) {
      throw std::runtime_error("delete id file count does not match --delete-count");
    }
    return tags;
  }
  if (config.delete_start + config.delete_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL) {
    throw std::runtime_error("delete range exceeds uint32 tag range");
  }
  tags.reserve(static_cast<size_t>(config.delete_count));
  for (uint64_t i = 0; i < config.delete_count; ++i) {
    tags.push_back(static_cast<uint32_t>(config.delete_start + i));
  }
  return tags;
}

std::vector<uint32_t> load_insert_tags(const Config &config) {
  if (!config.insert_tag_file.empty()) {
    std::ifstream reader(config.insert_tag_file);
    if (!reader.is_open()) {
      throw std::runtime_error("failed to open insert tag file: " + config.insert_tag_file);
    }
    std::vector<uint32_t> tags;
    uint64_t value = 0;
    while (reader >> value) {
      if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("insert tag exceeds uint32 range");
      }
      tags.push_back(static_cast<uint32_t>(value));
    }
    if (tags.size() != config.insert_count) {
      throw std::runtime_error("insert tag file count does not match --insert-count");
    }
    return tags;
  }
  if (config.insert_start + config.insert_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL) {
    throw std::runtime_error("insert range exceeds uint32 tag range");
  }
  std::vector<uint32_t> tags;
  tags.reserve(static_cast<size_t>(config.insert_count));
  for (uint64_t i = 0; i < config.insert_count; ++i) {
    tags.push_back(static_cast<uint32_t>(config.insert_start + i));
  }
  return tags;
}

void validate_unique_delete_tags(const std::vector<uint32_t> &tags) {
  std::unordered_set<uint32_t> seen;
  seen.reserve(tags.size());
  for (uint32_t tag : tags) {
    if (!seen.insert(tag).second) {
      throw std::runtime_error("duplicate delete tag: " + std::to_string(tag));
    }
  }
}

std::string fnv1a_tags_hex(const std::vector<uint32_t> &tags) {
  uint64_t hash = 1469598103934665603ULL;
  for (uint32_t tag : tags) {
    for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
      hash ^= static_cast<uint8_t>((tag >> (byte * 8)) & 0xffU);
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

uint32_t load_bin_dim(const std::string &data_bin) {
  std::ifstream reader(data_bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("failed to open data bin: " + data_bin);
  }
  int32_t npts_i32 = 0;
  int32_t dim_i32 = 0;
  reader.read(reinterpret_cast<char *>(&npts_i32), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
  if (!reader.good() || npts_i32 < 0 || dim_i32 <= 0) {
    throw std::runtime_error("failed to read data bin header: " + data_bin);
  }
  return static_cast<uint32_t>(dim_i32);
}

uint64_t disk_index_label_size(const std::string &prefix) {
  pipeann::SSDIndexMetadata<float> meta;
  meta.load_from_disk_index(prefix + "_disk.index");
  return meta.label_size;
}

bool densebit_sidecar_loadable(const std::string &prefix, uint64_t npoints) {
  const std::string sidecar_path = pipeann::DenseBitsetIndex::default_sidecar_path(prefix);
  if (!file_exists(sidecar_path)) {
    return false;
  }
  try {
    auto densebit = pipeann::DenseBitsetIndex::load(sidecar_path, npoints);
    return densebit != nullptr;
  } catch (const std::exception &) {
    return false;
  }
}

std::vector<uint32_t> parse_label_csv(const std::string &csv) {
  std::vector<uint32_t> labels;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    const std::string token = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!token.empty()) {
      labels.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return labels;
}

std::unique_ptr<pipeann::Distance<float>> make_distance(const std::string &metric_name) {
  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(pipeann::get_metric(metric_name));
  if (distance == nullptr) {
    throw std::runtime_error("failed to create distance function");
  }
  return std::unique_ptr<pipeann::Distance<float>>(distance);
}

pipeann::HybridFilterKind parse_selector(const std::string &selector_type) {
  if (selector_type == "none" || selector_type.empty()) return pipeann::HybridFilterKind::kUnsupported;
  if (selector_type == "intersect") return pipeann::HybridFilterKind::kIntersect;
  if (selector_type == "subset") return pipeann::HybridFilterKind::kSubset;
  if (selector_type == "range") return pipeann::HybridFilterKind::kRange;
  throw std::runtime_error("unsupported selector type: " + selector_type);
}

pipeann::HybridRouteOverride parse_route(const std::string &route) {
  if (route == "auto" || route.empty()) return pipeann::HybridRouteOverride::kAuto;
  if (route == "prefilter") return pipeann::HybridRouteOverride::kForcePrefilter;
  if (route == "graph") return pipeann::HybridRouteOverride::kForceGraphOnly;
  throw std::runtime_error("unsupported route: " + route);
}

pipeann::DynamicSSDIndex<float, uint32_t> open_dynamic_index(const Config &config,
                                                            pipeann::Distance<float> *distance) {
  pipeann::IndexBuildParameters parameters;
  const uint32_t total_threads = std::max<uint32_t>(1, config.insert_threads + config.search_threads);
  parameters.set(0, config.build_l, config.build_r, 1.2, total_threads, true, config.beamwidth);
  return pipeann::DynamicSSDIndex<float, uint32_t>(parameters, config.source_prefix, config.dest_prefix.empty()
                                                                          ? config.source_prefix + "_suite_merge"
                                                                          : config.dest_prefix,
                                                  distance, pipeann::get_metric(config.metric), BEAM_SEARCH,
                                                  config.mem_l != 0);
}

std::vector<float> load_vector_slice(const std::string &data_bin, uint64_t start, uint64_t count, size_t *dim_out) {
  std::ifstream reader(data_bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("failed to open data bin: " + data_bin);
  }
  int32_t total_i32 = 0;
  int32_t dim_i32 = 0;
  reader.read(reinterpret_cast<char *>(&total_i32), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
  if (start + count > static_cast<uint64_t>(total_i32)) {
    throw std::runtime_error("requested vector slice exceeds data bin size");
  }
  const uint64_t offset = 2ULL * sizeof(int32_t) + start * static_cast<uint64_t>(dim_i32) * sizeof(float);
  reader.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  std::vector<float> vectors(static_cast<size_t>(count) * static_cast<size_t>(dim_i32));
  reader.read(reinterpret_cast<char *>(vectors.data()), static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!reader.good()) {
    throw std::runtime_error("failed to read vector slice");
  }
  *dim_out = static_cast<size_t>(dim_i32);
  return vectors;
}

LoadedQueries load_queries(const Config &config) {
  LoadedQueries queries;
  if (!config.query_vector_csv.empty()) {
    std::vector<float> values;
    size_t start = 0;
    while (start < config.query_vector_csv.size()) {
      size_t comma = config.query_vector_csv.find(',', start);
      const std::string token = config.query_vector_csv.substr(
          start, comma == std::string::npos ? std::string::npos : comma - start);
      if (!token.empty()) {
        values.push_back(std::stof(token));
      }
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    if (values.empty()) {
      throw std::runtime_error("--query-vector-csv is empty");
    }
    queries.count = 1;
    queries.dim = values.size();
    queries.data.reset(new float[values.size()]);
    std::copy(values.begin(), values.end(), queries.data.get());
    return queries;
  }
  pipeann::load_bin<float>(config.query_bin, queries.data, queries.count, queries.dim);
  if (config.query_limit > 0 && config.query_limit < queries.count) {
    queries.count = static_cast<size_t>(config.query_limit);
  }
  return queries;
}

Truthset load_truthset(const Config &config, size_t query_count) {
  Truthset truth;
  if (config.truthset_bin.empty() || !file_exists(config.truthset_bin)) {
    return truth;
  }
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *gt_tags = nullptr;
  size_t gt_num = 0;
  size_t gt_dim = 0;
  pipeann::load_truthset(config.truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
  if (gt_num < query_count) {
    throw std::runtime_error("truthset has fewer rows than queries");
  }
  truth.count = query_count;
  truth.dim = gt_dim;
  truth.tags.resize(query_count * gt_dim);
  const uint32_t *source = gt_tags == nullptr ? reinterpret_cast<uint32_t *>(gt_ids) : gt_tags;
  std::copy(source, source + truth.tags.size(), truth.tags.begin());
  delete[] gt_ids;
  delete[] gt_dists;
  if (gt_tags != nullptr) delete[] gt_tags;
  return truth;
}

double recall_at_k(const std::vector<uint32_t> &results, const Truthset &truth, size_t query_count, uint32_t k) {
  if (truth.tags.empty() || truth.dim == 0) {
    return 0.0;
  }
  uint64_t matches = 0;
  for (size_t q = 0; q < query_count; ++q) {
    const uint32_t *truth_row = truth.tags.data() + q * truth.dim;
    for (uint32_t r = 0; r < k; ++r) {
      const uint32_t tag = results[q * k + r];
      for (uint32_t t = 0; t < std::min<uint32_t>(k, static_cast<uint32_t>(truth.dim)); ++t) {
        if (tag == truth_row[t]) {
          ++matches;
          break;
        }
      }
    }
  }
  return 100.0 * static_cast<double>(matches) / static_cast<double>(query_count * k);
}

double percentile(std::vector<double> sorted_values, double p) {
  if (sorted_values.empty()) return 0.0;
  std::sort(sorted_values.begin(), sorted_values.end());
  const size_t index = static_cast<size_t>(p * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[index];
}

std::vector<std::vector<char>> load_filter_buffers(const Config &config, size_t query_count) {
  if (!config.query_label_csv.empty()) {
    if (query_count != 1) {
      throw std::runtime_error("--query-label-csv supports exactly one query");
    }
    std::vector<uint32_t> labels = parse_label_csv(config.query_label_csv);
    std::vector<std::vector<char>> buffers(1);
    buffers[0].resize(sizeof(uint32_t) + labels.size() * sizeof(uint32_t), 0);
    const uint32_t label_count = static_cast<uint32_t>(labels.size());
    memcpy(buffers[0].data(), &label_count, sizeof(uint32_t));
    if (!labels.empty()) {
      memcpy(buffers[0].data() + sizeof(uint32_t), labels.data(), labels.size() * sizeof(uint32_t));
    }
    return buffers;
  }
  if (config.query_label_file.empty()) {
    return {};
  }
  pipeann::SpmatLabel labels(config.query_label_file);
  if (labels.labels_.size() < query_count) {
    throw std::runtime_error("query label file has fewer rows than query bin");
  }
  std::vector<std::vector<char>> buffers(query_count);
  const size_t label_size = labels.label_size();
  for (size_t q = 0; q < query_count; ++q) {
    buffers[q].resize(label_size, 0);
    labels.write(static_cast<uint32_t>(q), buffers[q].data());
  }
  return buffers;
}

SearchMetrics run_search(const Config &config, pipeann::DynamicSSDIndex<float, uint32_t> &index) {
  LoadedQueries queries = load_queries(config);
  Truthset truth = load_truthset(config, queries.count);
  std::vector<std::vector<char>> filters = load_filter_buffers(config, queries.count);
  const pipeann::HybridFilterKind filter_kind = parse_selector(config.selector_type);
  const pipeann::HybridRouteOverride route_override = parse_route(config.route);
  const bool use_filter = filter_kind != pipeann::HybridFilterKind::kUnsupported && !filters.empty();

  std::vector<uint32_t> tags(queries.count * config.k, std::numeric_limits<uint32_t>::max());
  std::vector<float> distances(queries.count * config.k, std::numeric_limits<float>::infinity());
  std::vector<double> latencies(queries.count, 0.0);
  std::vector<pipeann::HybridQueryStats> hybrid_stats(use_filter ? queries.count : 0);

  std::atomic<bool> sample_rss {true};
  std::atomic<uint64_t> query_peak_rss {0};
  const uint64_t rss_before_query = current_rss_kb();
  query_peak_rss.store(rss_before_query, std::memory_order_relaxed);
  std::thread rss_sampler([&]() {
    while (sample_rss.load(std::memory_order_relaxed)) {
      const uint64_t current = current_rss_kb();
      uint64_t previous = query_peak_rss.load(std::memory_order_relaxed);
      while (current > previous &&
             !query_peak_rss.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.search_threads) schedule(dynamic, 1)
  for (int64_t q = 0; q < static_cast<int64_t>(queries.count); ++q) {
    pipeann::QueryStats stats {};
    const auto query_start = std::chrono::steady_clock::now();
    index.search(queries.data.get() + static_cast<size_t>(q) * queries.dim, config.k, config.mem_l, config.search_l,
                 config.beamwidth, tags.data() + static_cast<size_t>(q) * config.k,
                 distances.data() + static_cast<size_t>(q) * config.k, &stats, true, filter_kind,
                 use_filter ? filters[static_cast<size_t>(q)].data() : nullptr,
                 use_filter ? &hybrid_stats[static_cast<size_t>(q)] : nullptr, route_override);
    const auto query_end = std::chrono::steady_clock::now();
    latencies[static_cast<size_t>(q)] =
        std::chrono::duration<double, std::micro>(query_end - query_start).count();
  }
  const auto end = std::chrono::steady_clock::now();
  sample_rss.store(false, std::memory_order_relaxed);
  rss_sampler.join();
  const uint64_t rss_after_query = current_rss_kb();
  uint64_t query_peak = query_peak_rss.load(std::memory_order_relaxed);
  query_peak = std::max(query_peak, rss_after_query);
  std::chrono::duration<double> elapsed = end - start;

  SearchMetrics metrics;
  metrics.elapsed_s = elapsed.count();
  metrics.qps = static_cast<double>(queries.count) / std::max(metrics.elapsed_s, 1e-9);
  metrics.avg_latency_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  metrics.p50_latency_us = percentile(latencies, 0.50);
  metrics.p95_latency_us = percentile(latencies, 0.95);
  metrics.p99_latency_us = percentile(latencies, 0.99);
  metrics.recall = recall_at_k(tags, truth, queries.count, config.k);
  metrics.process_max_rss_kb = std::max(max_rss_kb(), query_peak);
  metrics.rss_before_query_kb = rss_before_query;
  metrics.rss_after_query_kb = rss_after_query;
  metrics.query_peak_rss_kb = query_peak;
  metrics.query_peak_delta_kb = query_peak > rss_before_query ? query_peak - rss_before_query : 0;
  if (use_filter) {
    uint64_t min_threshold = std::numeric_limits<uint64_t>::max();
    uint64_t max_threshold = 0;
    double candidate_sum = 0.0;
    double route_overhead_sum = 0.0;
    for (const auto &hybrid : hybrid_stats) {
      candidate_sum += static_cast<double>(hybrid.candidate_count);
      route_overhead_sum += static_cast<double>(hybrid.route_overhead_us);
      min_threshold = std::min(min_threshold, hybrid.threshold);
      max_threshold = std::max(max_threshold, hybrid.threshold);
      switch (hybrid.decision) {
        case pipeann::HybridRouteDecision::kPrefilter:
          ++metrics.prefilter_count;
          break;
        case pipeann::HybridRouteDecision::kGraphOnly:
          ++metrics.graph_count;
          break;
        case pipeann::HybridRouteDecision::kPrefilterFastReturn:
          ++metrics.empty_count;
          break;
        case pipeann::HybridRouteDecision::kAutoGraphFallback:
        default:
          ++metrics.fallback_count;
          break;
      }
    }
    metrics.mean_candidate_count = candidate_sum / static_cast<double>(queries.count);
    metrics.mean_route_overhead_us = route_overhead_sum / static_cast<double>(queries.count);
    metrics.min_threshold = min_threshold == std::numeric_limits<uint64_t>::max() ? 0 : min_threshold;
    metrics.max_threshold = max_threshold;
  }
  return metrics;
}

std::string common_fields(const Config &config, const std::string &mode, uint64_t live_count);

SearchMetrics run_static_search(const Config &config, pipeann::SSDIndex<float, uint32_t> &index) {
  LoadedQueries queries = load_queries(config);
  Truthset truth = load_truthset(config, queries.count);
  std::vector<std::vector<char>> filters = load_filter_buffers(config, queries.count);
  const pipeann::HybridFilterKind filter_kind = parse_selector(config.selector_type);
  const pipeann::HybridRouteOverride route_override = parse_route(config.route);
  const bool use_filter = filter_kind != pipeann::HybridFilterKind::kUnsupported && !filters.empty();

  std::vector<uint32_t> tags(queries.count * config.k, std::numeric_limits<uint32_t>::max());
  std::vector<float> distances(queries.count * config.k, std::numeric_limits<float>::infinity());
  std::vector<double> latencies(queries.count, 0.0);
  std::vector<pipeann::HybridQueryStats> hybrid_stats(use_filter ? queries.count : 0);

  std::atomic<bool> sample_rss {true};
  std::atomic<uint64_t> query_peak_rss {0};
  const uint64_t rss_before_query = current_rss_kb();
  query_peak_rss.store(rss_before_query, std::memory_order_relaxed);
  std::thread rss_sampler([&]() {
    while (sample_rss.load(std::memory_order_relaxed)) {
      const uint64_t current = current_rss_kb();
      uint64_t previous = query_peak_rss.load(std::memory_order_relaxed);
      while (current > previous &&
             !query_peak_rss.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.search_threads) schedule(dynamic, 1)
  for (int64_t q = 0; q < static_cast<int64_t>(queries.count); ++q) {
    pipeann::QueryStats stats {};
    const auto query_start = std::chrono::steady_clock::now();
    if (use_filter) {
      index.hybrid_search(queries.data.get() + static_cast<size_t>(q) * queries.dim, config.k, config.mem_l,
                          config.search_l, tags.data() + static_cast<size_t>(q) * config.k,
                          distances.data() + static_cast<size_t>(q) * config.k, config.beamwidth, filter_kind,
                          filters[static_cast<size_t>(q)].data(), &stats,
                          &hybrid_stats[static_cast<size_t>(q)], route_override);
    } else {
      index.pipe_search(queries.data.get() + static_cast<size_t>(q) * queries.dim, config.k, config.mem_l,
                        config.search_l, tags.data() + static_cast<size_t>(q) * config.k,
                        distances.data() + static_cast<size_t>(q) * config.k, config.beamwidth, &stats);
    }
    const auto query_end = std::chrono::steady_clock::now();
    latencies[static_cast<size_t>(q)] =
        std::chrono::duration<double, std::micro>(query_end - query_start).count();
  }
  const auto end = std::chrono::steady_clock::now();
  sample_rss.store(false, std::memory_order_relaxed);
  rss_sampler.join();
  const uint64_t rss_after_query = current_rss_kb();
  uint64_t query_peak = query_peak_rss.load(std::memory_order_relaxed);
  query_peak = std::max(query_peak, rss_after_query);
  std::chrono::duration<double> elapsed = end - start;

  SearchMetrics metrics;
  metrics.elapsed_s = elapsed.count();
  metrics.qps = static_cast<double>(queries.count) / std::max(metrics.elapsed_s, 1e-9);
  metrics.avg_latency_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  metrics.p50_latency_us = percentile(latencies, 0.50);
  metrics.p95_latency_us = percentile(latencies, 0.95);
  metrics.p99_latency_us = percentile(latencies, 0.99);
  metrics.recall = recall_at_k(tags, truth, queries.count, config.k);
  metrics.process_max_rss_kb = std::max(max_rss_kb(), query_peak);
  metrics.rss_before_query_kb = rss_before_query;
  metrics.rss_after_query_kb = rss_after_query;
  metrics.query_peak_rss_kb = query_peak;
  metrics.query_peak_delta_kb = query_peak > rss_before_query ? query_peak - rss_before_query : 0;
  if (use_filter) {
    uint64_t min_threshold = std::numeric_limits<uint64_t>::max();
    uint64_t max_threshold = 0;
    double candidate_sum = 0.0;
    double route_overhead_sum = 0.0;
    for (const auto &hybrid : hybrid_stats) {
      candidate_sum += static_cast<double>(hybrid.candidate_count);
      route_overhead_sum += static_cast<double>(hybrid.route_overhead_us);
      min_threshold = std::min(min_threshold, hybrid.threshold);
      max_threshold = std::max(max_threshold, hybrid.threshold);
      switch (hybrid.decision) {
        case pipeann::HybridRouteDecision::kPrefilter:
          ++metrics.prefilter_count;
          break;
        case pipeann::HybridRouteDecision::kGraphOnly:
          ++metrics.graph_count;
          break;
        case pipeann::HybridRouteDecision::kPrefilterFastReturn:
          ++metrics.empty_count;
          break;
        case pipeann::HybridRouteDecision::kAutoGraphFallback:
        default:
          ++metrics.fallback_count;
          break;
      }
    }
    metrics.mean_candidate_count = candidate_sum / static_cast<double>(queries.count);
    metrics.mean_route_overhead_us = route_overhead_sum / static_cast<double>(queries.count);
    metrics.min_threshold = min_threshold == std::numeric_limits<uint64_t>::max() ? 0 : min_threshold;
    metrics.max_threshold = max_threshold;
  }
  return metrics;
}

int run_single_query_static_rss(Config config) {
  if (config.query_vector_csv.empty()) {
    throw std::runtime_error("--single-query-static-rss requires --query-vector-csv");
  }
  if (config.query_limit != 0 && config.query_limit != 1) {
    throw std::runtime_error("--single-query-static-rss supports exactly one query");
  }
  config.query_limit = 1;
  config.search_threads = 1;

  setenv("PIPEANN_PQ_MMAP", "1", 1);
  setenv("PIPEANN_PQ_MMAP_DROP_CACHE", "1", 0);

  const pipeann::Metric metric = pipeann::get_metric(config.metric);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<float> *nbr_handler = pipeann::get_nbr_handler<float>(metric, "pq");
  if (nbr_handler == nullptr) {
    throw std::runtime_error("failed to create PQ neighbor handler");
  }

  std::atomic<bool> process_sample_rss {true};
  std::atomic<uint64_t> process_peak_rss {current_rss_kb()};
  std::thread process_rss_sampler([&]() {
    while (process_sample_rss.load(std::memory_order_relaxed)) {
      const uint64_t current = current_rss_kb();
      uint64_t previous = process_peak_rss.load(std::memory_order_relaxed);
      while (current > previous &&
             !process_peak_rss.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  pipeann::SSDIndex<float, uint32_t> index(metric, reader, nbr_handler, true);
  index.enable_low_memory_search_mode(true);
  if (index.load(config.source_prefix.c_str(), 1, false) != 0) {
    throw std::runtime_error("failed to load static SSD index");
  }
  if (config.mem_l != 0) {
    index.load_mem_index(config.source_prefix + "_mem.index");
  }

  SearchMetrics metrics = run_static_search(config, index);
  process_sample_rss.store(false, std::memory_order_relaxed);
  process_rss_sampler.join();
  const uint64_t sampled_process_peak = std::max(process_peak_rss.load(std::memory_order_relaxed), current_rss_kb());
  metrics.process_max_rss_kb = std::max(sampled_process_peak, metrics.query_peak_rss_kb);
  const uint64_t live_count = index.meta_.npoints;
  std::ostringstream out;
  out << "{\"mode\":\"" << json_escape(config.mode) << "\","
      << "\"route\":\"" << json_escape(config.route) << "\","
      << "\"threads\":" << config.search_threads << ","
      << "\"insert_threads\":" << config.insert_threads << ","
      << "\"pq_bytes\":" << config.pq_bytes << ","
      << "\"flat_threshold\":" << config.flat_threshold << ","
      << "\"points\":" << live_count << ","
      << "\"chosen_L\":" << config.search_l << ","
      << "\"max_rss_kb\":" << metrics.process_max_rss_kb << ","
      << "\"live_point_count\":" << live_count
      << ",\"rss_path\":\"single_query_static_ssdindex\""
      << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << metrics.elapsed_s
      << ",\"qps\":" << metrics.qps
      << ",\"avg_latency_us\":" << metrics.avg_latency_us
      << ",\"p50_latency_us\":" << metrics.p50_latency_us
      << ",\"p95_latency_us\":" << metrics.p95_latency_us
      << ",\"p99_latency_us\":" << metrics.p99_latency_us
      << ",\"recall@10\":" << metrics.recall
      << ",\"process_max_rss_kb\":" << metrics.process_max_rss_kb
      << ",\"ru_maxrss_kb\":" << max_rss_kb()
      << ",\"sampled_process_peak_rss_kb\":" << sampled_process_peak
      << ",\"rss_before_query_kb\":" << metrics.rss_before_query_kb
      << ",\"rss_after_query_kb\":" << metrics.rss_after_query_kb
      << ",\"query_peak_rss_kb\":" << metrics.query_peak_rss_kb
      << ",\"query_peak_delta_kb\":" << metrics.query_peak_delta_kb
      << ",\"mean_candidate_count\":" << metrics.mean_candidate_count
      << ",\"mean_route_overhead_us\":" << metrics.mean_route_overhead_us
      << ",\"prefilter_count\":" << metrics.prefilter_count
      << ",\"graph_count\":" << metrics.graph_count
      << ",\"fallback_count\":" << metrics.fallback_count
      << ",\"empty_count\":" << metrics.empty_count
      << ",\"min_threshold\":" << metrics.min_threshold
      << ",\"max_threshold\":" << metrics.max_threshold << "}";
  append_jsonl(config, out.str());
  return 0;
}

double insert_range(const Config &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                    std::atomic<bool> *stop_flag, std::atomic<uint64_t> *inserted_out) {
  size_t dim = 0;
  std::vector<float> vectors = load_vector_slice(config.data_bin, config.insert_start, config.insert_count, &dim);
  std::vector<uint32_t> insert_tags = load_insert_tags(config);
  validate_unique_delete_tags(insert_tags);
  std::unique_ptr<pipeann::SpmatLabel> base_labels;
  if (!config.base_label_file.empty()) {
    base_labels.reset(new pipeann::SpmatLabel(config.base_label_file));
    for (uint32_t tag : insert_tags) {
      if (base_labels->labels_.size() <= tag) {
        throw std::runtime_error("base label file does not cover inserted tag " + std::to_string(tag));
      }
    }
  }
  std::atomic<uint64_t> inserted {0};
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic, 64)
  for (int64_t i = 0; i < static_cast<int64_t>(config.insert_count); ++i) {
    if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
      continue;
    }
    const uint32_t tag = insert_tags[static_cast<size_t>(i)];
    index.insert(vectors.data() + static_cast<size_t>(i) * dim, tag);
    if (base_labels != nullptr) {
      const auto &labels = base_labels->labels_[tag];
      index.update_labels(tag, labels.empty() ? nullptr : labels.data(), static_cast<uint32_t>(labels.size()));
    }
    inserted.fetch_add(1, std::memory_order_relaxed);
  }
  const auto end = std::chrono::steady_clock::now();
  if (inserted_out != nullptr) {
    inserted_out->store(inserted.load(), std::memory_order_relaxed);
  }
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

double delete_range(const Config &config, pipeann::DynamicSSDIndex<float, uint32_t> &index) {
  std::vector<uint32_t> tags = load_delete_tags(config);
  validate_unique_delete_tags(tags);
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic, 1024)
  for (int64_t i = 0; i < static_cast<int64_t>(tags.size()); ++i) {
    index.lazy_delete(tags[static_cast<size_t>(i)]);
  }
  const auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

double delete_tags(const Config &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                   const std::vector<uint32_t> &tags) {
  validate_unique_delete_tags(tags);
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic, 1024)
  for (int64_t i = 0; i < static_cast<int64_t>(tags.size()); ++i) {
    index.lazy_delete(tags[static_cast<size_t>(i)]);
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

std::string common_fields(const Config &config, const std::string &mode, uint64_t live_count) {
  std::ostringstream out;
  out << "\"mode\":\"" << json_escape(mode) << "\","
      << "\"status\":\"ok\","
      << "\"route\":\"" << json_escape(config.route) << "\","
      << "\"threads\":" << config.search_threads << ","
      << "\"cpu_cap\":" << config.cpu_cap << ","
      << "\"cpu_cap_enforced\":" << (cpu_cap_enforced(config.cpu_cap) ? "true" : "false") << ","
      << "\"cpu_affinity_allowed_cpus\":\"" << json_escape(cpus_allowed_list()) << "\","
      << "\"source_prefix\":\"" << json_escape(config.source_prefix) << "\","
      << "\"dest_prefix\":\"" << json_escape(config.dest_prefix) << "\","
      << "\"insert_threads\":" << config.insert_threads << ","
      << "\"pq_bytes\":" << config.pq_bytes << ","
      << "\"flat_threshold\":" << config.flat_threshold << ","
      << "\"flat_pq_pivots\":\"" << json_escape(config.flat_pq_pivots) << "\","
      << "\"points\":" << live_count << ","
      << "\"chosen_L\":" << config.search_l << ","
      << "\"search_l\":" << config.search_l << ","
      << "\"max_rss_kb\":" << max_rss_kb() << ","
      << "\"live_point_count\":" << live_count << ","
      << "\"raw_command\":\"" << json_escape(config.raw_command) << "\"";
  return out.str();
}

int run(Config config) {
  if (config.mode == "measure-dynamic-search" && config.single_query_static_rss) {
    return run_single_query_static_rss(config);
  }

  if (config.mode == "zero-insert-only") {
    auto distance = make_distance(config.metric);
    pipeann::IndexBuildParameters parameters;
    parameters.set(config.build_r, config.build_l, 384, 1.2, config.insert_threads, true,
                   config.beamwidth);
    auto index = pipeann::DynamicSSDIndex<float, uint32_t>(
        parameters, config.source_prefix, load_bin_dim(config.data_bin), distance.get(),
        pipeann::get_metric(config.metric), config.flat_threshold, PIPE_SEARCH, config.pq_bytes, config.flat_pq_pivots);

    const std::vector<uint32_t> insert_tags_vec = load_insert_tags(config);
    const std::string inserted_tag_hash = fnv1a_tags_hex(insert_tags_vec);
    const double insert_s = insert_range(config, index, nullptr, nullptr);
    if (index.is_flat_mode()) {
      throw std::runtime_error("zero-insert-only did not materialize to disk");
    }

    double merge_s = 0.0;
    std::string final_prefix = config.source_prefix;
    if (config.final_merge) {
      const auto start = std::chrono::steady_clock::now();
      index.final_merge(config.merge_threads);
      merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      final_prefix = config.source_prefix + "_merge";
    }

    const uint64_t live_count = index.live_point_count();
    const uint64_t label_size = disk_index_label_size(final_prefix);
    const bool sidecar_ok = densebit_sidecar_loadable(final_prefix, live_count);
    if (!sidecar_ok && !config.base_label_file.empty()) {
      throw std::runtime_error("zero-insert-only densebit sidecar missing or unloadable for " + final_prefix);
    }

    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"phase\":\"zero-insert-only\""
        << ",\"final_index_prefix\":\"" << json_escape(final_prefix) << "\""
        << ",\"flat_materialized\":true"
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << (insert_s + merge_s)
        << ",\"insert_count\":" << config.insert_count
        << ",\"insert_scope\":\""
        << (config.insert_tag_file.empty() ? "dense_tag_range_from_insert_start" : "insert_tag_file_tags") << "\""
        << ",\"inserted_tag_hash\":\"" << json_escape(inserted_tag_hash) << "\""
        << ",\"insert_wall_s\":" << insert_s
        << ",\"insert_elapsed_s\":" << insert_s
        << ",\"merge_wall_s\":" << merge_s
        << ",\"merge_elapsed_s\":" << merge_s
        << ",\"wall_s\":" << (insert_s + merge_s)
        << ",\"qps\":" << (static_cast<double>(config.insert_count) / std::max(insert_s, 1e-9))
        << ",\"avg_latency_us\":0,\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0"
        << ",\"main_index_label_size\":" << label_size
        << ",\"label_sidecar_loadable\":" << (sidecar_ok ? "true" : "false")
        << ",\"label_storage_mode\":\"" << (label_size == 0 ? "sidecar" : "embedded") << "\""
        << ",\"disk_format_version\":1}";
    append_jsonl(config, out.str());
    return 0;
  }

  auto distance = make_distance(config.metric);
  auto index = open_dynamic_index(config, distance.get());

  if (config.mode == "insert-only" || config.mode == "reinsert-batch") {
    const std::vector<uint32_t> insert_tags_vec = load_insert_tags(config);
    const std::string inserted_tag_hash = fnv1a_tags_hex(insert_tags_vec);
    const double insert_s = insert_range(config, index, nullptr, nullptr);
    double merge_s = 0.0;
    if (config.final_merge) {
      const auto start = std::chrono::steady_clock::now();
      index.final_merge(config.merge_threads);
      merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    const uint64_t live_count = index.live_point_count();
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << (insert_s + merge_s)
        << ",\"insert_count\":" << config.insert_count
        << ",\"insert_scope\":\""
        << (config.insert_tag_file.empty() ? "dense_tag_range_from_insert_start" : "insert_tag_file_tags") << "\""
        << ",\"inserted_tag_hash\":\"" << json_escape(inserted_tag_hash) << "\""
        << ",\"insert_elapsed_s\":" << insert_s
        << ",\"merge_elapsed_s\":" << merge_s
        << ",\"qps\":" << (static_cast<double>(config.insert_count) / std::max(insert_s, 1e-9))
        << ",\"avg_latency_us\":0,\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "delete-batch") {
    std::vector<uint32_t> delete_tags_vec = load_delete_tags(config);
    validate_unique_delete_tags(delete_tags_vec);
    const std::string deleted_tag_hash = fnv1a_tags_hex(delete_tags_vec);
    const double delete_s = delete_tags(config, index, delete_tags_vec);
    const auto merge_start = std::chrono::steady_clock::now();
    if (config.final_merge) {
      index.final_merge(config.merge_threads);
    }
    const double merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - merge_start).count();
    const uint64_t live_count = index.live_point_count();
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << (delete_s + merge_s)
        << ",\"delete_count\":" << delete_tags_vec.size()
        << ",\"deleted_tag_hash\":\"" << json_escape(deleted_tag_hash) << "\""
        << ",\"delete_scope\":\""
        << (config.delete_id_file.empty() ? "dense_tag_range_from_delete_start" : "delete_id_file_tags") << "\""
        << ",\"delete_elapsed_s\":" << delete_s
        << ",\"merge_elapsed_s\":" << merge_s
        << ",\"qps\":" << (static_cast<double>(delete_tags_vec.size()) / std::max(delete_s, 1e-9))
        << ",\"avg_latency_us\":0,\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "measure-delete-only" || config.mode == "measure-delete-then-merge") {
    std::vector<uint32_t> delete_tags_vec = load_delete_tags(config);
    const std::string deleted_tag_hash = fnv1a_tags_hex(delete_tags_vec);
    const double delete_s = delete_tags(config, index, delete_tags_vec);
    double merge_s = 0.0;
    if (config.mode == "measure-delete-then-merge") {
      const auto merge_start = std::chrono::steady_clock::now();
      index.final_merge(config.merge_threads);
      merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - merge_start).count();
    }
    const uint64_t live_count = index.live_point_count();
    uint64_t label_size = 0;
    bool sidecar_ok = false;
    if (config.mode == "measure-delete-then-merge") {
      label_size = disk_index_label_size(config.dest_prefix);
      sidecar_ok = densebit_sidecar_loadable(config.dest_prefix, live_count);
      if (!sidecar_ok) {
        throw std::runtime_error("merged densebit sidecar missing or unloadable for " + config.dest_prefix);
      }
    }
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"phase\":\"" << json_escape(config.mode) << "\""
        << ",\"delete_count\":" << delete_tags_vec.size()
        << ",\"deleted_tag_hash\":\"" << json_escape(deleted_tag_hash) << "\""
        << ",\"delete_scope\":\""
        << (config.delete_id_file.empty() ? "initial_dense_id_range" : "delete_id_file_tags") << "\""
        << ",\"insert_count\":0"
        << ",\"delete_wall_s\":" << std::fixed << std::setprecision(6) << delete_s
        << ",\"merge_wall_s\":" << merge_s
        << ",\"wall_s\":" << (delete_s + merge_s)
        << ",\"elapsed_s\":" << (delete_s + merge_s)
        << ",\"qps\":" << (static_cast<double>(delete_tags_vec.size()) / std::max(delete_s, 1e-9))
        << ",\"avg_latency_us\":" << (1e6 * delete_s / std::max<size_t>(delete_tags_vec.size(), 1))
        << ",\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0"
        << ",\"main_index_label_size\":" << label_size
        << ",\"label_sidecar_loadable\":" << (sidecar_ok ? "true" : "false")
        << ",\"label_storage_mode\":\"" << (label_size == 0 ? "sidecar" : "embedded") << "\""
        << ",\"disk_format_version\":1}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "measure-dynamic-search") {
    SearchMetrics metrics = run_search(config, index);
    const uint64_t live_count = index.live_point_count();
    std::string actual_route = "none";
    if (metrics.prefilter_count > 0 && metrics.graph_count == 0 && metrics.fallback_count == 0) {
      actual_route = "prefilter";
    } else if (metrics.graph_count > 0 && metrics.prefilter_count == 0 && metrics.fallback_count == 0) {
      actual_route = "graph";
    } else if (metrics.prefilter_count + metrics.graph_count + metrics.fallback_count > 0) {
      actual_route = "mixed";
    }
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"phase\":\"measure-dynamic-search\""
        << ",\"actual_route\":\"" << json_escape(actual_route) << "\""
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << metrics.elapsed_s
        << ",\"wall_s\":" << metrics.elapsed_s
        << ",\"qps\":" << metrics.qps
        << ",\"avg_latency_us\":" << metrics.avg_latency_us
        << ",\"p50_latency_us\":" << metrics.p50_latency_us
        << ",\"p95_latency_us\":" << metrics.p95_latency_us
        << ",\"p99_latency_us\":" << metrics.p99_latency_us
        << ",\"recall@10\":" << metrics.recall
        << ",\"process_max_rss_kb\":" << metrics.process_max_rss_kb
        << ",\"rss_before_query_kb\":" << metrics.rss_before_query_kb
        << ",\"rss_after_query_kb\":" << metrics.rss_after_query_kb
        << ",\"query_peak_rss_kb\":" << metrics.query_peak_rss_kb
        << ",\"query_peak_delta_kb\":" << metrics.query_peak_delta_kb
        << ",\"mean_candidate_count\":" << metrics.mean_candidate_count
        << ",\"candidate_count\":" << metrics.mean_candidate_count
        << ",\"mean_route_overhead_us\":" << metrics.mean_route_overhead_us
        << ",\"prefilter_count\":" << metrics.prefilter_count
        << ",\"graph_count\":" << metrics.graph_count
        << ",\"fallback_count\":" << metrics.fallback_count
        << ",\"empty_count\":" << metrics.empty_count
        << ",\"min_threshold\":" << metrics.min_threshold
        << ",\"max_threshold\":" << metrics.max_threshold
        << ",\"tau_m\":" << metrics.max_threshold
        << ",\"threshold_version\":0}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "search-during-insert") {
    std::atomic<bool> stop {false};
    std::atomic<uint64_t> inserted {0};
    std::thread inserter([&]() {
      insert_range(config, index, &stop, &inserted);
    });
    SearchMetrics metrics = run_search(config, index);
    stop.store(true, std::memory_order_relaxed);
    inserter.join();
    const uint64_t live_count = index.live_point_count();
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << metrics.elapsed_s
        << ",\"qps\":" << metrics.qps
        << ",\"avg_latency_us\":" << metrics.avg_latency_us
        << ",\"p50_latency_us\":" << metrics.p50_latency_us
        << ",\"p95_latency_us\":" << metrics.p95_latency_us
        << ",\"p99_latency_us\":" << metrics.p99_latency_us
        << ",\"recall@10\":" << metrics.recall
        << ",\"inserted_during_search\":" << inserted.load(std::memory_order_relaxed) << "}";
    append_jsonl(config, out.str());
    return 0;
  }

  throw std::runtime_error("unsupported mode: " + config.mode);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(parse_args(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "dynamic_update_suite_driver error: " << error.what() << std::endl;
    return 1;
  }
}
