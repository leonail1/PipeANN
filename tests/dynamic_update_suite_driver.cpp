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
#include <vector>

#include <omp.h>
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
  std::string selector_type = "none";
  std::string route = "auto";
  std::string jsonl_output;
  std::string metric = "l2";
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
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--mode") {
      config.mode = require_value(argc, argv, &index, arg);
    } else if (arg == "--source-prefix") {
      config.source_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--dest-prefix") {
      config.dest_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--data-bin") {
      config.data_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-bin") {
      config.query_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-vector-csv") {
      config.query_vector_csv = require_value(argc, argv, &index, arg);
    } else if (arg == "--truthset-bin") {
      config.truthset_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--base-label-file") {
      config.base_label_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-label-file") {
      config.query_label_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-label-csv") {
      config.query_label_csv = require_value(argc, argv, &index, arg);
    } else if (arg == "--selector-type") {
      config.selector_type = require_value(argc, argv, &index, arg);
    } else if (arg == "--route") {
      config.route = require_value(argc, argv, &index, arg);
    } else if (arg == "--jsonl-output") {
      config.jsonl_output = require_value(argc, argv, &index, arg);
    } else if (arg == "--metric") {
      config.metric = require_value(argc, argv, &index, arg);
    } else if (arg == "--insert-start") {
      config.insert_start = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-count") {
      config.insert_count = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--delete-start") {
      config.delete_start = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--delete-count") {
      config.delete_count = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-threads") {
      config.insert_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--search-threads") {
      config.search_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--merge-threads") {
      config.merge_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--build-l") {
      config.build_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--build-r") {
      config.build_r = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--beamwidth") {
      config.beamwidth = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--k") {
      config.k = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--search-l") {
      config.search_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--mem-l") {
      config.mem_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--query-limit") {
      config.query_limit = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--skip-final-merge") {
      config.final_merge = false;
    } else if (arg == "--single-query-static-rss") {
      config.single_query_static_rss = true;
    } else if (arg == "--help") {
      throw std::runtime_error(
          "usage: dynamic_update_suite_driver --mode insert-only|search-during-insert|delete-batch|reinsert-batch|measure-dynamic-search ...");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (config.mode.empty() || config.source_prefix.empty()) {
    throw std::runtime_error("--mode and --source-prefix are required");
  }
  if ((config.mode == "insert-only" || config.mode == "reinsert-batch" || config.mode == "search-during-insert")
      && config.data_bin.empty()) {
    throw std::runtime_error("--data-bin is required for insert modes");
  }
  if ((config.mode == "insert-only" || config.mode == "reinsert-batch" || config.mode == "delete-batch")
      && config.dest_prefix.empty()) {
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
  std::unique_ptr<pipeann::SpmatLabel> base_labels;
  if (!config.base_label_file.empty()) {
    base_labels.reset(new pipeann::SpmatLabel(config.base_label_file));
    if (base_labels->labels_.size() < config.insert_start + config.insert_count) {
      throw std::runtime_error("base label file does not cover inserted tag range");
    }
  }
  std::atomic<uint64_t> inserted {0};
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic, 64)
  for (int64_t i = 0; i < static_cast<int64_t>(config.insert_count); ++i) {
    if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
      continue;
    }
    const uint32_t tag = static_cast<uint32_t>(config.insert_start + static_cast<uint64_t>(i));
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
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic, 1024)
  for (int64_t i = 0; i < static_cast<int64_t>(config.delete_count); ++i) {
    index.lazy_delete(static_cast<uint32_t>(config.delete_start + static_cast<uint64_t>(i)));
  }
  const auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

std::string common_fields(const Config &config, const std::string &mode, uint64_t live_count) {
  std::ostringstream out;
  out << "\"mode\":\"" << json_escape(mode) << "\","
      << "\"route\":\"" << json_escape(config.route) << "\","
      << "\"threads\":" << config.search_threads << ","
      << "\"insert_threads\":" << config.insert_threads << ","
      << "\"points\":" << live_count << ","
      << "\"chosen_L\":" << config.search_l << ","
      << "\"max_rss_kb\":" << max_rss_kb() << ","
      << "\"live_point_count\":" << live_count;
  return out.str();
}

int run(Config config) {
  if (config.mode == "measure-dynamic-search" && config.single_query_static_rss) {
    return run_single_query_static_rss(config);
  }

  auto distance = make_distance(config.metric);
  auto index = open_dynamic_index(config, distance.get());

  if (config.mode == "insert-only" || config.mode == "reinsert-batch") {
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
        << ",\"insert_elapsed_s\":" << insert_s
        << ",\"merge_elapsed_s\":" << merge_s
        << ",\"qps\":" << (static_cast<double>(config.insert_count) / std::max(insert_s, 1e-9))
        << ",\"avg_latency_us\":0,\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "delete-batch") {
    const double delete_s = delete_range(config, index);
    const auto merge_start = std::chrono::steady_clock::now();
    if (config.final_merge) {
      index.final_merge(config.merge_threads);
    }
    const double merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - merge_start).count();
    const uint64_t live_count = index.live_point_count();
    std::ostringstream out;
    out << "{" << common_fields(config, config.mode, live_count)
        << ",\"elapsed_s\":" << std::fixed << std::setprecision(6) << (delete_s + merge_s)
        << ",\"delete_elapsed_s\":" << delete_s
        << ",\"merge_elapsed_s\":" << merge_s
        << ",\"qps\":" << (static_cast<double>(config.delete_count) / std::max(delete_s, 1e-9))
        << ",\"avg_latency_us\":0,\"p50_latency_us\":0,\"p95_latency_us\":0,\"p99_latency_us\":0,\"recall@10\":0}";
    append_jsonl(config, out.str());
    return 0;
  }

  if (config.mode == "measure-dynamic-search") {
    SearchMetrics metrics = run_search(config, index);
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
        << ",\"process_max_rss_kb\":" << metrics.process_max_rss_kb
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
