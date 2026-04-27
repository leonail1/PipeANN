#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#include "distance.h"
#include "dynamic_index.h"
#include "filter/hybrid_metadata.h"
#include "filter/label.h"
#include "utils/timer.h"
#include "utils.h"

namespace {
constexpr uint64_t kMetadataValidFlag = 1ULL << 0;

struct BucketSpec {
  std::string name;
  double selectivity = 0.0;
};

struct DriverConfig {
  std::string source_prefix;
  std::string dest_prefix;
  std::string data_bin;
  std::string label_file;
  std::string query_bin;
  std::string workload_dir;
  std::string probe_jsonl;
  std::string insert_summary_json;
  std::string selector_type = "intersect";
  std::string metric = "l2";
  uint64_t insert_start = 0;
  uint64_t insert_count = 0;
  uint32_t insert_threads = 1;
  uint32_t search_threads = 1;
  uint32_t materialize_threads = 1;
  uint32_t beamwidth = 4;
  uint64_t k = 10;
  uint64_t search_l = 100;
  uint32_t mem_l = 0;
  uint32_t poll_interval_ms = 100;
  uint32_t recalibration_sample_limit = 200;
  uint32_t recalibration_timeout_s = 900;
  bool skip_final_merge = false;
  std::vector<BucketSpec> bucket_specs;
};

struct ProbeResult {
  std::string bucket_name;
  double selectivity = 0.0;
  size_t query_count = 0;
  double avg_latency_us = 0.0;
  double p50_latency_us = 0.0;
  double p90_latency_us = 0.0;
  double p95_latency_us = 0.0;
  double p99_latency_us = 0.0;
  double qps = 0.0;
  double mean_candidate_count = 0.0;
  double mean_route_overhead_us = 0.0;
  uint64_t prefilter_count = 0;
  uint64_t graph_count = 0;
  uint64_t fallback_count = 0;
  uint64_t empty_count = 0;
  uint64_t min_threshold = 0;
  uint64_t max_threshold = 0;
  uint64_t min_threshold_version = 0;
  uint64_t max_threshold_version = 0;
  uint64_t probe_start_progress_count = 0;
  uint64_t probe_end_progress_count = 0;
  uint64_t live_point_count_start = 0;
  uint64_t live_point_count_end = 0;
};

struct ProbeSessionSummary {
  bool probe_ran = false;
  uint64_t first_probe_start_progress_count = 0;
  uint64_t max_probe_end_progress_count = 0;
};

struct InsertSummary {
  uint64_t insert_count = 0;
  double insert_elapsed_s = 0.0;
  double insert_qps = 0.0;
  double p50_insert_latency_us = 0.0;
  double p90_insert_latency_us = 0.0;
  double p95_insert_latency_us = 0.0;
  double p99_insert_latency_us = 0.0;
};

struct LoadedQueries {
  std::unique_ptr<float[]> data;
  size_t count = 0;
  size_t dim = 0;
};

std::string require_value(int argc, char **argv, int *index, const std::string &flag) {
  if (*index + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  return argv[++(*index)];
}

double percentile_from_sorted(const std::vector<double> &values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

std::string json_escape(const std::string &input) {
  std::string escaped;
  escaped.reserve(input.size() + 8);
  for (char ch : input) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

void append_jsonl_line(const std::string &path, const std::string &line) {
  if (path.empty()) {
    return;
  }
  std::ofstream writer(path, std::ios::app);
  if (!writer.is_open()) {
    throw std::runtime_error("failed to open JSONL output file: " + path);
  }
  writer << line << std::endl;
}

void write_insert_summary_json(const std::string &path, const DriverConfig &config, const InsertSummary &summary,
                               const ProbeSessionSummary &probe_summary) {
  if (path.empty()) {
    return;
  }
  std::ofstream writer(path);
  if (!writer.is_open()) {
    throw std::runtime_error("failed to open insert summary output file: " + path);
  }
  writer << std::fixed << std::setprecision(6);
  writer << "{\n";
  writer << "  \"mode\": \"insert_batch\",\n";
  writer << "  \"source_prefix\": \"" << json_escape(config.source_prefix) << "\",\n";
  writer << "  \"dest_prefix\": \"" << json_escape(config.dest_prefix) << "\",\n";
  writer << "  \"insert_start\": " << config.insert_start << ",\n";
  writer << "  \"insert_count\": " << summary.insert_count << ",\n";
  writer << "  \"insert_elapsed_s\": " << summary.insert_elapsed_s << ",\n";
  writer << "  \"insert_qps\": " << summary.insert_qps << ",\n";
  writer << "  \"p50_insert_latency_us\": " << summary.p50_insert_latency_us << ",\n";
  writer << "  \"p90_insert_latency_us\": " << summary.p90_insert_latency_us << ",\n";
  writer << "  \"p95_insert_latency_us\": " << summary.p95_insert_latency_us << ",\n";
  writer << "  \"p99_insert_latency_us\": " << summary.p99_insert_latency_us << ",\n";
  writer << "  \"probe_ran\": " << (probe_summary.probe_ran ? "true" : "false") << ",\n";
  writer << "  \"first_probe_start_progress_count\": " << probe_summary.first_probe_start_progress_count << ",\n";
  writer << "  \"max_probe_end_progress_count\": " << probe_summary.max_probe_end_progress_count << ",\n";
  writer << "  \"probe_started_near_insert_begin\": "
         << ((probe_summary.probe_ran && probe_summary.first_probe_start_progress_count <= 1) ? "true" : "false")
         << "\n";
  writer << "}\n";
}

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
  throw std::runtime_error("unsupported selector_type: " + selector_type);
}

uint64_t selector_mask_for_kind(pipeann::HybridFilterKind filter_kind) {
  switch (filter_kind) {
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

BucketSpec parse_bucket_spec(const std::string &spec) {
  const size_t colon = spec.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= spec.size()) {
    throw std::runtime_error("invalid --bucket-spec, expected name:selectivity but got: " + spec);
  }
  BucketSpec bucket;
  bucket.name = spec.substr(0, colon);
  bucket.selectivity = std::stod(spec.substr(colon + 1));
  return bucket;
}

DriverConfig parse_args(int argc, char **argv) {
  DriverConfig config;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--source-prefix") {
      config.source_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--dest-prefix") {
      config.dest_prefix = require_value(argc, argv, &index, arg);
    } else if (arg == "--data-bin") {
      config.data_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--label-file") {
      config.label_file = require_value(argc, argv, &index, arg);
    } else if (arg == "--query-bin") {
      config.query_bin = require_value(argc, argv, &index, arg);
    } else if (arg == "--workload-dir") {
      config.workload_dir = require_value(argc, argv, &index, arg);
    } else if (arg == "--probe-jsonl") {
      config.probe_jsonl = require_value(argc, argv, &index, arg);
    } else if (arg == "--insert-summary-json") {
      config.insert_summary_json = require_value(argc, argv, &index, arg);
    } else if (arg == "--selector-type") {
      config.selector_type = require_value(argc, argv, &index, arg);
    } else if (arg == "--metric") {
      config.metric = require_value(argc, argv, &index, arg);
    } else if (arg == "--insert-start") {
      config.insert_start = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-count") {
      config.insert_count = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--insert-threads") {
      config.insert_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--search-threads") {
      config.search_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--materialize-threads") {
      config.materialize_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--beamwidth") {
      config.beamwidth = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--k") {
      config.k = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--search-l") {
      config.search_l = std::stoull(require_value(argc, argv, &index, arg));
    } else if (arg == "--mem-l") {
      config.mem_l = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--poll-interval-ms") {
      config.poll_interval_ms = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--recalibration-sample-limit") {
      config.recalibration_sample_limit = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--recalibration-timeout-s") {
      config.recalibration_timeout_s = static_cast<uint32_t>(std::stoul(require_value(argc, argv, &index, arg)));
    } else if (arg == "--skip-final-merge") {
      config.skip_final_merge = true;
    } else if (arg == "--bucket-spec") {
      config.bucket_specs.push_back(parse_bucket_spec(require_value(argc, argv, &index, arg)));
    } else if (arg == "--probe-progress-fraction") {
      ++index;
    } else if (arg == "--help") {
      throw std::runtime_error(
          "usage: dynamic_prefilter_stage_driver --source-prefix <prefix> --dest-prefix <prefix> --label-file <base.spmat> "
          "--query-bin <query.bin> --workload-dir <dir> --insert-start <n> --insert-count <n> "
          "[--data-bin <bin>] [--probe-jsonl <path>] [--insert-summary-json <path>] "
          "[--bucket-spec name:selectivity ...] [--insert-threads N] [--search-threads N] "
          "[--materialize-threads N] [--beamwidth N] [--k N] [--search-l N] [--mem-l N] "
          "[--selector-type intersect|subset|range] [--metric l2|cosine|mips] [--skip-final-merge]"
      );
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (config.source_prefix.empty() || config.dest_prefix.empty() || config.label_file.empty()
      || config.query_bin.empty() || config.workload_dir.empty()) {
    throw std::runtime_error("missing required arguments for dynamic_prefilter_stage_driver");
  }
  if (config.insert_count > 0 && config.data_bin.empty()) {
    throw std::runtime_error("--data-bin is required when --insert-count > 0");
  }
  if (config.bucket_specs.empty()) {
    throw std::runtime_error("at least one --bucket-spec is required");
  }
  return config;
}

std::unique_ptr<pipeann::Distance<float>> create_distance(const std::string &metric_name) {
  const pipeann::Metric metric = pipeann::get_metric(metric_name);
  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(metric);
  if (distance == nullptr) {
    throw std::runtime_error("failed to create distance function for metric: " + metric_name);
  }
  return std::unique_ptr<pipeann::Distance<float>>(distance);
}

std::vector<float> load_insert_vectors(const DriverConfig &config, size_t *dim_out) {
  if (config.insert_count == 0) {
    *dim_out = 0;
    return {};
  }

  std::ifstream reader(config.data_bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("failed to open data bin: " + config.data_bin);
  }

  int32_t total_points_i32 = 0;
  int32_t dim_i32 = 0;
  reader.read(reinterpret_cast<char *>(&total_points_i32), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
  if (!reader.good()) {
    throw std::runtime_error("failed to read data bin header: " + config.data_bin);
  }

  const uint64_t total_points = static_cast<uint64_t>(total_points_i32);
  const uint64_t dim = static_cast<uint64_t>(dim_i32);
  if (config.insert_start + config.insert_count > total_points) {
    throw std::runtime_error("requested insert slice exceeds data bin size");
  }

  const uint64_t offset_bytes = 2ULL * sizeof(int32_t) + config.insert_start * dim * sizeof(float);
  reader.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
  if (!reader.good()) {
    throw std::runtime_error("failed to seek to insert slice in data bin");
  }

  std::vector<float> vectors(static_cast<size_t>(config.insert_count * dim));
  reader.read(reinterpret_cast<char *>(vectors.data()), static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!reader.good()) {
    throw std::runtime_error("failed to read insert vectors from data bin");
  }
  *dim_out = static_cast<size_t>(dim);
  return vectors;
}

LoadedQueries load_queries(const std::string &query_bin) {
  LoadedQueries queries;
  pipeann::load_bin<float>(query_bin, queries.data, queries.count, queries.dim);
  return queries;
}

uint64_t load_threshold_version(const std::string &index_prefix) {
  const std::string meta_path = pipeann::HybridMetadata::default_metadata_path(index_prefix);
  if (!file_exists(meta_path)) {
    return 0;
  }
  try {
    return pipeann::HybridMetadata::load(meta_path, false)->header().threshold_version;
  } catch (const std::exception &) {
    return 0;
  }
}

bool auto_route_ready(const std::string &index_prefix, uint64_t expected_live_count, uint64_t min_threshold_version,
                      pipeann::HybridFilterKind filter_kind) {
  const std::string meta_path = pipeann::HybridMetadata::default_metadata_path(index_prefix);
  if (!file_exists(meta_path)) {
    return false;
  }
  try {
    const auto metadata = pipeann::HybridMetadata::load(meta_path, true);
    const auto &header = metadata->header();
    if ((header.route_selector_mask & selector_mask_for_kind(filter_kind)) == 0) {
      return false;
    }
    if (header.n_calib != expected_live_count || header.n_live_snapshot != expected_live_count) {
      return false;
    }
    return header.threshold_version >= min_threshold_version;
  } catch (const std::exception &) {
    return false;
  }
}

void write_seed_hybrid_metadata(const std::string &index_prefix, pipeann::HybridFilterKind filter_kind,
                                uint64_t live_count, uint64_t preserved_threshold_version) {
  const std::string sidecar_path = pipeann::DenseBitsetIndex::default_sidecar_path(index_prefix);
  if (!file_exists(sidecar_path)) {
    throw std::runtime_error("missing densebit sidecar for auto route seed: " + sidecar_path);
  }

  const auto densebit = pipeann::DenseBitsetIndex::load(sidecar_path, 0);
  pipeann::HybridMetadataHeaderV1 header;
  header.flags = kMetadataValidFlag;
  header.route_selector_mask = selector_mask_for_kind(filter_kind);
  header.tau_m = 0;
  header.n_calib = live_count == 0 ? 0 : 1;
  header.n_live_snapshot = live_count;
  header.threshold_version = preserved_threshold_version;
  header.calib_epoch_sec = static_cast<uint64_t>(std::time(nullptr));
  header.calib_query_count = 0;
  header.calib_bucket_count = 0;
  header.calib_k = 0;
  header.calib_mem_L = 0;
  header.calib_beamwidth = 0;
  header.calib_l_search = 0;
  header.densebit_npoints = densebit->header().npoints;
  header.densebit_nlabels = densebit->header().nlabels;
  header.densebit_words_per_label = densebit->header().words_per_label;
  header.densebit_nnz = densebit->header().nnz;

  auto metadata = pipeann::HybridMetadata::create(header, {});
  metadata->write_atomically(pipeann::HybridMetadata::default_metadata_path(index_prefix));
}

pipeann::HybridRecalibrationConfig build_recalibration_config(const DriverConfig &config,
                                                              pipeann::HybridFilterKind filter_kind) {
  pipeann::HybridRecalibrationConfig recalibration_config;
  recalibration_config.k = config.k;
  recalibration_config.mem_L = config.mem_l;
  recalibration_config.l_search = config.search_l;
  recalibration_config.beamwidth = config.beamwidth;
  recalibration_config.foreground_budget.active_queries_low_watermark = 1;
  recalibration_config.foreground_budget.waiting_queries_low_watermark = 0;
  for (const auto &bucket : config.bucket_specs) {
    pipeann::HybridRecalibrationDataset dataset;
    dataset.filter_kind = filter_kind;
    dataset.query_bin = config.query_bin;
    dataset.query_label_file = config.workload_dir + "/" + bucket.name + "/probe_query.spmat";
    dataset.sample_limit = config.recalibration_sample_limit;
    recalibration_config.datasets.push_back(std::move(dataset));
  }
  return recalibration_config;
}

void wait_for_auto_recalibration(pipeann::DynamicSSDIndex<float, uint32_t> &index, const DriverConfig &config,
                                 const std::string &index_prefix, pipeann::HybridFilterKind filter_kind,
                                 uint64_t expected_live_count, uint64_t min_threshold_version) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(config.recalibration_timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto_route_ready(index_prefix, expected_live_count, min_threshold_version, filter_kind)
        && index.hybrid_recalibration_state() == pipeann::HybridRecalibrationState::kIdle) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(config.poll_interval_ms));
  }

  throw std::runtime_error("timed out waiting for automatic tau_m recalibration on " + index_prefix);
}

void prime_auto_route(pipeann::DynamicSSDIndex<float, uint32_t> &index, const DriverConfig &config,
                      pipeann::HybridFilterKind filter_kind, const std::string &index_prefix,
                      uint64_t expected_live_count) {
  const uint64_t previous_threshold_version = load_threshold_version(index_prefix);
  write_seed_hybrid_metadata(index_prefix, filter_kind, expected_live_count, previous_threshold_version);
  index.configure_hybrid_recalibration(build_recalibration_config(config, filter_kind));
  wait_for_auto_recalibration(index, config, index_prefix, filter_kind, expected_live_count,
                              previous_threshold_version + 1ULL);
}

std::string active_input_prefix(const DriverConfig &config) {
  const std::string shadow_prefix = config.source_prefix + "_shadow";
  return file_exists(shadow_prefix + "_disk.index") ? shadow_prefix : config.source_prefix;
}

void relabel_existing_points(const DriverConfig &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                             const pipeann::SpmatLabel &base_labels) {
  if (config.insert_start == 0) {
    return;
  }
  if (base_labels.labels_.size() < config.insert_start) {
    throw std::runtime_error("base label file does not cover the requested existing visible corpus");
  }

  for (uint64_t tag = 0; tag < config.insert_start; ++tag) {
    const auto &labels = base_labels.labels_[static_cast<size_t>(tag)];
    const uint32_t *label_ptr = labels.empty() ? nullptr : labels.data();
    const int label_result = index.update_labels(static_cast<uint32_t>(tag), label_ptr,
                                                 static_cast<uint32_t>(labels.size()));
    if (label_result != 0) {
      throw std::runtime_error("failed to relabel existing point " + std::to_string(tag));
    }
  }
}

ProbeResult run_probe_for_bucket(const DriverConfig &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                                 const LoadedQueries &queries, const BucketSpec &bucket_spec,
                                 pipeann::HybridFilterKind filter_kind, std::atomic<uint64_t> *progress) {
  const std::string label_file = config.workload_dir + "/" + bucket_spec.name + "/probe_query.spmat";
  pipeann::SpmatLabel query_labels(label_file);
  if (query_labels.labels_.size() != queries.count) {
    throw std::runtime_error("probe query label count mismatch for bucket " + bucket_spec.name);
  }

  const size_t max_filter_size = query_labels.label_size();
  std::vector<std::vector<char>> filter_buffers(queries.count);
  for (size_t query_idx = 0; query_idx < queries.count; ++query_idx) {
    filter_buffers[query_idx].resize(max_filter_size, 0);
    query_labels.write(static_cast<uint32_t>(query_idx), filter_buffers[query_idx].data());
  }

  std::vector<uint32_t> all_tags(static_cast<size_t>(queries.count * config.k), std::numeric_limits<uint32_t>::max());
  std::vector<float> all_dists(static_cast<size_t>(queries.count * config.k), std::numeric_limits<float>::infinity());
  std::vector<pipeann::QueryStats> stats(queries.count);
  std::vector<pipeann::HybridQueryStats> hybrid_stats(queries.count);
  std::vector<double> latencies_us(queries.count, 0.0);

  ProbeResult result;
  result.bucket_name = bucket_spec.name;
  result.selectivity = bucket_spec.selectivity;
  result.query_count = queries.count;
  result.probe_start_progress_count = progress == nullptr ? 0 : progress->load();
  result.live_point_count_start = index.live_point_count();

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(config.search_threads) schedule(dynamic, 1)
  for (int64_t query_idx = 0; query_idx < static_cast<int64_t>(queries.count); ++query_idx) {
    index.search(queries.data.get() + (query_idx * queries.dim), config.k, config.mem_l, config.search_l,
                 config.beamwidth, all_tags.data() + (query_idx * config.k),
                 all_dists.data() + (query_idx * config.k), &stats[static_cast<size_t>(query_idx)], true,
                 filter_kind, filter_buffers[static_cast<size_t>(query_idx)].data(),
                 &hybrid_stats[static_cast<size_t>(query_idx)], pipeann::HybridRouteOverride::kAuto);
    latencies_us[static_cast<size_t>(query_idx)] = stats[static_cast<size_t>(query_idx)].total_us;
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::sort(latencies_us.begin(), latencies_us.end());
  const double total_elapsed_s = std::chrono::duration<double>(end - start).count();
  const double avg_latency_us = latencies_us.empty()
      ? 0.0
      : (std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / static_cast<double>(latencies_us.size()));

  double sum_candidates = 0.0;
  double sum_route_overhead_us = 0.0;
  uint64_t min_threshold = std::numeric_limits<uint64_t>::max();
  uint64_t max_threshold = 0;
  uint64_t min_threshold_version = std::numeric_limits<uint64_t>::max();
  uint64_t max_threshold_version = 0;
  for (size_t query_idx = 0; query_idx < queries.count; ++query_idx) {
    const auto &hybrid = hybrid_stats[query_idx];
    sum_candidates += static_cast<double>(hybrid.candidate_count);
    sum_route_overhead_us += static_cast<double>(hybrid.route_overhead_us);
    min_threshold = std::min(min_threshold, hybrid.threshold);
    max_threshold = std::max(max_threshold, hybrid.threshold);
    min_threshold_version = std::min(min_threshold_version, hybrid.threshold_version);
    max_threshold_version = std::max(max_threshold_version, hybrid.threshold_version);
    switch (hybrid.decision) {
      case pipeann::HybridRouteDecision::kPrefilter:
        ++result.prefilter_count;
        break;
      case pipeann::HybridRouteDecision::kGraphOnly:
        ++result.graph_count;
        break;
      case pipeann::HybridRouteDecision::kPrefilterFastReturn:
        ++result.empty_count;
        break;
      case pipeann::HybridRouteDecision::kAutoGraphFallback:
      default:
        ++result.fallback_count;
        break;
    }
  }

  result.avg_latency_us = avg_latency_us;
  result.p50_latency_us = percentile_from_sorted(latencies_us, 0.50);
  result.p90_latency_us = percentile_from_sorted(latencies_us, 0.90);
  result.p95_latency_us = percentile_from_sorted(latencies_us, 0.95);
  result.p99_latency_us = percentile_from_sorted(latencies_us, 0.99);
  result.qps = total_elapsed_s > 0.0 ? (static_cast<double>(queries.count) / total_elapsed_s) : 0.0;
  result.mean_candidate_count = queries.count == 0 ? 0.0 : (sum_candidates / static_cast<double>(queries.count));
  result.mean_route_overhead_us =
      queries.count == 0 ? 0.0 : (sum_route_overhead_us / static_cast<double>(queries.count));
  result.min_threshold = min_threshold == std::numeric_limits<uint64_t>::max() ? 0 : min_threshold;
  result.max_threshold = max_threshold;
  result.min_threshold_version =
      min_threshold_version == std::numeric_limits<uint64_t>::max() ? 0 : min_threshold_version;
  result.max_threshold_version = max_threshold_version;
  result.probe_end_progress_count = progress == nullptr ? 0 : progress->load();
  result.live_point_count_end = index.live_point_count();
  return result;
}

void emit_probe_result(const std::string &jsonl_path, const ProbeResult &result) {
  std::ostringstream line;
  line << std::fixed << std::setprecision(6);
  line << "{";
  line << "\"mode\":\"during_insert_probe\",";
  line << "\"route\":\"auto\",";
  line << "\"bucket_name\":\"" << json_escape(result.bucket_name) << "\",";
  line << "\"selectivity\":" << result.selectivity << ",";
  line << "\"query_count\":" << result.query_count << ",";
  line << "\"avg_latency_us\":" << result.avg_latency_us << ",";
  line << "\"p50_latency_us\":" << result.p50_latency_us << ",";
  line << "\"p90_latency_us\":" << result.p90_latency_us << ",";
  line << "\"p95_latency_us\":" << result.p95_latency_us << ",";
  line << "\"p99_latency_us\":" << result.p99_latency_us << ",";
  line << "\"qps\":" << result.qps << ",";
  line << "\"mean_candidate_count\":" << result.mean_candidate_count << ",";
  line << "\"mean_route_overhead_us\":" << result.mean_route_overhead_us << ",";
  line << "\"prefilter_count\":" << result.prefilter_count << ",";
  line << "\"graph_count\":" << result.graph_count << ",";
  line << "\"fallback_count\":" << result.fallback_count << ",";
  line << "\"empty_count\":" << result.empty_count << ",";
  line << "\"min_threshold\":" << result.min_threshold << ",";
  line << "\"max_threshold\":" << result.max_threshold << ",";
  line << "\"min_threshold_version\":" << result.min_threshold_version << ",";
  line << "\"max_threshold_version\":" << result.max_threshold_version << ",";
  line << "\"probe_start_progress_count\":" << result.probe_start_progress_count << ",";
  line << "\"probe_end_progress_count\":" << result.probe_end_progress_count << ",";
  line << "\"live_point_count_start\":" << result.live_point_count_start << ",";
  line << "\"live_point_count_end\":" << result.live_point_count_end;
  line << "}";
  append_jsonl_line(jsonl_path, line.str());
}

ProbeSessionSummary run_probe_sequence(const DriverConfig &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                                       const LoadedQueries &queries, pipeann::HybridFilterKind filter_kind,
                                       std::atomic<uint64_t> *progress) {
  ProbeSessionSummary summary;
  bool first_bucket = true;
  for (const auto &bucket : config.bucket_specs) {
    const ProbeResult result = run_probe_for_bucket(config, index, queries, bucket, filter_kind, progress);
    emit_probe_result(config.probe_jsonl, result);
    if (first_bucket) {
      summary.first_probe_start_progress_count = result.probe_start_progress_count;
      first_bucket = false;
    }
    summary.max_probe_end_progress_count = std::max(summary.max_probe_end_progress_count, result.probe_end_progress_count);
    summary.probe_ran = true;
  }
  return summary;
}

InsertSummary insert_batch(const DriverConfig &config, pipeann::DynamicSSDIndex<float, uint32_t> &index,
                           const std::vector<float> &vectors, size_t dim, const pipeann::SpmatLabel &base_labels,
                           std::atomic<uint64_t> *progress) {
  InsertSummary summary;
  summary.insert_count = config.insert_count;
  if (config.insert_count == 0) {
    return summary;
  }
  if (base_labels.labels_.size() < config.insert_start + config.insert_count) {
    throw std::runtime_error("base label file does not cover the requested insert range");
  }

  std::vector<double> latencies_us(static_cast<size_t>(config.insert_count), 0.0);
  std::atomic<int> failure_count(0);
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(config.insert_threads) schedule(dynamic)
  for (int64_t offset = 0; offset < static_cast<int64_t>(config.insert_count); ++offset) {
    pipeann::Timer timer;
    const uint32_t tag = static_cast<uint32_t>(config.insert_start + static_cast<uint64_t>(offset));
    const float *point = vectors.data() + (static_cast<size_t>(offset) * dim);

    const int insert_result = index.insert(point, tag);
    if (insert_result < 0) {
      failure_count.fetch_add(1);
      continue;
    }

    const auto &labels = base_labels.labels_[tag];
    const uint32_t *label_ptr = labels.empty() ? nullptr : labels.data();
    const int label_result = index.update_labels(tag, label_ptr, static_cast<uint32_t>(labels.size()));
    if (label_result != 0) {
      failure_count.fetch_add(1);
      continue;
    }

    progress->fetch_add(1);
    latencies_us[static_cast<size_t>(offset)] = static_cast<double>(timer.elapsed());
  }

  auto end = std::chrono::high_resolution_clock::now();
  if (failure_count.load() != 0) {
    throw std::runtime_error("insert/update_labels failed for one or more points during batch insert");
  }

  std::sort(latencies_us.begin(), latencies_us.end());
  summary.insert_elapsed_s = std::chrono::duration<double>(end - start).count();
  summary.insert_qps = summary.insert_elapsed_s > 0.0
      ? (static_cast<double>(config.insert_count) / summary.insert_elapsed_s)
      : 0.0;
  summary.p50_insert_latency_us = percentile_from_sorted(latencies_us, 0.50);
  summary.p90_insert_latency_us = percentile_from_sorted(latencies_us, 0.90);
  summary.p95_insert_latency_us = percentile_from_sorted(latencies_us, 0.95);
  summary.p99_insert_latency_us = percentile_from_sorted(latencies_us, 0.99);
  return summary;
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const DriverConfig config = parse_args(argc, argv);
    const pipeann::Metric metric = pipeann::get_metric(config.metric);
    const pipeann::HybridFilterKind filter_kind = parse_filter_kind(config.selector_type);

    size_t vector_dim = 0;
    const std::vector<float> insert_vectors = load_insert_vectors(config, &vector_dim);
    const LoadedQueries queries = load_queries(config.query_bin);
    if (config.insert_count > 0 && queries.dim != vector_dim) {
      throw std::runtime_error("query dim does not match insert vector dim");
    }

    pipeann::SpmatLabel base_labels(config.label_file);
    auto distance = create_distance(config.metric);
    pipeann::IndexBuildParameters parameters;
    parameters.set(0, static_cast<uint32_t>(config.search_l), 384, 1.2,
                   config.insert_threads + config.search_threads, true, config.beamwidth);

    pipeann::DynamicSSDIndex<float, uint32_t> index(parameters, config.source_prefix, config.dest_prefix,
                                                    distance.get(), metric, BEAM_SEARCH, config.mem_l > 0);

    relabel_existing_points(config, index, base_labels);
    prime_auto_route(index, config, filter_kind, active_input_prefix(config), index.live_point_count());

    ProbeSessionSummary probe_summary;
    InsertSummary insert_summary;
    std::atomic<uint64_t> progress(0);

    if (config.insert_count > 0) {
      std::promise<void> start_promise;
      std::shared_future<void> start_future(start_promise.get_future());

      auto insert_future = std::async(std::launch::async, [&, start_future]() mutable {
        start_future.wait();
        return insert_batch(config, index, insert_vectors, vector_dim, base_labels, &progress);
      });

      auto probe_future = std::async(std::launch::async, [&, start_future]() mutable {
        start_future.wait();
        return run_probe_sequence(config, index, queries, filter_kind, &progress);
      });

      start_promise.set_value();
      insert_summary = insert_future.get();
      probe_summary = probe_future.get();
      write_insert_summary_json(config.insert_summary_json, config, insert_summary, probe_summary);
    }

    if (!config.skip_final_merge) {
      index.final_merge(config.materialize_threads);
      prime_auto_route(index, config, filter_kind, config.dest_prefix, index.live_point_count());
      LOG(INFO) << "[ok] materialized updated prefix to " << config.dest_prefix;
    }
    return 0;
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}
