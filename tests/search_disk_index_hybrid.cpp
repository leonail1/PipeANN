#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <omp.h>
#include <ssd_index.h>

#include "filter/label.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "utils/log.h"

namespace {
struct StructuredOutputOptions {
  std::string jsonl_output;
  std::string query_vector_csv;
  std::string query_label_csv;
  bool single_query_static_rss = false;

  bool enabled() const {
    return !jsonl_output.empty();
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

bool parse_route_override(const std::string &force_route, pipeann::HybridRouteOverride *route_override,
                          bool *validate_auto_route) {
  *validate_auto_route = false;
  if (force_route == "auto") {
    *route_override = pipeann::HybridRouteOverride::kAuto;
    return true;
  }
  if (force_route == "validate-auto") {
    *route_override = pipeann::HybridRouteOverride::kAuto;
    *validate_auto_route = true;
    return true;
  }
  if (force_route == "prefilter") {
    *route_override = pipeann::HybridRouteOverride::kForcePrefilter;
    return true;
  }
  if (force_route == "graph") {
    *route_override = pipeann::HybridRouteOverride::kForceGraphOnly;
    return true;
  }
  return false;
}

bool results_match(size_t lhs_count, const std::vector<uint32_t> &lhs_tags, const std::vector<float> &lhs_dists,
                   size_t rhs_count, const std::vector<uint32_t> &rhs_tags, const std::vector<float> &rhs_dists) {
  return lhs_count == rhs_count && lhs_tags == rhs_tags && lhs_dists == rhs_dists;
}

bool parse_structured_output_options(int argc, char **argv, int start_index, StructuredOutputOptions *options) {
  for (int index = start_index; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--jsonl-output") {
      if (index + 1 >= argc) {
        LOG(ERROR) << "Missing value for --jsonl-output";
        return false;
      }
      options->jsonl_output = argv[++index];
      continue;
    }
    if (arg == "--query-vector-csv") {
      if (index + 1 >= argc) {
        LOG(ERROR) << "Missing value for --query-vector-csv";
        return false;
      }
      options->query_vector_csv = argv[++index];
      continue;
    }
    if (arg == "--query-label-csv") {
      if (index + 1 >= argc) {
        LOG(ERROR) << "Missing value for --query-label-csv";
        return false;
      }
      options->query_label_csv = argv[++index];
      continue;
    }
    if (arg == "--single-query-static-rss") {
      options->single_query_static_rss = true;
      continue;
    }

    LOG(ERROR) << "Unknown option: " << arg;
    return false;
  }

  return true;
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

void append_jsonl_line(const StructuredOutputOptions &options, const std::string &line) {
  if (!options.enabled()) {
    return;
  }

  std::ofstream writer(options.jsonl_output, std::ios::app);
  if (!writer.is_open()) {
    throw std::runtime_error("Failed to open JSONL output file: " + options.jsonl_output);
  }
  writer << line << std::endl;
}

std::vector<float> parse_float_csv(const std::string &csv) {
  std::vector<float> values;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    const std::string token = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!token.empty()) {
      values.push_back(std::stof(token));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return values;
}

std::vector<uint32_t> parse_uint32_csv(const std::string &csv) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    const std::string token = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!token.empty()) {
      values.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return values;
}

uint64_t current_hwm_kb() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::istringstream parser(line);
      std::string key;
      uint64_t value = 0;
      std::string unit;
      parser >> key >> value >> unit;
      return value;
    }
  }
  return 0;
}
}  // namespace

template<typename T>
int search_disk_index(int argc, char **argv) {
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num = 0, query_dim = 0, gt_num = 0, gt_dim = 0;
  std::vector<uint64_t> Lvec;

  int index = 2;
  const std::string index_prefix_path(argv[index++]);
  const uint32_t num_threads = std::atoi(argv[index++]);
  const uint32_t beamwidth = std::atoi(argv[index++]);
  const std::string query_bin(argv[index++]);
  const std::string truthset_bin(argv[index++]);
  const uint64_t recall_at = std::atoi(argv[index++]);
  const std::string dist_metric(argv[index++]);
  const std::string nbr_type(argv[index++]);
  const std::string selector_type(argv[index++]);
  const std::string query_label_file(argv[index++]);
  const std::string force_route(argv[index++]);
  const uint64_t relaxed_monotonicity_lmax = std::atoi(argv[index++]);
  const uint32_t mem_L = std::atoi(argv[index++]);
  int optional_index = index;
  while (optional_index < argc && std::strncmp(argv[optional_index], "--", 2) != 0) {
    const uint64_t cur_l = std::atoi(argv[optional_index]);
    if (cur_l >= recall_at) {
      Lvec.push_back(cur_l);
    }
    ++optional_index;
  }

  StructuredOutputOptions structured_output_options;
  if (!parse_structured_output_options(argc, argv, optional_index, &structured_output_options)) {
    return -1;
  }

  const pipeann::HybridFilterKind filter_kind = parse_filter_kind(selector_type);
  if (filter_kind == pipeann::HybridFilterKind::kUnsupported) {
    LOG(ERROR) << "Hybrid search driver only supports intersect/subset/range selectors. Got: " << selector_type;
    return -1;
  }

  pipeann::HybridRouteOverride route_override;
  bool validate_auto_route = false;
  if (!parse_route_override(force_route, &route_override, &validate_auto_route)) {
    LOG(ERROR) << "Unknown force_route: " << force_route << ". Use auto/validate-auto/prefilter/graph";
    return -1;
  }

  if (relaxed_monotonicity_lmax != 0) {
    LOG(ERROR) << "search_disk_index_hybrid does not support relaxed_monotonicity_lmax yet";
    return -1;
  }

  const pipeann::Metric metric = pipeann::get_metric(dist_metric);

  if (Lvec.empty()) {
    LOG(ERROR) << "No valid Lsearch found. Lsearch must be >= recall_at";
    return -1;
  }

  LOG(INFO) << "Search parameters: threads=" << num_threads << ", beamwidth=" << beamwidth
            << ", force_route=" << force_route;

  if (!structured_output_options.query_vector_csv.empty()) {
    const std::vector<float> values = parse_float_csv(structured_output_options.query_vector_csv);
    if (values.empty()) {
      LOG(ERROR) << "--query-vector-csv is empty";
      return -1;
    }
    query_num = 1;
    query_dim = values.size();
    query = new T[values.size()];
    for (size_t dim = 0; dim < values.size(); ++dim) {
      query[dim] = static_cast<T>(values[dim]);
    }
  } else {
    pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
  }

  bool calc_recall_flag = false;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and ground truth data";
      return -1;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  if (nbr_handler == nullptr) {
    LOG(ERROR) << "Unknown neighbor type: " << nbr_type;
    return -1;
  }

  std::unique_ptr<pipeann::SSDIndex<T>> index_ptr(new pipeann::SSDIndex<T>(metric, reader, nbr_handler, true));
  index_ptr->enable_low_memory_search_mode(true);
  if (index_ptr->load(index_prefix_path.c_str(), num_threads, false) != 0) {
    return -1;
  }

  if (mem_L != 0) {
    const auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    index_ptr->load_mem_index(mem_index_path);
  }

  LOG(INFO) << "Hybrid runtime enabled: " << (index_ptr->hybrid_enabled() ? "true" : "false");
  if ((route_override == pipeann::HybridRouteOverride::kAuto || validate_auto_route) && !index_ptr->hybrid_enabled()) {
    LOG(ERROR) << force_route << " requires calibrated hybrid runtime assets to be present and loadable";
    return -1;
  }

  omp_set_num_threads(num_threads);

  std::vector<std::vector<char>> filter_buffers(query_num);
  if (!structured_output_options.query_label_csv.empty()) {
    if (query_num != 1) {
      LOG(ERROR) << "--query-label-csv supports exactly one query";
      return -1;
    }
    const std::vector<uint32_t> labels = parse_uint32_csv(structured_output_options.query_label_csv);
    filter_buffers[0].resize(sizeof(uint32_t) + labels.size() * sizeof(uint32_t), 0);
    const uint32_t label_count = static_cast<uint32_t>(labels.size());
    memcpy(filter_buffers[0].data(), &label_count, sizeof(uint32_t));
    if (!labels.empty()) {
      memcpy(filter_buffers[0].data() + sizeof(uint32_t), labels.data(), labels.size() * sizeof(uint32_t));
    }
  } else {
    pipeann::SpmatLabel query_labels(query_label_file);
    if (query_labels.labels_.size() != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and query labels";
      return -1;
    }
    const size_t max_filter_size = query_labels.label_size();
    for (size_t query_idx = 0; query_idx < query_num; ++query_idx) {
      filter_buffers[query_idx].resize(max_filter_size, 0);
      query_labels.write(query_idx, filter_buffers[query_idx].data());
    }
  }

  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) -> bool {
    const uint64_t L = Lvec[test_id];
    if (validate_auto_route) {
      uint64_t result_mismatch_count = 0;
      uint64_t decision_mismatch_count = 0;
      uint64_t fallback_count = 0;
      uint64_t prefilter_count = 0;
      uint64_t graph_count = 0;
      uint64_t empty_count = 0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+:result_mismatch_count, decision_mismatch_count, fallback_count, prefilter_count, graph_count, empty_count)
      for (int64_t query_idx = 0; query_idx < static_cast<int64_t>(query_num); ++query_idx) {
        std::vector<uint32_t> auto_tags(static_cast<size_t>(recall_at), std::numeric_limits<uint32_t>::max());
        std::vector<float> auto_dists(static_cast<size_t>(recall_at), std::numeric_limits<float>::infinity());
        pipeann::QueryStats auto_stats{};
        pipeann::HybridQueryStats auto_hybrid_stats{};
        const size_t auto_result_count = index_ptr->hybrid_search(
            query + (query_idx * query_dim), recall_at, mem_L, L, auto_tags.data(), auto_dists.data(), beamwidth,
            filter_kind, filter_buffers[query_idx].data(), &auto_stats, &auto_hybrid_stats,
            pipeann::HybridRouteOverride::kAuto);

        bool decision_mismatch = false;
        switch (auto_hybrid_stats.decision) {
          case pipeann::HybridRouteDecision::kAutoGraphFallback:
            ++fallback_count;
            decision_mismatch = true;
            break;
          case pipeann::HybridRouteDecision::kPrefilter:
            ++prefilter_count;
            decision_mismatch = auto_hybrid_stats.candidate_count == 0
                                || auto_hybrid_stats.candidate_count > auto_hybrid_stats.threshold;
            break;
          case pipeann::HybridRouteDecision::kGraphOnly:
            ++graph_count;
            decision_mismatch = auto_hybrid_stats.candidate_count == 0
                                || auto_hybrid_stats.candidate_count <= auto_hybrid_stats.threshold;
            break;
          case pipeann::HybridRouteDecision::kPrefilterFastReturn:
            ++empty_count;
            decision_mismatch = auto_hybrid_stats.candidate_count != 0 || auto_result_count != 0;
            break;
        }
        if (decision_mismatch) {
          ++decision_mismatch_count;
        }

        if (auto_hybrid_stats.decision == pipeann::HybridRouteDecision::kAutoGraphFallback
            || auto_hybrid_stats.decision == pipeann::HybridRouteDecision::kPrefilterFastReturn) {
          continue;
        }

        const pipeann::HybridRouteOverride reference_override =
            auto_hybrid_stats.decision == pipeann::HybridRouteDecision::kPrefilter
                ? pipeann::HybridRouteOverride::kForcePrefilter
                : pipeann::HybridRouteOverride::kForceGraphOnly;
        const pipeann::HybridRouteDecision expected_reference_decision =
            auto_hybrid_stats.decision == pipeann::HybridRouteDecision::kPrefilter
                ? pipeann::HybridRouteDecision::kPrefilter
                : pipeann::HybridRouteDecision::kGraphOnly;

        std::vector<uint32_t> reference_tags(static_cast<size_t>(recall_at), std::numeric_limits<uint32_t>::max());
        std::vector<float> reference_dists(static_cast<size_t>(recall_at), std::numeric_limits<float>::infinity());
        pipeann::QueryStats reference_stats{};
        pipeann::HybridQueryStats reference_hybrid_stats{};
        const size_t reference_result_count = index_ptr->hybrid_search(
            query + (query_idx * query_dim), recall_at, mem_L, L, reference_tags.data(), reference_dists.data(),
            beamwidth, filter_kind, filter_buffers[query_idx].data(), &reference_stats, &reference_hybrid_stats,
            reference_override);

        if (reference_hybrid_stats.decision != expected_reference_decision) {
          ++decision_mismatch_count;
        }
        if (!results_match(auto_result_count, auto_tags, auto_dists, reference_result_count, reference_tags,
                           reference_dists)) {
          ++result_mismatch_count;
        }
      }

      if (output) {
        std::cout << std::setw(6) << L << std::setw(12) << query_num << std::setw(12) << result_mismatch_count
                  << std::setw(14) << decision_mismatch_count << std::setw(10) << fallback_count << std::setw(10)
                  << prefilter_count << std::setw(10) << graph_count << std::setw(10) << empty_count << std::endl;
      }

      append_jsonl_line(
          structured_output_options,
          "{\"format\":\"pipeann.hybrid.validate.v1\","
          "\"index_prefix\":\"" + json_escape(index_prefix_path) + "\"," 
          "\"selector_type\":\"" + json_escape(selector_type) + "\"," 
          "\"query_label_file\":\"" + json_escape(query_label_file) + "\"," 
          "\"query_count\":" + std::to_string(query_num) + ","
          "\"route\":\"" + json_escape(force_route) + "\"," 
          "\"L\":" + std::to_string(L) + ","
          "\"result_mismatch_count\":" + std::to_string(result_mismatch_count) + ","
          "\"decision_mismatch_count\":" + std::to_string(decision_mismatch_count) + ","
          "\"fallback_count\":" + std::to_string(fallback_count) + ","
          "\"prefilter_count\":" + std::to_string(prefilter_count) + ","
          "\"graph_count\":" + std::to_string(graph_count) + ","
          "\"empty_count\":" + std::to_string(empty_count) + "}");

      return result_mismatch_count == 0 && decision_mismatch_count == 0 && fallback_count == 0;
    }

    auto *stats = new pipeann::QueryStats[query_num];
    auto *hybrid_stats = new pipeann::HybridQueryStats[query_num];

    query_result_tags[test_id].assign(static_cast<size_t>(recall_at) * query_num,
                                      std::numeric_limits<uint32_t>::max());
    query_result_dists[test_id].assign(static_cast<size_t>(recall_at) * query_num,
                                       std::numeric_limits<float>::infinity());

    const auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t query_idx = 0; query_idx < static_cast<int64_t>(query_num); ++query_idx) {
      index_ptr->hybrid_search(query + (query_idx * query_dim), recall_at, mem_L, L,
                               query_result_tags[test_id].data() + (query_idx * recall_at),
                               query_result_dists[test_id].data() + (query_idx * recall_at), beamwidth,
                               filter_kind, filter_buffers[query_idx].data(), stats + query_idx,
                               hybrid_stats + query_idx, route_override);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    const float qps = diff.count() > 0.0 ? static_cast<float>(query_num / diff.count()) : 0.0f;
    const float mean_latency = static_cast<float>(
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.total_us; }));
    const float latency_50 = static_cast<float>(pipeann::get_percentile_stats(
        stats, query_num, 0.50f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    const float latency_95 = static_cast<float>(pipeann::get_percentile_stats(
        stats, query_num, 0.95f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    const float latency_99 = static_cast<float>(pipeann::get_percentile_stats(
        stats, query_num, 0.99f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    const float latency_999 = static_cast<float>(pipeann::get_percentile_stats(
        stats, query_num, 0.999f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    const float mean_hops = static_cast<float>(
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_hops; }));
    const float mean_ios = static_cast<float>(
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_ios; }));

    double sum_candidates = 0.0;
    double sum_route_overhead_us = 0.0;
    uint64_t fallback_count = 0;
    uint64_t prefilter_count = 0;
    uint64_t graph_count = 0;
    uint64_t empty_count = 0;
    uint64_t min_threshold = std::numeric_limits<uint64_t>::max();
    uint64_t max_threshold = 0;
    uint64_t min_threshold_version = std::numeric_limits<uint64_t>::max();
    uint64_t max_threshold_version = 0;
    for (size_t query_idx = 0; query_idx < query_num; ++query_idx) {
      sum_candidates += static_cast<double>(hybrid_stats[query_idx].candidate_count);
      sum_route_overhead_us += static_cast<double>(hybrid_stats[query_idx].route_overhead_us);
      min_threshold = std::min(min_threshold, hybrid_stats[query_idx].threshold);
      max_threshold = std::max(max_threshold, hybrid_stats[query_idx].threshold);
      min_threshold_version = std::min(min_threshold_version, hybrid_stats[query_idx].threshold_version);
      max_threshold_version = std::max(max_threshold_version, hybrid_stats[query_idx].threshold_version);
      switch (hybrid_stats[query_idx].decision) {
        case pipeann::HybridRouteDecision::kAutoGraphFallback:
          ++fallback_count;
          break;
        case pipeann::HybridRouteDecision::kPrefilter:
          ++prefilter_count;
          break;
        case pipeann::HybridRouteDecision::kGraphOnly:
          ++graph_count;
          break;
        case pipeann::HybridRouteDecision::kPrefilterFastReturn:
          ++empty_count;
          break;
      }
    }
    if (query_num == 0) {
      min_threshold = 0;
      min_threshold_version = 0;
    }
    std::string resolved_route = "mixed";
    if (prefilter_count == query_num) {
      resolved_route = "prefilter";
    } else if (graph_count == query_num) {
      resolved_route = "graph";
    } else if (empty_count == query_num) {
      resolved_route = "empty";
    } else if (fallback_count == query_num) {
      resolved_route = "fallback";
    }
    const std::string tau_m_json =
        min_threshold == max_threshold ? std::to_string(max_threshold) : std::string("null");
    const std::string threshold_version_json = min_threshold_version == max_threshold_version
                                                   ? std::to_string(max_threshold_version)
                                                   : std::string("null");
    const double mean_candidates = query_num == 0 ? 0.0 : (sum_candidates / static_cast<double>(query_num));
    const double mean_route_overhead =
        query_num == 0 ? 0.0 : (sum_route_overhead_us / static_cast<double>(query_num));

    float recall = 0.0f;
    if (calc_recall_flag) {
      recall = pipeann::calculate_recall(static_cast<uint32_t>(query_num), gt_ids, gt_dists,
                                         static_cast<uint32_t>(gt_dim), query_result_tags[test_id].data(),
                                         static_cast<uint32_t>(recall_at), static_cast<uint32_t>(recall_at));
    }
    const uint64_t process_max_rss_kb = current_hwm_kb();

    if (output) {
      std::cout << std::setw(6) << L << std::setw(12) << force_route << std::setw(12) << qps << std::setw(14)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios << std::setw(14) << mean_candidates << std::setw(14) << mean_route_overhead
                << std::setw(10) << fallback_count << std::setw(10) << prefilter_count << std::setw(10)
                << graph_count << std::setw(10) << empty_count;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall;
      }
      std::cout << std::endl;
    }

    append_jsonl_line(
        structured_output_options,
        "{\"format\":\"pipeann.hybrid.search.v1\","
        "\"index_prefix\":\"" + json_escape(index_prefix_path) + "\"," 
        "\"selector_type\":\"" + json_escape(selector_type) + "\"," 
        "\"query_label_file\":\"" + json_escape(query_label_file) + "\"," 
        "\"query_count\":" + std::to_string(query_num) + ","
        "\"threads\":" + std::to_string(num_threads) + ","
        "\"beamwidth\":" + std::to_string(beamwidth) + ","
        "\"mem_L\":" + std::to_string(mem_L) + ","
        "\"route\":\"" + json_escape(force_route) + "\"," 
        "\"resolved_route\":\"" + json_escape(resolved_route) + "\","
        "\"L\":" + std::to_string(L) + ","
        "\"hybrid_enabled\":" + std::string(index_ptr->hybrid_enabled() ? "true" : "false") + ","
        "\"tau_m\":" + tau_m_json + ","
        "\"threshold_version\":" + threshold_version_json + ","
        "\"min_tau_m\":" + std::to_string(min_threshold) + ","
        "\"max_tau_m\":" + std::to_string(max_threshold) + ","
        "\"min_threshold_version\":" + std::to_string(min_threshold_version) + ","
        "\"max_threshold_version\":" + std::to_string(max_threshold_version) + ","
        "\"qps\":" + std::to_string(qps) + ","
        "\"avg_latency_us\":" + std::to_string(mean_latency) + ","
        "\"p50_latency_us\":" + std::to_string(latency_50) + ","
        "\"p95_latency_us\":" + std::to_string(latency_95) + ","
        "\"p99_latency_us\":" + std::to_string(latency_99) + ","
        "\"p999_latency_us\":" + std::to_string(latency_999) + ","
        "\"mean_hops\":" + std::to_string(mean_hops) + ","
        "\"mean_ios\":" + std::to_string(mean_ios) + ","
        "\"mean_candidate_count\":" + std::to_string(mean_candidates) + ","
        "\"mean_route_overhead_us\":" + std::to_string(mean_route_overhead) + ","
        "\"fallback_count\":" + std::to_string(fallback_count) + ","
        "\"prefilter_count\":" + std::to_string(prefilter_count) + ","
        "\"graph_count\":" + std::to_string(graph_count) + ","
        "\"empty_count\":" + std::to_string(empty_count) + ","
        "\"recall_at\":" + std::to_string(recall_at) + ","
        "\"process_max_rss_kb\":" + std::to_string(process_max_rss_kb) + ","
        "\"max_rss_kb\":" + std::to_string(process_max_rss_kb) + ","
        "\"recall\":" + (calc_recall_flag ? std::to_string(recall) : std::string("null")) + "}");

    delete[] hybrid_stats;
    delete[] stats;

    return true;
  };

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  if (validate_auto_route) {
    std::cout << std::setw(6) << "L" << std::setw(12) << "Queries" << std::setw(12) << "ResultMis"
              << std::setw(14) << "DecisionMis" << std::setw(10) << "Fallback" << std::setw(10)
              << "Prefilt" << std::setw(10) << "Graph" << std::setw(10) << "Empty" << std::endl;
    std::cout << std::string(84, '=') << std::endl;
  } else {
    const std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "L" << std::setw(12) << "Route" << std::setw(12) << "QPS" << std::setw(14)
              << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12)
              << "Mean IOs" << std::setw(14) << "MeanCand" << std::setw(14) << "MeanRouteUs"
              << std::setw(10) << "Fallback" << std::setw(10) << "Prefilt" << std::setw(10) << "Graph"
              << std::setw(10) << "Empty";
    if (calc_recall_flag) {
      std::cout << std::setw(12) << recall_string;
    }
    std::cout << std::endl;
    std::cout << std::string(142, '=') << std::endl;
  }

  bool all_passed = true;
  for (uint32_t test_id = 0; test_id < Lvec.size(); ++test_id) {
    all_passed = run_tests(test_id, true) && all_passed;
  }

  return all_passed ? 0 : -1;
}

int main(int argc, char **argv) {
  if (argc < 16) {
    std::cout << "Usage: " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <query_file.bin>"
              << " <truthset.bin (use \"null\" for none)>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <selector_type (intersect/subset/range)>"
              << " <query_label.spmat>"
              << " <force_route (auto/validate-auto/prefilter/graph)>"
              << " <relaxed_monotonicity_lmax (currently must be 0)>"
              << " <mem_L (0 means no mem index)>"
              << " <L1> [L2] ... [--jsonl-output output.jsonl]"
              << " [--query-vector-csv csv] [--query-label-csv csv] [--single-query-static-rss]" << std::endl;
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
