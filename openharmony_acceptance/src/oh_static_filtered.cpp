#include "oh_common.h"

#include <omp.h>

#include "filter/attribute.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include "utils.h"

namespace {

template<typename T>
int run_static(int argc, char **argv) {
  oh::Args args(argc, argv);
  const std::string index_prefix = args.get("index-prefix");
  const std::string query_path = args.get("query");
  const std::string gt_path = args.get("gt", "null");
  const std::string label_config = args.get("label-config", "");
  const std::string selector_id = args.get("selector-id", "unknown");
  const std::string metric_name = args.get("metric", "l2");
  const std::string nbr_type = args.get("nbr-type", "pq");
  const uint32_t threads = args.u32("threads", std::max(1u, std::thread::hardware_concurrency()));
  const uint32_t beamwidth = args.u32("beamwidth", 32);
  const uint32_t mem_l = args.u32("mem-L", 0);
  const uint64_t k = args.u64("k", 10);
  const uint64_t l_search = args.u64("L", 100);
  const double recall_min = args.f64("recall-min", 98.0);
  auto l_candidates = oh::parse_u32_list(args.get("L-candidates", ""));
  if (l_candidates.empty()) {
    l_candidates.push_back(static_cast<uint32_t>(l_search));
  }
  const auto out_jsonl = std::filesystem::path(args.get("out-jsonl", "results/static_filtered.jsonl"));

  if (index_prefix.empty() || query_path.empty()) {
    throw std::runtime_error("--index-prefix and --query are required");
  }

  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  pipeann::load_bin<T>(query_path, query, query_num, query_dim);

  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *gt_tags = nullptr;
  size_t gt_num = 0, gt_dim = 0;
  bool calc_recall = false;
  if (gt_path != "null" && file_exists(gt_path)) {
    pipeann::load_truthset(gt_path, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    calc_recall = gt_num == query_num;
  }

  auto metric = pipeann::get_metric(metric_name);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  pipeann::IndexBuildParameters params;
  params.max_nthreads = threads;
  pipeann::SSDIndex<T> index(metric, reader, nbr_handler, true, &params);
  if (index.load(index_prefix.c_str(), false) != 0) {
    throw std::runtime_error("Failed to load index: " + index_prefix);
  }

  std::map<uint32_t, pipeann::AttrIndex *> base_stores;
  pipeann::Selector *selector = nullptr;
  std::vector<pipeann::Attributes> query_attrs;
  if (!label_config.empty() && label_config != "null") {
    base_stores = pipeann::load_base_attr_from_config(label_config, index.meta_.npoints);
    auto ret = pipeann::load_selector_from_config(label_config, base_stores);
    selector = ret.first;
    query_attrs = std::move(ret.second);
  }

  struct SearchMetrics {
    uint32_t L = 0;
    float recall = -1.0f;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double avg_ios = 0.0;
    double pre_filter_ratio = 0.0;
    double in_filter_ratio = 0.0;
    double post_filter_ratio = 0.0;
  };

  auto run_one_l = [&](uint32_t current_l) {
    std::vector<uint32_t> result_tags(query_num * k);
    std::vector<float> result_dists(query_num * k);
    std::vector<pipeann::QueryStats> stats(query_num);
    omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
      if (selector != nullptr) {
        index.spec_filter_search(query + i * query_dim, k, current_l, selector, query_attrs[i],
                                 result_tags.data() + i * k, result_dists.data() + i * k, beamwidth,
                                 stats.data() + i);
      } else {
        index.pipe_search(query + i * query_dim, k, mem_l, current_l, result_tags.data() + i * k,
                          result_dists.data() + i * k, beamwidth, stats.data() + i);
      }
    }

    std::vector<double> latency_ms(query_num);
    std::vector<double> ios(query_num);
    std::vector<double> pre(query_num), in(query_num), post(query_num);
    for (size_t i = 0; i < query_num; ++i) {
      latency_ms[i] = stats[i].total_us / 1000.0;
      ios[i] = stats[i].n_ios;
      pre[i] = stats[i].n_filter[pipeann::PRE_FILTER];
      in[i] = stats[i].n_filter[pipeann::IN_FILTER];
      post[i] = stats[i].n_filter[pipeann::POST_FILTER];
    }
    auto sorted = oh::sorted_copy(latency_ms);
    SearchMetrics metrics;
    metrics.L = current_l;
    metrics.avg_latency_ms = oh::mean(latency_ms);
    metrics.p50_latency_ms = oh::percentile(sorted, 0.50);
    metrics.p95_latency_ms = oh::percentile(sorted, 0.95);
    metrics.p99_latency_ms = oh::percentile(sorted, 0.99);
    metrics.avg_ios = oh::mean(ios);
    metrics.pre_filter_ratio = oh::mean(pre);
    metrics.in_filter_ratio = oh::mean(in);
    metrics.post_filter_ratio = oh::mean(post);
    if (calc_recall) {
      metrics.recall = pipeann::calculate_recall(static_cast<uint32_t>(query_num), gt_ids, gt_dists,
                                                 static_cast<uint32_t>(gt_dim), result_tags.data(),
                                                 static_cast<uint32_t>(k), static_cast<uint32_t>(k));
    }
    return metrics;
  };

  std::vector<SearchMetrics> sweep;
  sweep.reserve(l_candidates.size());
  SearchMetrics selected;
  bool selected_set = false;
  for (uint32_t candidate_l : l_candidates) {
    auto metrics = run_one_l(candidate_l);
    sweep.push_back(metrics);
    if (!calc_recall || metrics.recall >= recall_min) {
      selected = metrics;
      selected_set = true;
      break;
    }
  }
  if (!selected_set) {
    selected = sweep.back();
  }

  oh::ensure_parent(out_jsonl);
  std::ofstream out(out_jsonl, std::ios::app);
  out << "{\"selector_id\":\"" << oh::json_escape(selector_id) << "\",\"query_count\":" << query_num
      << ",\"k\":" << k << ",\"L\":" << selected.L << ",\"selected_L\":" << selected.L << ",\"threads\":"
      << threads << ",\"recall_at_10\":" << selected.recall << ",\"avg_latency_ms\":" << selected.avg_latency_ms
      << ",\"p50_latency_ms\":" << selected.p50_latency_ms << ",\"p95_latency_ms\":" << selected.p95_latency_ms
      << ",\"p99_latency_ms\":" << selected.p99_latency_ms << ",\"avg_ios\":" << selected.avg_ios
      << ",\"pre_filter_ratio\":" << selected.pre_filter_ratio << ",\"in_filter_ratio\":"
      << selected.in_filter_ratio << ",\"post_filter_ratio\":" << selected.post_filter_ratio
      << ",\"l_sweep\":[";
  for (size_t i = 0; i < sweep.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "{\"L\":" << sweep[i].L << ",\"recall_at_10\":" << sweep[i].recall << ",\"avg_latency_ms\":"
        << sweep[i].avg_latency_ms << "}";
  }
  out << "]}\n";

  delete selector;
  for (auto &[_, store] : base_stores) {
    delete store;
  }
  delete[] query;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    std::string type = args.get("type", "float");
    if (type == "float") {
      return run_static<float>(argc, argv);
    }
    if (type == "uint8") {
      return run_static<uint8_t>(argc, argv);
    }
    if (type == "int8") {
      return run_static<int8_t>(argc, argv);
    }
    throw std::runtime_error("Unsupported --type: " + type);
  } catch (const std::exception &e) {
    std::cerr << "oh_static_filtered failed: " << e.what() << std::endl;
    return 1;
  }
}
