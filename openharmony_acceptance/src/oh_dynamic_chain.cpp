#include "oh_common.h"

#include <atomic>
#include <future>
#include <memory>

#include "dynamic_index.h"
#include "filter/attribute.h"
#include "filter/selector.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include "utils.h"
#include "utils/picojson.h"

namespace {

struct ManifestRow {
  std::string selector_id;
  std::string selector_type;
  std::string label_config;
};

struct LiveAttrIndexes {
  std::shared_ptr<pipeann::AttrIndex> label;
  std::shared_ptr<pipeann::AttrIndex> range;

  std::map<uint32_t, pipeann::AttrIndex *> as_base_stores() const {
    std::map<uint32_t, pipeann::AttrIndex *> stores;
    if (label) {
      stores[0] = label.get();
    }
    if (range) {
      stores[1] = range.get();
    }
    return stores;
  }
};

struct LoadedSelector {
  std::unique_ptr<pipeann::Selector> selector;
  std::vector<pipeann::Attributes> attrs;
};

struct CheckpointMetrics {
  uint32_t L = 0;
  float recall = -1.0f;
  double avg_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
  double p99_latency_ms = 0.0;
  double avg_ios = 0.0;
  double pre_filter_ratio = 0.0;
  double in_filter_ratio = 0.0;
  double post_filter_ratio = 0.0;
};

template<typename T>
struct StaticCheckpointIndex {
  std::shared_ptr<AlignedFileReader> reader;
  std::unique_ptr<pipeann::AbstractNeighbor<T>> nbr_handler;
  std::unique_ptr<pipeann::SSDIndex<T>> index;

  StaticCheckpointIndex(pipeann::Metric metric, const std::string &nbr_type, uint32_t threads,
                        const std::string &index_prefix) {
    reader.reset(new LinuxAlignedFileReader());
    nbr_handler.reset(pipeann::get_nbr_handler<T>(metric, nbr_type));
    pipeann::IndexBuildParameters params;
    params.max_nthreads = threads;
    index.reset(new pipeann::SSDIndex<T>(metric, reader, nbr_handler.get(), true, &params));
    if (index->load(index_prefix.c_str(), false) != 0) {
      throw std::runtime_error("Failed to load static checkpoint index: " + index_prefix);
    }
  }
};

void wait_for_gt_file(const std::filesystem::path &gt_path) {
  if (gt_path.empty()) {
    return;
  }
  uint32_t waited_seconds = 0;
  while (!std::filesystem::exists(gt_path)) {
    if (waited_seconds == 0 || waited_seconds % 30 == 0) {
      std::cerr << "Waiting for groundtruth: " << gt_path << " waited_s=" << waited_seconds << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ++waited_seconds;
  }
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      fields.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  fields.push_back(cur);
  return fields;
}

std::vector<ManifestRow> load_manifest(const std::filesystem::path &path) {
  std::vector<ManifestRow> rows;
  if (path.empty()) {
    return rows;
  }
  std::ifstream reader(path);
  std::string line;
  std::getline(reader, line);
  while (std::getline(reader, line)) {
    auto fields = split_csv_line(line);
    if (fields.size() < 6) {
      continue;
    }
    rows.push_back({fields[0], fields[1], fields[5]});
  }
  return rows;
}

std::pair<uint32_t, uint32_t> delete_range_for_cycle(uint32_t cycle, uint32_t npoints) {
  uint32_t delete_count = npoints * 6 / 10;
  if (cycle % 2 == 1) {
    return {npoints - delete_count, npoints};
  }
  return {0, delete_count};
}

void copy_attr_family_for_update(const std::string &source, const std::string &target) {
  if (source == target) {
    return;
  }
  for (const std::string &suffix : {std::string(), std::string(".filter"), std::string(".quantize")}) {
    std::filesystem::path src = source + suffix;
    std::filesystem::path dst = target + suffix;
    if (std::filesystem::exists(src)) {
      oh::ensure_parent(dst);
      std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

uint32_t eq_label(size_t i) {
  return static_cast<uint32_t>(i);
}

uint32_t int_a_label(size_t i) {
  return static_cast<uint32_t>(100 + i * 2);
}

uint32_t int_b_label(size_t i) {
  return static_cast<uint32_t>(101 + i * 2);
}

pipeann::Attributes attrs_for_tag(uint32_t tag, const std::vector<uint64_t> &rank) {
  pipeann::Attributes attrs;
  pipeann::Attribute labels;
  auto sels = oh::selectivities();
  uint64_t npoints = rank.size();
  for (size_t s = 0; s < sels.size(); ++s) {
    if (rank[tag] < oh::target_count(npoints, sels[s].second)) {
      labels.push_back(eq_label(s));
      labels.push_back(int_a_label(s));
      labels.push_back(int_b_label(s));
    }
  }
  attrs.set(0, labels);
  attrs.set(1, pipeann::Attribute{static_cast<uint32_t>(rank[tag])});
  return attrs;
}

LoadedSelector load_selector_from_live_indexes(const std::string &config_path, const LiveAttrIndexes &live_indexes) {
  if (config_path.empty() || config_path == "null") {
    return {};
  }

  std::ifstream config_file(config_path);
  if (!config_file) {
    throw std::runtime_error("Failed to open selector config: " + config_path);
  }
  picojson::value config;
  std::string err = picojson::parse(config, config_file);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse selector config " + config_path + ": " + err);
  }

  const auto &root = config.get<picojson::object>();
  auto query_it = root.find("query");
  if (query_it == root.end()) {
    return {};
  }

  LoadedSelector loaded;
  auto base_stores = live_indexes.as_base_stores();
  loaded.selector.reset(pipeann::parse_selector_from_json(query_it->second, base_stores, loaded.attrs));
  return loaded;
}

template<typename T>
struct QueryBundle {
  T *data = nullptr;
  size_t n = 0;
  size_t dim = 0;
  std::unique_ptr<pipeann::Selector> selector;
  std::vector<pipeann::Attributes> attrs;

  ~QueryBundle() {
    delete[] data;
  }
};

template<typename T>
void foreground_search(DynamicIndex<T> &index, QueryBundle<T> &queries, uint32_t k, uint32_t L, uint32_t rounds,
                       uint32_t search_threads, const std::string &phase, uint32_t cycle, std::ofstream &out) {
  if (queries.n == 0 || rounds == 0) {
    return;
  }
  std::vector<double> latencies;
  latencies.reserve(rounds);
  for (uint32_t r = 0; r < rounds; ++r) {
    uint32_t qid = r % static_cast<uint32_t>(queries.n);
    std::vector<uint32_t> one_ids(k);
    std::vector<float> one_dists(k);
    auto t0 = std::chrono::high_resolution_clock::now();
    pipeann::QueryStats stats;
    if (queries.selector) {
      index.search(queries.data + static_cast<size_t>(qid) * queries.dim, k, L, one_ids.data(), one_dists.data(), &stats,
                   queries.selector.get(), &queries.attrs[qid]);
    } else {
      index.search(queries.data + static_cast<size_t>(qid) * queries.dim, k, L, one_ids.data(), one_dists.data(), &stats);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    latencies.push_back(wall_ms);
  }
  auto sorted = oh::sorted_copy(latencies);
  out << "{\"cycle\":" << cycle << ",\"phase\":\"" << phase << "\",\"rounds\":" << rounds
      << ",\"threads\":" << search_threads
      << ",\"avg_latency_ms\":" << oh::mean(latencies) << ",\"p95_latency_ms\":" << oh::percentile(sorted, 0.95)
      << ",\"p99_latency_ms\":" << oh::percentile(sorted, 0.99) << "}\n";
  out.flush();
}

void write_progress(std::ofstream &out, uint32_t cycle, const std::string &phase, uint64_t done, uint64_t total,
                    double elapsed_ms) {
  double percent = total == 0 ? 0.0 : (100.0 * static_cast<double>(done) / static_cast<double>(total));
  double rate = elapsed_ms <= 0.0 ? 0.0 : (1000.0 * static_cast<double>(done) / elapsed_ms);
  out << "{\"cycle\":" << cycle << ",\"phase\":\"" << oh::json_escape(phase) << "\",\"done\":" << done
      << ",\"total\":" << total << ",\"percent\":" << percent << ",\"elapsed_ms\":" << elapsed_ms
      << ",\"rate_per_s\":" << rate << "}\n";
  out.flush();
  std::cerr << "[oh_dynamic_progress] cycle=" << cycle << " phase=" << phase << " done=" << done << "/"
            << total << " percent=" << percent << " elapsed_ms=" << elapsed_ms << " rate_per_s=" << rate
            << std::endl;
}

template<typename T>
CheckpointMetrics checkpoint_search_once_static(pipeann::SSDIndex<T> &index, T *query, size_t query_num,
                                                size_t query_dim, const ManifestRow &row,
                                                const LiveAttrIndexes &live_indexes,
                                                const std::filesystem::path &gt_dir, uint32_t cycle, uint32_t k,
                                                uint32_t L, uint32_t search_threads) {
  LoadedSelector loaded;
  if (row.selector_type != "match_all" && row.label_config != "null" && !row.label_config.empty()) {
    loaded = load_selector_from_live_indexes(row.label_config, live_indexes);
    if (loaded.selector && loaded.attrs.size() != query_num) {
      throw std::runtime_error("Query attr count mismatch for selector " + row.selector_id);
    }
  }

  std::vector<uint32_t> result_tags(query_num * k);
  std::vector<float> result_dists(query_num * k);
  std::vector<pipeann::QueryStats> stats(query_num);
  omp_set_num_threads(search_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
    if (loaded.selector) {
      index.spec_filter_search(query + static_cast<size_t>(i) * query_dim, k, L, loaded.selector.get(),
                               loaded.attrs[static_cast<size_t>(i)],
                               result_tags.data() + static_cast<size_t>(i) * k,
                               result_dists.data() + static_cast<size_t>(i) * k, 32,
                               &stats[static_cast<size_t>(i)]);
    } else {
      index.pipe_search(query + static_cast<size_t>(i) * query_dim, k, 0, L,
                        result_tags.data() + static_cast<size_t>(i) * k,
                        result_dists.data() + static_cast<size_t>(i) * k, 32, &stats[static_cast<size_t>(i)]);
    }
  }

  std::vector<double> latencies(query_num);
  std::vector<double> ios(query_num);
  std::vector<double> pre(query_num), in(query_num), post(query_num);
  for (size_t i = 0; i < query_num; ++i) {
    latencies[i] = stats[i].total_us / 1000.0;
    ios[i] = stats[i].n_ios;
    pre[i] = stats[i].n_filter[pipeann::PRE_FILTER];
    in[i] = stats[i].n_filter[pipeann::IN_FILTER];
    post[i] = stats[i].n_filter[pipeann::POST_FILTER];
  }
  auto sorted = oh::sorted_copy(latencies);

  CheckpointMetrics metrics;
  metrics.L = L;
  metrics.avg_latency_ms = oh::mean(latencies);
  metrics.p95_latency_ms = oh::percentile(sorted, 0.95);
  metrics.p99_latency_ms = oh::percentile(sorted, 0.99);
  metrics.avg_ios = oh::mean(ios);
  metrics.pre_filter_ratio = oh::mean(pre);
  metrics.in_filter_ratio = oh::mean(in);
  metrics.post_filter_ratio = oh::mean(post);
  auto gt_path = gt_dir / ("cycle" + std::to_string(cycle) + "_" + row.selector_id + ".bin");
  if (!gt_dir.empty()) {
    wait_for_gt_file(gt_path);
  }
  if (std::filesystem::exists(gt_path)) {
    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    uint32_t *gt_tags = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    pipeann::load_truthset(gt_path.string(), gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    if (gt_num == query_num) {
      metrics.recall =
          pipeann::calculate_recall(static_cast<uint32_t>(query_num), gt_ids, gt_dists, static_cast<uint32_t>(gt_dim),
                                    result_tags.data(), k, k);
    }
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
  }
  return metrics;
}

template<typename T>
CheckpointMetrics checkpoint_search_once(DynamicIndex<T> &index, T *query, size_t query_num, size_t query_dim,
                                         const ManifestRow &row, const LiveAttrIndexes &live_indexes,
                                         const std::filesystem::path &gt_dir, uint32_t cycle, uint32_t k,
                                         uint32_t L, uint32_t search_threads) {
  LoadedSelector loaded;
  if (row.selector_type != "match_all" && row.label_config != "null" && !row.label_config.empty()) {
    loaded = load_selector_from_live_indexes(row.label_config, live_indexes);
    if (loaded.selector && loaded.attrs.size() != query_num) {
      throw std::runtime_error("Query attr count mismatch for selector " + row.selector_id);
    }
  }

  std::vector<uint32_t> result_tags(query_num * k);
  std::vector<float> result_dists(query_num * k);
  std::vector<pipeann::QueryStats> stats(query_num);
  omp_set_num_threads(search_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
    index.search(query + static_cast<size_t>(i) * query_dim, k, L, result_tags.data() + static_cast<size_t>(i) * k,
                 result_dists.data() + static_cast<size_t>(i) * k, &stats[static_cast<size_t>(i)],
                 loaded.selector.get(), loaded.selector ? &loaded.attrs[static_cast<size_t>(i)] : nullptr);
  }

  std::vector<double> latencies(query_num);
  std::vector<double> ios(query_num);
  std::vector<double> pre(query_num), in(query_num), post(query_num);
  for (size_t i = 0; i < query_num; ++i) {
    latencies[i] = stats[i].total_us / 1000.0;
    ios[i] = stats[i].n_ios;
    pre[i] = stats[i].n_filter[pipeann::PRE_FILTER];
    in[i] = stats[i].n_filter[pipeann::IN_FILTER];
    post[i] = stats[i].n_filter[pipeann::POST_FILTER];
  }
  auto sorted = oh::sorted_copy(latencies);

  CheckpointMetrics metrics;
  metrics.L = L;
  metrics.avg_latency_ms = oh::mean(latencies);
  metrics.p95_latency_ms = oh::percentile(sorted, 0.95);
  metrics.p99_latency_ms = oh::percentile(sorted, 0.99);
  metrics.avg_ios = oh::mean(ios);
  metrics.pre_filter_ratio = oh::mean(pre);
  metrics.in_filter_ratio = oh::mean(in);
  metrics.post_filter_ratio = oh::mean(post);
  auto gt_path = gt_dir / ("cycle" + std::to_string(cycle) + "_" + row.selector_id + ".bin");
  if (!gt_dir.empty()) {
    wait_for_gt_file(gt_path);
  }
  if (std::filesystem::exists(gt_path)) {
    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    uint32_t *gt_tags = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    pipeann::load_truthset(gt_path.string(), gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    if (gt_num == query_num) {
      metrics.recall =
          pipeann::calculate_recall(static_cast<uint32_t>(query_num), gt_ids, gt_dists, static_cast<uint32_t>(gt_dim),
                                    result_tags.data(), k, k);
    }
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
  }
  return metrics;
}

template<typename T>
void checkpoint_search_static(pipeann::SSDIndex<T> &index, T *query, size_t query_num, size_t query_dim,
                              const ManifestRow &row, const LiveAttrIndexes &live_indexes,
                              const std::filesystem::path &gt_dir, uint32_t cycle, uint32_t k,
                              const std::vector<uint32_t> &l_candidates, double recall_min,
                              uint32_t search_threads, std::ofstream &out) {
  std::vector<CheckpointMetrics> sweep;
  sweep.reserve(l_candidates.size());
  CheckpointMetrics selected;
  bool selected_set = false;
  for (uint32_t candidate_l : l_candidates) {
    auto metrics = checkpoint_search_once_static(index, query, query_num, query_dim, row, live_indexes, gt_dir, cycle,
                                                 k, candidate_l, search_threads);
    sweep.push_back(metrics);
    if (metrics.recall >= recall_min) {
      selected = metrics;
      selected_set = true;
      break;
    }
  }
  if (!selected_set) {
    selected = sweep.back();
  }

  out << "{\"cycle\":" << cycle << ",\"selector_id\":\"" << oh::json_escape(row.selector_id)
      << "\",\"selector_type\":\"" << oh::json_escape(row.selector_type)
      << "\",\"checkpoint_mode\":\"static\",\"L\":" << selected.L
      << ",\"selected_L\":" << selected.L << ",\"threads\":" << search_threads
      << ",\"recall_at_10\":" << selected.recall << ",\"avg_latency_ms\":"
      << selected.avg_latency_ms << ",\"p95_latency_ms\":" << selected.p95_latency_ms << ",\"p99_latency_ms\":"
      << selected.p99_latency_ms << ",\"avg_ios\":" << selected.avg_ios << ",\"pre_filter_ratio\":"
      << selected.pre_filter_ratio << ",\"in_filter_ratio\":" << selected.in_filter_ratio
      << ",\"post_filter_ratio\":" << selected.post_filter_ratio << ",\"l_sweep\":[";
  for (size_t i = 0; i < sweep.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "{\"L\":" << sweep[i].L << ",\"recall_at_10\":" << sweep[i].recall << ",\"avg_latency_ms\":"
        << sweep[i].avg_latency_ms << ",\"avg_ios\":" << sweep[i].avg_ios << ",\"pre_filter_ratio\":"
        << sweep[i].pre_filter_ratio << ",\"in_filter_ratio\":" << sweep[i].in_filter_ratio
        << ",\"post_filter_ratio\":" << sweep[i].post_filter_ratio << "}";
  }
  out << "]}\n";
  out.flush();
}

template<typename T>
void checkpoint_search(DynamicIndex<T> &index, T *query, size_t query_num, size_t query_dim, const ManifestRow &row,
                       const LiveAttrIndexes &live_indexes, const std::filesystem::path &gt_dir, uint32_t cycle,
                       uint32_t k, const std::vector<uint32_t> &l_candidates, double recall_min,
                       uint32_t search_threads, std::ofstream &out) {
  std::vector<CheckpointMetrics> sweep;
  sweep.reserve(l_candidates.size());
  CheckpointMetrics selected;
  bool selected_set = false;
  for (uint32_t candidate_l : l_candidates) {
    auto metrics = checkpoint_search_once(index, query, query_num, query_dim, row, live_indexes, gt_dir, cycle, k,
                                          candidate_l, search_threads);
    sweep.push_back(metrics);
    if (metrics.recall >= recall_min) {
      selected = metrics;
      selected_set = true;
      break;
    }
  }
  if (!selected_set) {
    selected = sweep.back();
  }

  out << "{\"cycle\":" << cycle << ",\"selector_id\":\"" << oh::json_escape(row.selector_id)
      << "\",\"selector_type\":\"" << oh::json_escape(row.selector_type)
      << "\",\"checkpoint_mode\":\"dynamic\",\"L\":" << selected.L
      << ",\"selected_L\":" << selected.L << ",\"threads\":" << search_threads
      << ",\"recall_at_10\":" << selected.recall << ",\"avg_latency_ms\":"
      << selected.avg_latency_ms << ",\"p95_latency_ms\":" << selected.p95_latency_ms << ",\"p99_latency_ms\":"
      << selected.p99_latency_ms << ",\"avg_ios\":" << selected.avg_ios << ",\"pre_filter_ratio\":"
      << selected.pre_filter_ratio << ",\"in_filter_ratio\":" << selected.in_filter_ratio
      << ",\"post_filter_ratio\":" << selected.post_filter_ratio << ",\"l_sweep\":[";
  for (size_t i = 0; i < sweep.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "{\"L\":" << sweep[i].L << ",\"recall_at_10\":" << sweep[i].recall << ",\"avg_latency_ms\":"
        << sweep[i].avg_latency_ms << ",\"avg_ios\":" << sweep[i].avg_ios << ",\"pre_filter_ratio\":"
        << sweep[i].pre_filter_ratio << ",\"in_filter_ratio\":" << sweep[i].in_filter_ratio
        << ",\"post_filter_ratio\":" << sweep[i].post_filter_ratio << "}";
  }
  out << "]}\n";
  out.flush();
}

template<typename T>
int run_dynamic(int argc, char **argv) {
  oh::Args args(argc, argv);
  const auto index_prefix = args.get("index-prefix");
  const auto updates_path = std::filesystem::path(args.get("updates"));
  const auto query_path = args.get("query");
  const auto label_config = args.get("label-config", "");
  const std::string label_index = args.get("label-index", index_prefix + ".label.0");
  const std::string range_index = args.get("range-index", index_prefix + ".label.1");
  const auto metric = pipeann::get_metric(args.get("metric", "l2"));
  const std::string nbr_type = args.get("nbr-type", "pq");
  const uint32_t npoints = args.u32("npoints", 1000000);
  const uint32_t cycles = args.u32("cycles", 5);
  const uint32_t k = args.u32("k", 10);
  const uint32_t L = args.u32("L", 100);
  const double recall_min = args.f64("recall-min", 98.0);
  auto l_candidates = oh::parse_u32_list(args.get("L-candidates", ""));
  if (l_candidates.empty()) {
    l_candidates.push_back(L);
  }
  const uint32_t insert_threads = args.u32("insert-threads", std::max(1u, std::thread::hardware_concurrency() / 2));
  const uint32_t search_threads = args.u32("search-threads", std::max(1u, std::thread::hardware_concurrency() / 2));
  const uint32_t merge_threads = args.u32("merge-threads", std::max(1u, std::thread::hardware_concurrency()));
  const uint32_t foreground_rounds = args.u32("foreground-rounds", 32);
  const uint32_t foreground_interval_ms = args.u32("foreground-interval-ms", 1000);
  const bool foreground_enabled = args.get("foreground-enabled", "1") != "0";
  const bool checkpoint_enabled = args.get("checkpoint-enabled", "1") != "0";
  const bool save_after_insert = args.get("save-after-insert", "0") != "0";
  const std::string checkpoint_mode = args.get("checkpoint-mode", "dynamic");
  const auto out_dynamic = std::filesystem::path(args.get("out-jsonl", "results/dynamic_chain.jsonl"));
  const auto out_foreground = std::filesystem::path(args.get("out-foreground-jsonl", "results/foreground_latency.jsonl"));
  const auto out_progress = std::filesystem::path(args.get("out-progress-jsonl", "results/dynamic_progress.jsonl"));
  const auto checkpoint_out =
      std::filesystem::path(args.get("out-checkpoint-jsonl", "results/dynamic_checkpoint_search.jsonl"));
  const auto selector_manifest = std::filesystem::path(args.get("selector-manifest", ""));
  const auto gt_dir = std::filesystem::path(args.get("gt-dir", ""));

  if (index_prefix.empty() || updates_path.empty() || query_path.empty()) {
    throw std::runtime_error("--index-prefix, --updates and --query are required");
  }
  if (checkpoint_enabled && checkpoint_mode == "static" && !save_after_insert) {
    throw std::runtime_error("--checkpoint-mode static requires --save-after-insert 1");
  }
  if (checkpoint_mode != "dynamic" && checkpoint_mode != "static") {
    throw std::runtime_error("Unsupported --checkpoint-mode: " + checkpoint_mode);
  }

  QueryBundle<T> queries;
  pipeann::load_bin<T>(query_path, queries.data, queries.n, queries.dim);

  pipeann::IndexBuildParameters params;
  params.num_threads = insert_threads + search_threads;
  params.max_nthreads = insert_threads + search_threads;
  DynamicIndex<T> index(static_cast<uint32_t>(queries.dim), metric, &params);
  index.load(index_prefix, true);
  index.omp_set_num_threads(search_threads);
  std::unique_ptr<DynamicIndex<T>> foreground_index;
  if (foreground_enabled) {
    foreground_index.reset(new DynamicIndex<T>(static_cast<uint32_t>(queries.dim), metric, &params));
    foreground_index->load(index_prefix, false);
    foreground_index->omp_set_num_threads(search_threads);
  }
  const std::string update_label_index = index.index_prefix() + ".label.0";
  const std::string update_range_index = index.index_prefix() + ".label.1";
  copy_attr_family_for_update(label_index, update_label_index);
  copy_attr_family_for_update(range_index, update_range_index);

  LiveAttrIndexes live_indexes;
  live_indexes.label = index.load_attr_index_from_file(0, update_label_index, "label");
  live_indexes.range = index.load_attr_index_from_file(1, update_range_index, "range");
  LiveAttrIndexes foreground_indexes;
  if (foreground_enabled) {
    foreground_indexes.label = load_attr_index_from_file(label_index, "label", npoints);
    foreground_indexes.range = load_attr_index_from_file(range_index, "range", npoints);
  }
  if (foreground_enabled && !label_config.empty() && label_config != "null") {
    auto loaded = load_selector_from_live_indexes(label_config, foreground_indexes);
    queries.selector = std::move(loaded.selector);
    queries.attrs = std::move(loaded.attrs);
    if (queries.selector && queries.attrs.size() != queries.n) {
      throw std::runtime_error("Foreground query attr count mismatch");
    }
  }

  auto rank = oh::stable_ranks(npoints);
  oh::ensure_parent(out_dynamic);
  oh::ensure_parent(out_progress);
  std::ofstream dyn(out_dynamic, std::ios::app);
  std::ofstream fg;
  if (foreground_enabled) {
    oh::ensure_parent(out_foreground);
    fg.open(out_foreground, std::ios::app);
  }
  std::ofstream progress(out_progress, std::ios::app);
  std::ofstream checkpoint;
  if (checkpoint_enabled) {
    oh::ensure_parent(checkpoint_out);
    checkpoint.open(checkpoint_out, std::ios::app);
  }
  auto manifest_rows = checkpoint_enabled ? load_manifest(selector_manifest) : std::vector<ManifestRow>();

  const uint64_t update_rows_per_cycle = static_cast<uint64_t>(npoints) * 6 / 10;
  for (uint32_t cycle = 1; cycle <= cycles; ++cycle) {
    auto [begin, end] = delete_range_for_cycle(cycle, npoints);
    uint32_t count = end - begin;
    std::vector<uint32_t> tags(count);
    std::iota(tags.begin(), tags.end(), begin);

    auto t0 = std::chrono::high_resolution_clock::now();
    write_progress(progress, cycle, "mark_delete_start", 0, count, 0.0);
    index.remove(tags.data(), static_cast<uint32_t>(tags.size()));
    auto t1 = std::chrono::high_resolution_clock::now();
    double delete_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    write_progress(progress, cycle, "mark_delete_done", count, count, delete_ms);
    if (foreground_enabled) {
      foreground_search(*foreground_index, queries, k, L, foreground_rounds, search_threads, "after_mark_delete", cycle, fg);
    }

    write_progress(progress, cycle, "merge_start", 0, 0, 0.0);
    auto merge_start = std::chrono::high_resolution_clock::now();
    auto merge_future = std::async(std::launch::async, [&]() { index.save(index.index_prefix(), merge_threads); });
    while (merge_future.wait_for(std::chrono::milliseconds(foreground_interval_ms)) != std::future_status::ready) {
      auto now = std::chrono::high_resolution_clock::now();
      write_progress(progress, cycle, "merge_running", 0, 0,
                     std::chrono::duration<double, std::milli>(now - merge_start).count());
      if (foreground_enabled) {
        foreground_search(*foreground_index, queries, k, L, foreground_rounds, search_threads, "merge", cycle, fg);
      }
    }
    merge_future.get();
    auto t2 = std::chrono::high_resolution_clock::now();
    double merge_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    write_progress(progress, cycle, "merge_done", 0, 0, merge_ms);

    uint32_t update_dim = 0;
    auto update_data =
        oh::read_bin_rows<T>(updates_path, static_cast<uint64_t>(cycle - 1) * update_rows_per_cycle, count, update_dim);
    if (update_dim != queries.dim) {
      throw std::runtime_error("Update vector dimension mismatch");
    }
    std::vector<pipeann::Attributes> insert_attrs(count);
    for (uint32_t i = 0; i < count; ++i) {
      insert_attrs[i] = attrs_for_tag(begin + i, rank);
    }
    std::atomic<uint64_t> inserted_count{0};
    auto insert_start = std::chrono::high_resolution_clock::now();
    write_progress(progress, cycle, "insert_start", 0, count, 0.0);
    auto insert_future = std::async(std::launch::async, [&]() {
#pragma omp parallel for schedule(dynamic) num_threads(insert_threads)
      for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        index.insert(update_data.data() + static_cast<size_t>(i) * update_dim, begin + static_cast<uint32_t>(i),
                     &insert_attrs[static_cast<size_t>(i)]);
        inserted_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
    while (insert_future.wait_for(std::chrono::milliseconds(foreground_interval_ms)) != std::future_status::ready) {
      auto now = std::chrono::high_resolution_clock::now();
      write_progress(progress, cycle, "insert_running", inserted_count.load(std::memory_order_relaxed), count,
                     std::chrono::duration<double, std::milli>(now - insert_start).count());
      if (foreground_enabled) {
        foreground_search(*foreground_index, queries, k, L, foreground_rounds, search_threads, "insert", cycle, fg);
      }
    }
    insert_future.get();
    auto t3 = std::chrono::high_resolution_clock::now();
    double insert_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    write_progress(progress, cycle, "insert_done", count, count, insert_ms);

    double post_insert_save_ms = 0.0;
    if (save_after_insert) {
      write_progress(progress, cycle, "post_insert_save_start", 0, 0, 0.0);
      auto save_start = std::chrono::high_resolution_clock::now();
      index.save(index.index_prefix(), merge_threads);
      auto save_done = std::chrono::high_resolution_clock::now();
      post_insert_save_ms = std::chrono::duration<double, std::milli>(save_done - save_start).count();
      write_progress(progress, cycle, "post_insert_save_done", 0, 0, post_insert_save_ms);
    }

    if (foreground_enabled) {
      foreground_search(*foreground_index, queries, k, L, foreground_rounds, search_threads, "after_insert", cycle, fg);
    }
    if (checkpoint_enabled) {
      std::unique_ptr<StaticCheckpointIndex<T>> static_checkpoint;
      if (checkpoint_mode == "static") {
        static_checkpoint.reset(new StaticCheckpointIndex<T>(metric, nbr_type, search_threads, index.index_prefix()));
      }
      for (const auto &row : manifest_rows) {
        if (checkpoint_mode == "static") {
          checkpoint_search_static(*static_checkpoint->index, queries.data, queries.n, queries.dim, row, live_indexes,
                                   gt_dir, cycle, k, l_candidates, recall_min, search_threads, checkpoint);
        } else {
          checkpoint_search(index, queries.data, queries.n, queries.dim, row, live_indexes, gt_dir, cycle, k,
                            l_candidates, recall_min, search_threads, checkpoint);
        }
      }
    }

    dyn << "{\"cycle\":" << cycle << ",\"delete_begin\":" << begin << ",\"delete_end\":" << end
        << ",\"deleted_count\":" << count << ",\"delete_ms\":" << delete_ms
        << ",\"delete_ms_per_vector\":" << (delete_ms / static_cast<double>(count)) << ",\"merge_ms\":" << merge_ms
        << ",\"insert_ms\":" << insert_ms << ",\"post_insert_save_ms\":" << post_insert_save_ms
        << ",\"search_threads\":" << search_threads << ",\"live_count\":" << npoints << "}\n";
    dyn.flush();
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    std::string type = args.get("type", "float");
    if (type == "float") {
      return run_dynamic<float>(argc, argv);
    }
    if (type == "uint8") {
      return run_dynamic<uint8_t>(argc, argv);
    }
    if (type == "int8") {
      return run_dynamic<int8_t>(argc, argv);
    }
    throw std::runtime_error("Unsupported --type: " + type);
  } catch (const std::exception &e) {
    std::cerr << "oh_dynamic_chain failed: " << e.what() << std::endl;
    return 1;
  }
}
