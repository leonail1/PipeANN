#include "oh_common.h"

#include <atomic>
#include <cstdlib>
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

std::string shell_quote(const std::string &value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

void run_shell_command(const std::string &command) {
  std::cerr << "[oh_zero_start] running: " << command << std::endl;
  int rc = std::system(command.c_str());
  if (rc != 0) {
    throw std::runtime_error("Command failed with rc=" + std::to_string(rc) + ": " + command);
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

void write_attrs_for_tag_prefix(const std::filesystem::path &label_spmat, const std::filesystem::path &range_bin,
                                uint32_t tag_count, const std::vector<uint64_t> &rank) {
  const int64_t n_label_cols = 256;
  const size_t max_label_values = oh::selectivities().size() * 3;
  const int32_t padding_label_base = 200;
  std::vector<std::vector<int32_t>> label_rows(tag_count);
  std::vector<std::vector<float>> label_vals(tag_count);
  std::vector<uint32_t> range_values(tag_count);
  auto sels = oh::selectivities();
  for (uint32_t tag = 0; tag < tag_count; ++tag) {
    range_values[tag] = static_cast<uint32_t>(rank[tag]);
    for (size_t s = 0; s < sels.size(); ++s) {
      if (rank[tag] < oh::target_count(rank.size(), sels[s].second)) {
        label_rows[tag].push_back(static_cast<int32_t>(eq_label(s)));
        label_vals[tag].push_back(1.0f);
        label_rows[tag].push_back(static_cast<int32_t>(int_a_label(s)));
        label_vals[tag].push_back(1.0f);
        label_rows[tag].push_back(static_cast<int32_t>(int_b_label(s)));
        label_vals[tag].push_back(1.0f);
      }
    }
  }
  if (tag_count > 0) {
    while (label_rows[0].size() < max_label_values) {
      label_rows[0].push_back(padding_label_base + static_cast<int32_t>(label_rows[0].size()));
      label_vals[0].push_back(1.0f);
    }
  }
  oh::write_spmat(label_spmat, static_cast<int64_t>(tag_count), n_label_cols, label_rows, label_vals);
  oh::write_bin_matrix<uint32_t>(range_bin, range_values, tag_count, 1);
}

double selectivity_for_selector_id(const std::string &selector_id) {
  if (selector_id.rfind("match_all", 0) == 0) {
    return 1.0;
  }
  auto pos = selector_id.rfind("_s");
  if (pos == std::string::npos) {
    return 1.0;
  }
  const std::string suffix = selector_id.substr(pos + 1);
  for (const auto &entry : oh::selectivities()) {
    if (entry.first == suffix) {
      return entry.second;
    }
  }
  return 1.0;
}

bool zero_tag_matches_selector(uint32_t tag, const std::string &selector_id, const std::vector<uint64_t> &rank) {
  if (selector_id.empty() || selector_id.rfind("match_all", 0) == 0) {
    return true;
  }
  const uint32_t count = oh::target_count(rank.size(), selectivity_for_selector_id(selector_id));
  return rank[tag] < count;
}

template<typename T>
double l2_distance(const T *a, const T *b, uint32_t dim) {
  double dist = 0.0;
  for (uint32_t d = 0; d < dim; ++d) {
    const double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
    dist += diff * diff;
  }
  return dist;
}

void write_progress(std::ofstream &out, uint32_t cycle, const std::string &phase, uint64_t done, uint64_t total,
                    double elapsed_ms);

template<typename T>
void build_zero_start_bootstrap(const std::string &type, const std::filesystem::path &base_path,
                                const std::filesystem::path &zero_work_dir, const std::string &index_prefix,
                                uint32_t bootstrap_npoints, uint32_t final_npoints,
                                const std::vector<uint64_t> &rank, const std::string &build_binary, uint32_t r,
                                uint32_t r_dense, uint32_t build_l, uint32_t pq_bytes, uint32_t mem_gb,
                                uint32_t build_threads, const std::string &metric) {
  if (bootstrap_npoints == 0 || bootstrap_npoints >= final_npoints) {
    throw std::runtime_error("--bootstrap-npoints must be in [1, npoints)");
  }
  std::filesystem::create_directories(zero_work_dir);
  uint32_t base_dim = 0;
  auto bootstrap_data = oh::read_bin_rows<T>(base_path, 0, bootstrap_npoints, base_dim);
  auto bootstrap_bin = zero_work_dir / "zero_bootstrap_10k.bin";
  auto bootstrap_label = zero_work_dir / "zero_bootstrap_10k_labels.spmat";
  auto bootstrap_range = zero_work_dir / "zero_bootstrap_10k_range.bin";
  oh::write_bin_matrix<T>(bootstrap_bin, bootstrap_data, bootstrap_npoints, base_dim);
  write_attrs_for_tag_prefix(bootstrap_label, bootstrap_range, bootstrap_npoints, rank);

  std::ostringstream cmd;
  cmd << shell_quote(build_binary) << " " << shell_quote(type) << " " << shell_quote(bootstrap_bin.string()) << " "
      << shell_quote(index_prefix) << " " << r << " " << r_dense << " " << build_l << " " << pq_bytes << " "
      << mem_gb << " " << build_threads << " " << shell_quote(metric) << " pq"
      << " label_spmat " << shell_quote(bootstrap_label.string()) << " range " << shell_quote(bootstrap_range.string());
  run_shell_command(cmd.str());
}

template<typename T>
void insert_base_tail_to_native_index(DynamicIndex<T> &index, const std::filesystem::path &base_path,
                                      uint32_t start_tag, uint32_t final_npoints, uint32_t expected_dim,
                                      const std::vector<uint64_t> &rank, uint32_t insert_threads,
                                      std::ofstream &progress) {
  const uint32_t total = final_npoints - start_tag;
  const uint32_t chunk_rows = 100000;
  uint64_t inserted_total = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  write_progress(progress, 0, "zero_native_insert_start", 0, total, 0.0);
  for (uint32_t offset = start_tag; offset < final_npoints; offset += chunk_rows) {
    uint32_t count = std::min<uint32_t>(chunk_rows, final_npoints - offset);
    uint32_t dim = 0;
    auto data = oh::read_bin_rows<T>(base_path, offset, count, dim);
    if (dim != expected_dim) {
      throw std::runtime_error("Zero-start base vector dimension mismatch");
    }
    std::vector<pipeann::Attributes> attrs(count);
    for (uint32_t i = 0; i < count; ++i) {
      attrs[i] = attrs_for_tag(offset + i, rank);
    }
#pragma omp parallel for schedule(dynamic) num_threads(insert_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
      index.insert(data.data() + static_cast<size_t>(i) * dim, offset + static_cast<uint32_t>(i),
                   &attrs[static_cast<size_t>(i)]);
    }
    inserted_total += count;
    auto now = std::chrono::high_resolution_clock::now();
    write_progress(progress, 0, "zero_native_insert_running", inserted_total, total,
                   std::chrono::duration<double, std::milli>(now - start_time).count());
  }
  auto done = std::chrono::high_resolution_clock::now();
  write_progress(progress, 0, "zero_native_insert_done", total, total,
                 std::chrono::duration<double, std::milli>(done - start_time).count());
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

void copy_optional_snapshot_file(const std::string &source, const std::string &target) {
  if (std::filesystem::exists(source)) {
    oh::ensure_parent(target);
    std::filesystem::copy_file(source, target, std::filesystem::copy_options::overwrite_existing);
  } else {
    std::filesystem::remove(target);
  }
}

const std::vector<std::string> &snapshot_suffixes() {
  static const std::vector<std::string> suffixes = {
      "_disk.index",        "_disk.index.tags", "_pq_compressed.bin", "_pq_pivots.bin",
      "_partition.bin.aligned", ".label.0",         ".label.0.filter",   ".label.0.quantize",
      ".label.1",          ".label.1.filter",   ".label.1.quantize"};
  return suffixes;
}

void remove_index_family_snapshot(const std::string &snapshot_prefix) {
  for (const auto &suffix : snapshot_suffixes()) {
    std::filesystem::remove(snapshot_prefix + suffix);
  }
}

void copy_index_family_for_snapshot(const std::string &source_prefix, const std::string &snapshot_prefix) {
  for (const auto &suffix : snapshot_suffixes()) {
    copy_optional_snapshot_file(source_prefix + suffix, snapshot_prefix + suffix);
  }
}

template<typename T>
void reload_foreground_snapshot(std::unique_ptr<StaticCheckpointIndex<T>> &foreground_snapshot,
                                LiveAttrIndexes &foreground_indexes, QueryBundle<T> &queries, pipeann::Metric metric,
                                const std::string &nbr_type, const std::string &source_prefix,
                                const std::string &snapshot_base_prefix, uint32_t &snapshot_version,
                                const std::string &label_config, uint32_t search_threads) {
  if (nbr_type != "pq") {
    throw std::runtime_error("Foreground snapshot currently supports only --nbr-type pq");
  }
  if (snapshot_version > 1) {
    remove_index_family_snapshot(snapshot_base_prefix + "_v" + std::to_string(snapshot_version - 2));
  }
  const std::string snapshot_prefix = snapshot_base_prefix + "_v" + std::to_string(snapshot_version++);
  copy_index_family_for_snapshot(source_prefix, snapshot_prefix);

  auto next_snapshot = std::make_unique<StaticCheckpointIndex<T>>(metric, nbr_type, search_threads, snapshot_prefix);
  const uint32_t snapshot_npoints = static_cast<uint32_t>(next_snapshot->index->meta_.npoints);

  foreground_indexes.label = load_attr_index_from_file(snapshot_prefix + ".label.0", "label", snapshot_npoints);
  foreground_indexes.range = load_attr_index_from_file(snapshot_prefix + ".label.1", "range", snapshot_npoints);

  queries.selector.reset();
  queries.attrs.clear();
  if (!label_config.empty() && label_config != "null") {
    auto loaded = load_selector_from_live_indexes(label_config, foreground_indexes);
    queries.selector = std::move(loaded.selector);
    queries.attrs = std::move(loaded.attrs);
    if (queries.selector && queries.attrs.size() != queries.n) {
      throw std::runtime_error("Foreground query attr count mismatch");
    }
  }
  foreground_snapshot = std::move(next_snapshot);
}

template<typename T>
void zero_start_exact_probe(const std::filesystem::path &base_path, QueryBundle<T> &queries, uint32_t bootstrap_npoints,
                            const std::string &selector_id, const std::vector<uint64_t> &rank, uint32_t k,
                            uint32_t rounds, uint32_t search_threads, std::ofstream &out) {
  if (!out || queries.n == 0 || rounds == 0) {
    return;
  }
  uint32_t dim = 0;
  auto data = oh::read_bin_rows<T>(base_path, 0, bootstrap_npoints, dim);
  if (dim != queries.dim) {
    throw std::runtime_error("Zero-start exact probe dimension mismatch");
  }
  std::vector<uint32_t> candidates;
  candidates.reserve(bootstrap_npoints);
  for (uint32_t tag = 0; tag < bootstrap_npoints; ++tag) {
    if (zero_tag_matches_selector(tag, selector_id, rank)) {
      candidates.push_back(tag);
    }
  }
  std::vector<double> latencies;
  latencies.reserve(rounds);
  for (uint32_t r = 0; r < rounds; ++r) {
    uint32_t qid = r % static_cast<uint32_t>(queries.n);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, uint32_t>> heap;
    heap.reserve(candidates.size());
    const T *query = queries.data + static_cast<size_t>(qid) * dim;
    for (uint32_t tag : candidates) {
      const T *vec = data.data() + static_cast<size_t>(tag) * dim;
      heap.emplace_back(l2_distance(query, vec, dim), tag);
    }
    const size_t keep = std::min<size_t>(k, heap.size());
    if (keep < heap.size()) {
      std::nth_element(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(keep), heap.end());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  auto sorted = oh::sorted_copy(latencies);
  out << "{\"cycle\":0,\"phase\":\"zero_start_exact\",\"rounds\":" << rounds << ",\"threads\":"
      << search_threads << ",\"selector_id\":\"" << oh::json_escape(selector_id)
      << "\",\"candidate_count\":" << candidates.size() << ",\"avg_latency_ms\":" << oh::mean(latencies)
      << ",\"p95_latency_ms\":" << oh::percentile(sorted, 0.95) << ",\"p99_latency_ms\":"
      << oh::percentile(sorted, 0.99) << "}\n";
  out.flush();
}

template<typename T>
void foreground_search(pipeann::SSDIndex<T> &index, QueryBundle<T> &queries, uint32_t k, uint32_t L, uint32_t rounds,
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
      index.spec_filter_search(queries.data + static_cast<size_t>(qid) * queries.dim, k, L, queries.selector.get(),
                               queries.attrs[qid], one_ids.data(), one_dists.data(), 32, &stats);
    } else {
      index.pipe_search(queries.data + static_cast<size_t>(qid) * queries.dim, k, 0, L, one_ids.data(),
                        one_dists.data(), 32, &stats);
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
  const std::string type = args.get("type", "float");
  const auto index_prefix = args.get("index-prefix");
  const auto updates_path = std::filesystem::path(args.get("updates"));
  const auto base_path = std::filesystem::path(args.get("base", ""));
  const auto query_path = args.get("query");
  const auto label_config = args.get("label-config", "");
  const std::string label_index = args.get("label-index", index_prefix + ".label.0");
  const std::string range_index = args.get("range-index", index_prefix + ".label.1");
  const auto metric = pipeann::get_metric(args.get("metric", "l2"));
  const std::string metric_name = args.get("metric", "l2");
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
  const bool start_from_zero = args.get("start-from-zero", "0") != "0";
  const uint32_t bootstrap_npoints = args.u32("bootstrap-npoints", 10000);
  const auto zero_work_dir =
      std::filesystem::path(args.get("zero-work-dir", (std::filesystem::path(index_prefix).parent_path() /
                                                       "zero_start_bootstrap").string()));
  const std::string build_binary = args.get("build-binary", "build/tests/build_disk_index_filtered");
  const uint32_t build_r = args.u32("build-R", 96);
  const uint32_t build_r_dense = args.u32("build-R-dense", 0);
  const uint32_t build_l = args.u32("build-L", 128);
  const uint32_t build_pq_bytes = args.u32("build-PQ-bytes", 32);
  const uint32_t build_mem_gb = args.u32("build-mem-gb", 64);
  const uint32_t build_threads = args.u32("build-threads", std::max(1u, std::thread::hardware_concurrency()));
  const std::string zero_probe_selector_id = args.get("zero-probe-selector-id", "");
  const std::string checkpoint_mode = args.get("checkpoint-mode", "dynamic");
  const auto out_dynamic = std::filesystem::path(args.get("out-jsonl", "results/dynamic_chain.jsonl"));
  const auto out_foreground = std::filesystem::path(args.get("out-foreground-jsonl", "results/foreground_latency.jsonl"));
  const auto out_progress = std::filesystem::path(args.get("out-progress-jsonl", "results/dynamic_progress.jsonl"));
  const auto out_zero =
      std::filesystem::path(args.get("out-zero-jsonl", "results/zero_start_exact.jsonl"));
  const auto checkpoint_out =
      std::filesystem::path(args.get("out-checkpoint-jsonl", "results/dynamic_checkpoint_search.jsonl"));
  const auto selector_manifest = std::filesystem::path(args.get("selector-manifest", ""));
  const auto gt_dir = std::filesystem::path(args.get("gt-dir", ""));

  if (index_prefix.empty() || updates_path.empty() || query_path.empty()) {
    throw std::runtime_error("--index-prefix, --updates and --query are required");
  }
  if (start_from_zero && base_path.empty()) {
    throw std::runtime_error("--base is required when --start-from-zero 1");
  }
  if (checkpoint_enabled && checkpoint_mode == "static" && !save_after_insert) {
    throw std::runtime_error("--checkpoint-mode static requires --save-after-insert 1");
  }
  if (checkpoint_mode != "dynamic" && checkpoint_mode != "static") {
    throw std::runtime_error("Unsupported --checkpoint-mode: " + checkpoint_mode);
  }

  QueryBundle<T> queries;
  pipeann::load_bin<T>(query_path, queries.data, queries.n, queries.dim);
  auto rank = oh::stable_ranks(npoints);

  oh::ensure_parent(out_dynamic);
  oh::ensure_parent(out_progress);
  std::ofstream dyn(out_dynamic, std::ios::app);
  std::ofstream fg;
  if (foreground_enabled) {
    oh::ensure_parent(out_foreground);
    fg.open(out_foreground, std::ios::app);
  }
  std::ofstream zero_out;
  if (start_from_zero) {
    oh::ensure_parent(out_zero);
    zero_out.open(out_zero, std::ios::app);
  }
  std::ofstream progress(out_progress, std::ios::app);
  std::ofstream checkpoint;
  if (checkpoint_enabled) {
    oh::ensure_parent(checkpoint_out);
    checkpoint.open(checkpoint_out, std::ios::app);
  }
  auto manifest_rows = checkpoint_enabled ? load_manifest(selector_manifest) : std::vector<ManifestRow>();

  if (start_from_zero) {
    write_progress(progress, 0, "zero_memory_insert_start", 0, bootstrap_npoints, 0.0);
    auto zero_t0 = std::chrono::high_resolution_clock::now();
    if (foreground_enabled && !zero_probe_selector_id.empty()) {
      if (metric_name != "l2") {
        throw std::runtime_error("zero_start_exact_probe currently supports only l2 metric");
      }
      zero_start_exact_probe(base_path, queries, bootstrap_npoints, zero_probe_selector_id, rank, k,
                             foreground_rounds, search_threads, zero_out);
    }
    build_zero_start_bootstrap<T>(type, base_path, zero_work_dir, index_prefix, bootstrap_npoints, npoints, rank,
                                  build_binary, build_r, build_r_dense, build_l, build_pq_bytes, build_mem_gb,
                                  build_threads, metric_name);
    auto zero_t1 = std::chrono::high_resolution_clock::now();
    write_progress(progress, 0, "zero_memory_insert_done", bootstrap_npoints, bootstrap_npoints,
                   std::chrono::duration<double, std::milli>(zero_t1 - zero_t0).count());
  }

  pipeann::IndexBuildParameters params;
  params.num_threads = insert_threads + search_threads;
  params.max_nthreads = insert_threads + search_threads;
  DynamicIndex<T> index(static_cast<uint32_t>(queries.dim), metric, &params);
  index.load(index_prefix, true);
  index.omp_set_num_threads(search_threads);
  const std::string update_label_index = index.index_prefix() + ".label.0";
  const std::string update_range_index = index.index_prefix() + ".label.1";
  const std::string source_label_index = start_from_zero ? index_prefix + ".label.0" : label_index;
  const std::string source_range_index = start_from_zero ? index_prefix + ".label.1" : range_index;
  copy_attr_family_for_update(source_label_index, update_label_index);
  copy_attr_family_for_update(source_range_index, update_range_index);

  LiveAttrIndexes live_indexes;
  live_indexes.label = index.load_attr_index_from_file(0, update_label_index, "label");
  live_indexes.range = index.load_attr_index_from_file(1, update_range_index, "range");

  if (start_from_zero) {
    insert_base_tail_to_native_index(index, base_path, bootstrap_npoints, npoints, static_cast<uint32_t>(queries.dim),
                                     rank, insert_threads, progress);
    write_progress(progress, 0, "zero_initial_save_start", 0, 0, 0.0);
    auto save_start = std::chrono::high_resolution_clock::now();
    index.save(index.index_prefix(), merge_threads);
    auto save_done = std::chrono::high_resolution_clock::now();
    write_progress(progress, 0, "zero_initial_save_done", 0, 0,
                   std::chrono::duration<double, std::milli>(save_done - save_start).count());
  }

  std::unique_ptr<StaticCheckpointIndex<T>> foreground_snapshot;
  LiveAttrIndexes foreground_indexes;
  uint32_t foreground_snapshot_version = 0;
  const std::string foreground_snapshot_base = index.index_prefix() + "_foreground_snapshot";
  if (foreground_enabled) {
    reload_foreground_snapshot(foreground_snapshot, foreground_indexes, queries, metric, nbr_type, index.index_prefix(),
                               foreground_snapshot_base, foreground_snapshot_version, label_config, search_threads);
  }

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
      foreground_search(*foreground_snapshot->index, queries, k, L, foreground_rounds, search_threads, "after_mark_delete", cycle, fg);
    }

    write_progress(progress, cycle, "merge_start", 0, 0, 0.0);
    auto merge_start = std::chrono::high_resolution_clock::now();
    auto merge_future = std::async(std::launch::async, [&]() { index.save(index.index_prefix(), merge_threads); });
    while (merge_future.wait_for(std::chrono::milliseconds(foreground_interval_ms)) != std::future_status::ready) {
      auto now = std::chrono::high_resolution_clock::now();
      write_progress(progress, cycle, "merge_running", 0, 0,
                     std::chrono::duration<double, std::milli>(now - merge_start).count());
      if (foreground_enabled) {
        foreground_search(*foreground_snapshot->index, queries, k, L, foreground_rounds, search_threads, "merge", cycle, fg);
      }
    }
    merge_future.get();
    auto t2 = std::chrono::high_resolution_clock::now();
    double merge_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    write_progress(progress, cycle, "merge_done", 0, 0, merge_ms);
    if (foreground_enabled) {
      reload_foreground_snapshot(foreground_snapshot, foreground_indexes, queries, metric, nbr_type, index.index_prefix(),
                                 foreground_snapshot_base, foreground_snapshot_version, label_config, search_threads);
    }

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
        foreground_search(*foreground_snapshot->index, queries, k, L, foreground_rounds, search_threads, "insert", cycle, fg);
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
      if (foreground_enabled) {
        reload_foreground_snapshot(foreground_snapshot, foreground_indexes, queries, metric, nbr_type, index.index_prefix(),
                                   foreground_snapshot_base, foreground_snapshot_version, label_config, search_threads);
      }
    }

    if (foreground_enabled) {
      foreground_search(*foreground_snapshot->index, queries, k, L, foreground_rounds, search_threads, "after_insert", cycle, fg);
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
