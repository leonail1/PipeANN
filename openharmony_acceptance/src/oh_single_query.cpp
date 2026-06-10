#include "oh_common.h"

#include "filter/attribute.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include "utils.h"

#include <thread>

namespace {

uint64_t current_rss_bytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      uint64_t kb = 0;
      std::string unit;
      status >> kb >> unit;
      return kb * 1024;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0;
}

void write_rss_trace(const std::string &path, const std::string &stage) {
  if (path.empty()) {
    return;
  }
  oh::ensure_parent(path);
  std::ofstream out(path, std::ios::app);
  out << "{\"stage\":\"" << oh::json_escape(stage) << "\",\"rss_bytes\":" << current_rss_bytes() << "}\n";
}

template<typename T>
void load_first_query(const std::string &query_path, T *&query, size_t &query_num, size_t &query_dim) {
  std::ifstream reader(query_path, std::ios::binary);
  if (!reader) {
    throw std::runtime_error("failed to open query file: " + query_path);
  }
  int npts_i32 = 0;
  int dim_i32 = 0;
  reader.read(reinterpret_cast<char *>(&npts_i32), sizeof(int));
  reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int));
  if (!reader || npts_i32 <= 0 || dim_i32 <= 0) {
    throw std::runtime_error("invalid query file header: " + query_path);
  }
  query_num = 1;
  query_dim = static_cast<size_t>(dim_i32);
  query = new T[query_dim];
  reader.read(reinterpret_cast<char *>(query), query_dim * sizeof(T));
  if (!reader) {
    throw std::runtime_error("failed to read first query vector: " + query_path);
  }
}

template<typename T>
int run_single(int argc, char **argv) {
  oh::Args args(argc, argv);
  const std::string index_prefix = args.get("index-prefix");
  const std::string query_path = args.get("query");
  const std::string label_config = args.get("label-config", "");
  const std::string selector_id = args.get("selector-id", "single_query");
  const std::string metric_name = args.get("metric", "l2");
  const std::string nbr_type = args.get("nbr-type", "pq");
  const uint32_t beamwidth = args.u32("beamwidth", 32);
  const uint32_t mem_l = args.u32("mem-L", 0);
  const uint64_t k = args.u64("k", 10);
  const uint64_t l_search = args.u64("L", 100);
  const auto out_jsonl = std::filesystem::path(args.get("out-jsonl", "results/single_query_resource.jsonl"));
  const std::string rss_trace_path = args.get("rss-trace", "");
  const uint64_t pause_after_load_seconds = args.u64("pause-after-load-seconds", 0);
  const uint64_t pause_after_search_seconds = args.u64("pause-after-search-seconds", 0);

  if (index_prefix.empty() || query_path.empty()) {
    throw std::runtime_error("--index-prefix and --query are required");
  }

  write_rss_trace(rss_trace_path, "start");
  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  load_first_query(query_path, query, query_num, query_dim);
  if (query_num == 0) {
    throw std::runtime_error("query file is empty");
  }
  write_rss_trace(rss_trace_path, "after_query_load");

  auto metric = pipeann::get_metric(metric_name);
  setenv("PIPEANN_PQ_STREAM_LOAD", args.get("pq-stream", "0").c_str(), 1);
  setenv("PIPEANN_PQ_MMAP_LOAD", args.get("pq-mmap", "0").c_str(), 1);
  setenv("PIPEANN_PQ_MMAP_DONTNEED_EVERY", args.get("pq-mmap-dontneed-every", "0").c_str(), 1);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  pipeann::IndexBuildParameters params;
  params.max_nthreads = 1;
  pipeann::SSDIndex<T> index(metric, reader, nbr_handler, true, &params);
  if (index.load(index_prefix.c_str(), false) != 0) {
    throw std::runtime_error("Failed to load index: " + index_prefix);
  }
  write_rss_trace(rss_trace_path, "after_index_load");

  std::map<uint32_t, pipeann::AttrIndex *> base_stores;
  pipeann::Selector *selector = nullptr;
  std::vector<pipeann::Attributes> query_attrs;
  if (!label_config.empty() && label_config != "null") {
    base_stores = pipeann::load_base_attr_from_config(label_config, index.meta_.npoints);
    auto ret = pipeann::load_selector_from_config(label_config, base_stores);
    selector = ret.first;
    query_attrs = std::move(ret.second);
  }
  write_rss_trace(rss_trace_path, "after_selector_load");
  if (pause_after_load_seconds > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(pause_after_load_seconds));
  }

  std::vector<uint32_t> result_tags(k);
  std::vector<float> result_dists(k);
  pipeann::QueryStats stats;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (selector != nullptr) {
    index.spec_filter_search(query, k, l_search, selector, query_attrs[0], result_tags.data(), result_dists.data(),
                             beamwidth, &stats);
  } else {
    index.pipe_search(query, k, mem_l, l_search, result_tags.data(), result_dists.data(), beamwidth, &stats);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  write_rss_trace(rss_trace_path, "after_search");
  if (pause_after_search_seconds > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(pause_after_search_seconds));
  }
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double latency_ms = stats.total_us > 0 ? stats.total_us / 1000.0 : wall_ms;

  oh::ensure_parent(out_jsonl);
  std::ofstream out(out_jsonl, std::ios::app);
  out << "{\"selector_id\":\"" << oh::json_escape(selector_id) << "\",\"latency_ms\":" << latency_ms
      << ",\"wall_ms\":" << wall_ms << ",\"ios\":" << stats.n_ios << ",\"k\":" << k << ",\"L\":" << l_search
      << "}\n";

  delete selector;
  for (auto &[_, store] : base_stores) {
    delete store;
  }
  delete[] query;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    std::string type = args.get("type", "float");
    if (type == "float") {
      return run_single<float>(argc, argv);
    }
    if (type == "uint8") {
      return run_single<uint8_t>(argc, argv);
    }
    if (type == "int8") {
      return run_single<int8_t>(argc, argv);
    }
    throw std::runtime_error("Unsupported --type: " + type);
  } catch (const std::exception &e) {
    std::cerr << "oh_single_query failed: " << e.what() << std::endl;
    return 1;
  }
}
