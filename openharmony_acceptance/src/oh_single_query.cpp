#include "oh_common.h"

#include "filter/attribute.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include "utils.h"

namespace {

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

  if (index_prefix.empty() || query_path.empty()) {
    throw std::runtime_error("--index-prefix and --query are required");
  }

  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  pipeann::load_bin<T>(query_path, query, query_num, query_dim);
  if (query_num == 0) {
    throw std::runtime_error("query file is empty");
  }

  auto metric = pipeann::get_metric(metric_name);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  pipeann::IndexBuildParameters params;
  params.max_nthreads = 1;
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
