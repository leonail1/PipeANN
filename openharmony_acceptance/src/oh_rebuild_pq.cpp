#include "oh_common.h"

#include "distance.h"
#include "nbr/pq_nbr.h"

namespace {

template<typename T>
int run_rebuild(int argc, char **argv) {
  oh::Args args(argc, argv);
  const std::string index_prefix = args.get("index-prefix");
  const std::string data_path = args.get("data");
  const std::string metric_name = args.get("metric", "l2");
  const uint32_t pq_bytes = args.u32("pq-bytes", 16);
  if (index_prefix.empty() || data_path.empty()) {
    throw std::runtime_error("--index-prefix and --data are required");
  }
  auto metric = pipeann::get_metric(metric_name);
  pipeann::PQNeighbor<T> pq(metric);
  pq.build(index_prefix, data_path, pq_bytes);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    std::string type = args.get("type", "float");
    if (type == "float") {
      return run_rebuild<float>(argc, argv);
    }
    if (type == "uint8") {
      return run_rebuild<uint8_t>(argc, argv);
    }
    if (type == "int8") {
      return run_rebuild<int8_t>(argc, argv);
    }
    throw std::runtime_error("Unsupported --type: " + type);
  } catch (const std::exception &e) {
    std::cerr << "oh_rebuild_pq failed: " << e.what() << std::endl;
    return 1;
  }
}
