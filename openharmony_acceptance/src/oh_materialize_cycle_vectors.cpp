#include "oh_common.h"

namespace {

std::pair<uint64_t, uint64_t> delete_range_for_cycle(uint32_t cycle, uint64_t npoints) {
  uint64_t delete_count = npoints * 6 / 10;
  if (cycle % 2 == 1) {
    return {npoints - delete_count, npoints};
  }
  return {0, delete_count};
}

template<typename T>
int run_materialize(int argc, char **argv) {
  oh::Args args(argc, argv);
  const auto base_path = std::filesystem::path(args.get("base"));
  const auto updates_path = std::filesystem::path(args.get("updates"));
  const auto out_path = std::filesystem::path(args.get("out"));
  const uint32_t cycle = args.u32("cycle", 0);
  const uint64_t npoints = args.u64("npoints", 1000000);
  const uint64_t update_rows_per_cycle = args.u64("update-rows-per-cycle", npoints * 6 / 10);

  if (base_path.empty() || updates_path.empty() || out_path.empty()) {
    throw std::runtime_error("--base, --updates and --out are required");
  }

  uint32_t base_dim = 0;
  auto current = oh::read_bin_rows<T>(base_path, 0, npoints, base_dim);
  for (uint32_t c = 1; c <= cycle; ++c) {
    auto [begin, end] = delete_range_for_cycle(c, npoints);
    uint64_t count = end - begin;
    uint32_t update_dim = 0;
    auto batch = oh::read_bin_rows<T>(updates_path, static_cast<uint64_t>(c - 1) * update_rows_per_cycle, count,
                                      update_dim);
    if (update_dim != base_dim) {
      throw std::runtime_error("Update vector dimension mismatch");
    }
    std::copy(batch.begin(), batch.end(), current.begin() + static_cast<std::ptrdiff_t>(begin * base_dim));
  }

  oh::write_bin_matrix<T>(out_path, current, static_cast<uint32_t>(npoints), base_dim);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    std::string type = args.get("type", "float");
    if (type == "float") {
      return run_materialize<float>(argc, argv);
    }
    if (type == "uint8") {
      return run_materialize<uint8_t>(argc, argv);
    }
    if (type == "int8") {
      return run_materialize<int8_t>(argc, argv);
    }
    throw std::runtime_error("Unsupported --type: " + type);
  } catch (const std::exception &e) {
    std::cerr << "oh_materialize_cycle_vectors failed: " << e.what() << std::endl;
    return 1;
  }
}
