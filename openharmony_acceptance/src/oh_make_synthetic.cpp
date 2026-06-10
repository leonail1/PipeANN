#include "oh_common.h"

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    const auto out_dir = std::filesystem::path(args.get("out-dir", "acceptance_work/smoke"));
    const uint32_t nbase = args.u32("nbase", 2000);
    const uint32_t nupdates = args.u32("nupdates", 4000);
    const uint32_t nquery = args.u32("nquery", 50);
    const uint32_t dim = args.u32("dim", 32);
    const uint64_t seed = args.u64("seed", 42);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, 4.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.25f);
    constexpr uint32_t kCenters = 16;
    std::vector<float> centers(kCenters * dim);
    for (auto &v : centers) {
      v = center_dist(rng);
    }

    auto make_vectors = [&](uint32_t n) {
      std::vector<float> data(static_cast<size_t>(n) * dim);
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = static_cast<uint32_t>(oh::hash64(i + data.size()) % kCenters);
        for (uint32_t d = 0; d < dim; ++d) {
          data[static_cast<size_t>(i) * dim + d] = centers[static_cast<size_t>(c) * dim + d] + noise_dist(rng);
        }
      }
      return data;
    };

    auto base = make_vectors(nbase);
    auto updates = make_vectors(nupdates);
    std::vector<float> query(static_cast<size_t>(nquery) * dim);
    for (uint32_t i = 0; i < nquery; ++i) {
      uint32_t src = static_cast<uint32_t>(oh::hash64(i) % nbase);
      for (uint32_t d = 0; d < dim; ++d) {
        query[static_cast<size_t>(i) * dim + d] = base[static_cast<size_t>(src) * dim + d] + noise_dist(rng) * 0.05f;
      }
    }

    std::filesystem::create_directories(out_dir);
    oh::write_bin_matrix<float>(out_dir / "base.bin", base, nbase, dim);
    oh::write_bin_matrix<float>(out_dir / "updates.bin", updates, nupdates, dim);
    oh::write_bin_matrix<float>(out_dir / "query.bin", query, nquery, dim);
    std::cout << "{\"base\":\"" << (out_dir / "base.bin").string() << "\",\"updates\":\""
              << (out_dir / "updates.bin").string() << "\",\"query\":\"" << (out_dir / "query.bin").string()
              << "\"}\n";
  } catch (const std::exception &e) {
    std::cerr << "oh_make_synthetic failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
