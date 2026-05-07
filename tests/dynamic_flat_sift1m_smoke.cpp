#include "dynamic_index.h"
#include "distance.h"
#include "utils.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  void require(bool condition, const std::string &message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void check_search(pipeann::DynamicSSDIndex<float, uint32_t> &index, const float *query, uint32_t expected_tag,
                    uint64_t topk, uint64_t search_l, bool require_exact_top1) {
    std::vector<uint32_t> tags(static_cast<size_t>(topk), std::numeric_limits<uint32_t>::max());
    std::vector<float> dists(static_cast<size_t>(topk), std::numeric_limits<float>::infinity());
    pipeann::QueryStats stats;
    index.search(query, topk, 0, search_l, 4, tags.data(), dists.data(), &stats);
    if (require_exact_top1) {
      require(tags[0] == expected_tag,
              "search expected tag " + std::to_string(expected_tag) + " but got " + std::to_string(tags[0]));
    } else {
      require(tags[0] != std::numeric_limits<uint32_t>::max(), "disk search returned no result");
    }
  }
}

int main(int argc, char **argv) {
  try {
    const std::string data_path = argc > 1 ? argv[1] : "data/sift1m/sift_base.bin";
    const std::string index_prefix = argc > 2 ? argv[2] : "/tmp/pipeann_flat_sift1m_smoke";
    const uint64_t insert_count = argc > 3 ? static_cast<uint64_t>(std::stoull(argv[3])) : 12000ULL;
    const uint64_t threshold = argc > 4 ? static_cast<uint64_t>(std::stoull(argv[4])) : 10000ULL;

    size_t npts = 0, dim = 0;
    std::vector<float> data;
    pipeann::load_bin<float>(data_path, data, npts, dim);
    require(npts >= insert_count, "SIFT input does not contain enough points");
    require(dim > 0, "SIFT input dimension is zero");

    pipeann::IndexBuildParameters params;
    params.set(64, 96, 384, 1.2f, 4, true, 4);
    pipeann::DistanceL2Float dist;
    pipeann::DynamicSSDIndex<float, uint32_t> index(params, index_prefix, static_cast<uint32_t>(dim), &dist,
                                                    pipeann::Metric::L2, threshold, PIPE_SEARCH);

    const std::vector<uint64_t> checkpoints = {1, threshold - 1, threshold, threshold + 1, insert_count};
    size_t checkpoint_pos = 0;
    for (uint64_t i = 0; i < insert_count; ++i) {
      const int id = index.insert(data.data() + static_cast<size_t>(i) * dim, static_cast<uint32_t>(i));
      require(id >= 0, "insert returned negative id at " + std::to_string(i));

      while (checkpoint_pos < checkpoints.size() && checkpoints[checkpoint_pos] == i + 1) {
        const uint64_t count = checkpoints[checkpoint_pos];
        check_search(index, data.data() + static_cast<size_t>(i) * dim, static_cast<uint32_t>(i), 10, 96,
                     count <= threshold + 1);
        if (count <= threshold) {
          require(index.is_flat_mode(), "index left flat mode too early at count " + std::to_string(count));
        } else {
          require(!index.is_flat_mode(), "index did not materialize after threshold at count " + std::to_string(count));
          require(file_exists(index_prefix + "_disk.index"), "disk index file missing after materialization");
          require(file_exists(index_prefix + "_pq_pivots.bin"), "PQ pivots missing after materialization");
          require(file_exists(index_prefix + "_pq_compressed.bin"), "PQ compressed vectors missing after materialization");
        }
        std::cout << "checkpoint count=" << count << " flat=" << (index.is_flat_mode() ? 1 : 0)
                  << " live=" << index.live_point_count() << " ok" << std::endl;
        ++checkpoint_pos;
      }
    }

    std::cout << "dynamic_flat_sift1m_smoke: ok inserted=" << insert_count
              << " threshold=" << threshold << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "dynamic_flat_sift1m_smoke: " << e.what() << std::endl;
    return 1;
  }
}
