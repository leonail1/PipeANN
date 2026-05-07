#include "dynamic_index.h"
#include "distance.h"

#include <cmath>
#include <cstring>
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

  std::vector<char> label_filter(std::initializer_list<uint32_t> labels) {
    std::vector<char> buffer(sizeof(uint32_t) * (labels.size() + 1), 0);
    const uint32_t count = static_cast<uint32_t>(labels.size());
    std::memcpy(buffer.data(), &count, sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t label : labels) {
      std::memcpy(buffer.data() + sizeof(uint32_t) * (idx + 1), &label, sizeof(uint32_t));
      ++idx;
    }
    return buffer;
  }

  void add_labels(pipeann::DynamicSSDIndex<float, uint32_t> &index, uint32_t tag,
                  std::initializer_list<uint32_t> labels) {
    std::vector<uint32_t> label_vec(labels);
    const int rc = index.update_labels(tag, label_vec.data(), static_cast<uint32_t>(label_vec.size()));
    require(rc == 0, "update_labels failed for tag " + std::to_string(tag));
  }
}

int main() {
  try {
    pipeann::IndexBuildParameters params;
    params.set(4, 8, 32, 1.2f, 1, true, 4);
    pipeann::DistanceL2Float dist;
    pipeann::DynamicSSDIndex<float, uint32_t> index(params, "/tmp/pipeann_dynamic_flat_index", 2, &dist,
                                                    pipeann::Metric::L2, 10000, PIPE_SEARCH);

    require(index.is_flat_mode(), "index should start in flat mode");
    require(index.live_point_count() == 0, "empty flat index should have zero live points");

    uint32_t tags[3] = {};
    float dists[3] = {};
    const float q0[2] = {0.0f, 0.0f};
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr);
    require(tags[0] == std::numeric_limits<uint32_t>::max(), "empty search should return no tags");

    const float p10[2] = {0.0f, 0.0f};
    const float p20[2] = {1.0f, 0.0f};
    const float p30[2] = {2.0f, 0.0f};
    const float p40[2] = {5.0f, 0.0f};
    index.insert(p10, 10);
    index.insert(p20, 20);
    index.insert(p30, 30);
    index.insert(p40, 40);

    add_labels(index, 10, {1});
    add_labels(index, 20, {2});
    add_labels(index, 30, {1, 2});
    add_labels(index, 40, {3});

    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr);
    require(tags[0] == 10 && tags[1] == 20 && tags[2] == 30, "flat exact search order mismatch");
    require(std::fabs(dists[0] - 0.0f) < 1e-5f && std::fabs(dists[1] - 1.0f) < 1e-5f,
            "flat exact distances mismatch");

    auto intersect_1 = label_filter({1});
    pipeann::HybridQueryStats hybrid_stats;
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr, true, pipeann::HybridFilterKind::kIntersect,
                 intersect_1.data(), &hybrid_stats);
    require(tags[0] == 10 && tags[1] == 30, "flat intersect filter mismatch");
    require(hybrid_stats.candidate_count == 2, "flat intersect candidate count mismatch");
    require(hybrid_stats.decision == pipeann::HybridRouteDecision::kPrefilter,
            "flat filter should use prefilter decision");

    auto subset_12 = label_filter({1, 2});
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr, true, pipeann::HybridFilterKind::kSubset, subset_12.data());
    require(tags[0] == 30 && tags[1] == std::numeric_limits<uint32_t>::max(), "flat subset filter mismatch");

    auto range_23 = label_filter({2, 3});
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr, true, pipeann::HybridFilterKind::kRange, range_23.data());
    require(tags[0] == 20 && tags[1] == 30 && tags[2] == 40, "flat range filter mismatch");

    index.lazy_delete(10);
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr);
    require(tags[0] == 20, "deleted tag should not be returned");

    add_labels(index, 20, {4});
    auto intersect_2 = label_filter({2});
    index.search(q0, 3, 0, 3, 4, tags, dists, nullptr, true, pipeann::HybridFilterKind::kIntersect,
                 intersect_2.data());
    require(tags[0] == 30 && tags[1] == std::numeric_limits<uint32_t>::max(),
            "label update should affect flat filter results");

    pipeann::DynamicSSDIndex<float, uint32_t> transition_index(params, "/tmp/pipeann_dynamic_flat_index_transition", 2,
                                                               &dist, pipeann::Metric::L2, 3, PIPE_SEARCH);
    transition_index.insert(p10, 10);
    transition_index.insert(p20, 20);
    transition_index.insert(p30, 30);
    require(transition_index.is_flat_mode(), "threshold should not materialize at exactly the threshold");
    transition_index.insert(p40, 40);
    require(!transition_index.is_flat_mode(), "threshold+1 should materialize to disk mode");
    transition_index.search(q0, 2, 0, 10, 4, tags, dists, nullptr);
    require(tags[0] == 10, "disk search after flat materialization should return inserted points");

    std::cout << "dynamic_flat_index: ok" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "dynamic_flat_index: " << e.what() << std::endl;
    return 1;
  }
}
