#pragma once

#include <cstdint>
#include <vector>

#include "filter/attribute.h"
#include "nbr/abstract_nbr.h"
#include "utils.h"

namespace pipeann {
  // PiPNN (Pick-in-Partitions Nearest Neighbors) is a graph construction algorithm.
  // It partitions the dataset into overlapping sub-problems and leverages dense matrix multiplication kernels.
  struct WeightedEdge {
    uint32_t u, v;
    float dist;
  };

  struct HashPruneSlot {
    uint16_t hash;
    uint32_t id;
    float dist;

    bool operator<(const HashPruneSlot &other) const {
      return hash < other.hash;
    }
  };

  // HashPruner deduplicates and prunes PiPNN candidate edges before materializing the final graph.
  struct HashPruner {
    float *_norm_vectors;
    float *_sketch;
    std::vector<std::vector<HashPruneSlot>> _reservoirs;
    int _num_hyperplanes;
    int _dims;
    int64_t _num_nodes;
    size_t _climit;

    template<typename T>
    HashPruner(int64_t num_nodes, int64_t dims, int num_hyperplanes, int climit, const T *raw_data, uint32_t seed = 42);

    ~HashPruner();

    void update(const std::vector<WeightedEdge> &edges);
    void to_graph(std::vector<std::vector<unsigned>> &final_graph);
  };
}  // namespace pipeann
