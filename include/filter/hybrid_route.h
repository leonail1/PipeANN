#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "filter/densebit_index.h"

namespace pipeann {
  enum class HybridRouteDecision : uint8_t {
    kAutoGraphFallback = 0,
    kPrefilter = 1,
    kGraphOnly = 2,
    kPrefilterFastReturn = 3,
  };

  enum class HybridRouteOverride : uint8_t {
    kAuto = 0,
    kForcePrefilter = 1,
    kForceGraphOnly = 2,
  };

  struct HybridQueryStats {
    uint64_t candidate_count = 0;
    uint64_t threshold = 0;
    uint64_t threshold_version = 0;
    HybridRouteDecision decision = HybridRouteDecision::kAutoGraphFallback;
    uint64_t route_overhead_us = 0;
  };

  inline size_t compute_prefilter_rerank_l(uint64_t k_search, size_t candidate_count, size_t total_points) {
    if (candidate_count == 0) {
      return 0;
    }

    const char *override_value = std::getenv("PIPEANN_PREFILTER_RERANK_L");
    if (override_value != nullptr && override_value[0] != '\0') {
      char *end_ptr = nullptr;
      const unsigned long long parsed_value = std::strtoull(override_value, &end_ptr, 10);
      if (end_ptr != override_value && end_ptr != nullptr && *end_ptr == '\0') {
        size_t target = static_cast<size_t>(parsed_value);
        target = std::max<size_t>(target, static_cast<size_t>(k_search));
        return std::min(target, candidate_count);
      }
    }

    const double selectivity =
        total_points > 0 ? static_cast<double>(candidate_count) / static_cast<double>(total_points) : 1.0;
    size_t target = 192;
    if (selectivity <= 0.005) {
      target = 96;
    } else if (selectivity <= 0.02) {
      target = 128;
    } else if (selectivity <= 0.1) {
      target = 160;
    }

    target = std::max<size_t>(target, static_cast<size_t>(k_search));
    return std::min(target, candidate_count);
  }
}  // namespace pipeann