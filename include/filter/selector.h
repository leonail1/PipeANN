#ifndef SELECTOR_H_
#define SELECTOR_H_

#include <algorithm>
#include <cstring>
#include <vector>

#include "ssd_index_defs.h"

namespace pipeann {
  /*
   * Selector is defined similar to Selector in faiss.
   * It is used to filter the results of the search (using query Label and target Label).
   */

  struct AbstractSelector {
    virtual ~AbstractSelector() = default;

    // Check if the target metadata meets the filter criteria.
    virtual bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) = 0;
  };

  // A dummy selector that always returns true.
  struct DummySelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      return true;
    }
  };

  // A simple range filter selector.
  // Query metadata is normally a label buffer:
  //   [count=1][value] means equality, [count=2][low][high] means range.
  // Target metadata is a raw uint32 scalar. The selector deliberately does not infer
  // variable-length label buffers from target_labels because the interface does not
  // carry a buffer length, and old callers may pass only four bytes.
  // If target_labels is nullptr, returns false (no extra data).
  struct RangeSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      (void) target_id;
      if (unlikely(target_labels == nullptr)) {
        return false; /* nullptr means no extra data */
      }

      uint32_t low = 0, high = 0;
      if (!decode_query_range(query_labels, &low, &high)) {
        return false;
      }

      uint32_t target = 0;
      memcpy(&target, target_labels, sizeof(uint32_t));
      return target >= low && target <= high;
    }

   private:
    static bool decode_query_range(const void *query_labels, uint32_t *low, uint32_t *high) {
      if (query_labels == nullptr || low == nullptr || high == nullptr) {
        return false;
      }

      uint32_t count_or_low = 0;
      memcpy(&count_or_low, query_labels, sizeof(uint32_t));
      if (count_or_low == 0) {
        return false;
      }

      if (count_or_low == 1 || count_or_low == 2) {
        uint32_t first = 0, second = 0;
        memcpy(&first, static_cast<const char *>(query_labels) + sizeof(uint32_t), sizeof(uint32_t));
        if (count_or_low == 1) {
          *low = first;
          *high = first;
          return true;
        }
        memcpy(&second, static_cast<const char *>(query_labels) + 2 * sizeof(uint32_t), sizeof(uint32_t));
        *low = std::min(first, second);
        *high = std::max(first, second);
        return true;
      }

      uint32_t raw_high = 0;
      memcpy(&raw_high, static_cast<const char *>(query_labels) + sizeof(uint32_t), sizeof(uint32_t));
      *low = std::min(count_or_low, raw_high);
      *high = std::max(count_or_low, raw_high);
      return true;
    }

  };

  // The selector checks if query and target label sets have non-empty intersection.
  // Assumptions:
  // - Query metadata: Contains label set Fq in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  // - Target metadata: Contains label set Fx in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  //   Labels may not be sorted and may contain duplicates
  struct LabelIntersectionSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      uint32_t query_count, target_count;
      memcpy(&query_count, query_labels, sizeof(uint32_t));
      memcpy(&target_count, target_labels, sizeof(uint32_t));

      if (query_count == 0 || target_count == 0) {
        return false;
      }

      std::vector<uint32_t> query_labels_vec(query_count);
      std::vector<uint32_t> target_labels_vec(target_count);
      memcpy(query_labels_vec.data(), (char *) query_labels + sizeof(uint32_t), query_count * sizeof(uint32_t));
      memcpy(target_labels_vec.data(), (char *) target_labels + sizeof(uint32_t), target_count * sizeof(uint32_t));

      for (uint32_t q_idx = 0; q_idx < query_count; ++q_idx) {
        for (uint32_t t_idx = 0; t_idx < target_count; ++t_idx) {
          if (query_labels_vec[q_idx] == target_labels_vec[t_idx]) {
            return true;
          }
        }
      }
      return false;
    }
  };

  // The selector checks if query set is a subset of the target label set.
  // Could be used in NIPS 2023 bigann benchmark.
  // - Query metadata: Contains label set Fq in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  // - Target metadata: Contains label set Fx in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  struct LabelSubsetSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      uint32_t query_count, target_count;

      memcpy(&query_count, query_labels, sizeof(uint32_t));
      memcpy(&target_count, target_labels, sizeof(uint32_t));

      if (query_count == 0) {
        return true;
      }
      if (target_count == 0) {
        return false;
      }

      std::vector<uint32_t> query_labels_vec(query_count);
      std::vector<uint32_t> target_labels_vec(target_count);
      memcpy(query_labels_vec.data(), (char *) query_labels + sizeof(uint32_t), query_count * sizeof(uint32_t));
      memcpy(target_labels_vec.data(), (char *) target_labels + sizeof(uint32_t), target_count * sizeof(uint32_t));

      for (uint32_t q_idx = 0; q_idx < query_count; ++q_idx) {
        bool found = false;
        for (uint32_t t_idx = 0; t_idx < target_count; ++t_idx) {
          if (query_labels_vec[q_idx] == target_labels_vec[t_idx]) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    }
  };

  template<typename T>
  inline AbstractSelector *get_selector(const std::string &selector_type) {
    if (selector_type == "range") {
      return new RangeSelector();
    } else if (selector_type == "intersect") {
      return new LabelIntersectionSelector();
    } else if (selector_type == "subset") {
      return new LabelSubsetSelector();
    }
    return nullptr;
  }
}  // namespace pipeann

#endif
