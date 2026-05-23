#pragma once
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "nbr/abstract_nbr.h"
#include "filter/label.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  inline bool validate_label_build_inputs(AbstractLabel *label, const char *label_source_file) {
    const bool has_label_writer = label != nullptr;
    const bool has_label_source = label_source_file != nullptr && label_source_file[0] != '\0';
    if (!has_label_writer && !has_label_source) {
      return true;
    }
    if (!has_label_writer) {
      LOG(INFO) << "label_source_file provided without a label writer; labels will be written to sidecar only";
      return true;
    }
    if (!has_label_source) {
      LOG(ERROR) << "Filtered build requires label_source_file to generate <index_prefix>_labels.densebit";
      return false;
    }
    return true;
  }

  template<typename T>
  void normalize_data_file(const std::string &inFileName, const std::string &outFileName);

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const uint64_t nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file);

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_index_file,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler, AbstractLabel *label,
                        const char *label_source_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &output_file, AbstractLabel *label,
                          const std::string &label_source_file = "");
}  // namespace pipeann
