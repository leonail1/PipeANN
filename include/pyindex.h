#pragma once

#include "dynamic_index.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyIndexInterface {
 public:
  PyIndexInterface() = delete;
  PyIndexInterface(uint32_t data_dim, const std::string &data_type, pipeann::Metric metric,
                   pipeann::IndexBuildParameters *params = nullptr);

  // lifecycle and I/O
  void load(const std::string &index_prefix);
  bool save(const std::string &index_prefix);

  // build
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 0,
             uint32_t memory_use_GB = 0, const std::vector<pipeann::Attributes> *attrs_vec = nullptr,
             uint32_t range_dense = 0, const std::string &train_query_path = "", uint32_t R_ood = 0,
             uint32_t L_ood = 1500);
  std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(uint32_t key, const std::string &filename,
                                                                const std::string &attr_type);
  std::pair<pipeann::Selector *, std::vector<pipeann::Attributes>> load_filter_from_json(
      const std::string &config_path);

  // updates and queries
  std::tuple<py::array, py::array> search(py::array &queries, uint32_t topk, uint32_t L, pipeann::Selector *selector,
                                          const std::vector<pipeann::Attributes> &query_attrs, float range);
  void add(py::array &vectors, py::array &tags, const std::vector<pipeann::Attributes> *attrs_vec = nullptr);
  void remove(py::array &tags);
  void set_index_prefix(const std::string &index_prefix);
  void omp_set_num_threads(uint32_t num_threads);
  size_t npoints() const;

  std::string to_string() const;

 private:
  template<typename T>
  DynamicIndex<T> *get() const {
    return dynamic_cast<DynamicIndex<T> *>(impl_.get());
  }

  // Type-erased dispatch: calls fn(DynamicIndex<T>*) for the correct T.
  template<typename F>
  decltype(auto) dispatch(F &&fn) const {
    if (auto *p = get<float>())
      return fn(p);
    if (auto *p = get<uint8_t>())
      return fn(p);
    if (auto *p = get<int8_t>())
      return fn(p);
    throw std::runtime_error("Invalid underlying index");
  }

 private:
  uint32_t data_dim_;
  pipeann::Metric metric_;
  std::unique_ptr<BaseDynamicIndex> impl_;
};
