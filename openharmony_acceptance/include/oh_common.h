#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace oh {

namespace fs = std::filesystem;

struct Args {
  std::map<std::string, std::string> values;

  explicit Args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key(argv[i]);
      if (key.rfind("--", 0) != 0) {
        throw std::runtime_error("Unexpected positional argument: " + key);
      }
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + key);
      }
      values[key.substr(2)] = argv[++i];
    }
  }

  std::string get(const std::string &key, const std::string &fallback = "") const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }

  uint64_t u64(const std::string &key, uint64_t fallback = 0) const {
    auto value = get(key);
    return value.empty() ? fallback : std::stoull(value);
  }

  uint32_t u32(const std::string &key, uint32_t fallback = 0) const {
    return static_cast<uint32_t>(u64(key, fallback));
  }

  double f64(const std::string &key, double fallback = 0.0) const {
    auto value = get(key);
    return value.empty() ? fallback : std::stod(value);
  }
};

inline void ensure_parent(const fs::path &path) {
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }
}

inline std::string json_escape(const std::string &text) {
  std::ostringstream out;
  for (char c : text) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

inline void write_text(const fs::path &path, const std::string &text) {
  ensure_parent(path);
  std::ofstream writer(path);
  if (!writer) {
    throw std::runtime_error("Failed to open " + path.string());
  }
  writer << text;
}

inline uint64_t file_size_or_zero(const fs::path &path) {
  std::error_code ec;
  if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
    return 0;
  }
  return fs::file_size(path, ec);
}

inline std::vector<double> sorted_copy(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values;
}

inline double percentile(const std::vector<double> &sorted_values, double p) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  auto idx = static_cast<size_t>(p * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[std::min(idx, sorted_values.size() - 1)];
}

inline double mean(const std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

inline std::vector<uint32_t> parse_u32_list(const std::string &text) {
  std::vector<uint32_t> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      values.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
  }
  return values;
}

inline uint64_t now_ms() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

inline uint64_t hash64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline std::vector<uint64_t> stable_ranks(uint64_t npoints) {
  std::vector<uint64_t> ids(npoints);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [](uint64_t a, uint64_t b) {
    return hash64(a) < hash64(b);
  });
  std::vector<uint64_t> rank(npoints);
  for (uint64_t i = 0; i < ids.size(); ++i) {
    rank[ids[i]] = i;
  }
  return rank;
}

inline std::vector<std::pair<std::string, double>> selectivities() {
  return {
      {"s0001", 0.0001}, {"s001", 0.001}, {"s01", 0.01}, {"s05", 0.05},
      {"s10", 0.10},    {"s25", 0.25},  {"s50", 0.50}, {"s100", 1.00},
  };
}

template<typename T>
void write_bin_matrix(const fs::path &path, const std::vector<T> &data, uint32_t npts, uint32_t dim) {
  ensure_parent(path);
  std::ofstream writer(path, std::ios::binary);
  if (!writer) {
    throw std::runtime_error("Failed to open " + path.string());
  }
  writer.write(reinterpret_cast<const char *>(&npts), sizeof(uint32_t));
  writer.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
  writer.write(reinterpret_cast<const char *>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void read_bin_metadata(const fs::path &path, uint32_t &npts, uint32_t &dim) {
  std::ifstream reader(path, std::ios::binary);
  if (!reader) {
    throw std::runtime_error("Failed to open " + path.string());
  }
  reader.read(reinterpret_cast<char *>(&npts), sizeof(uint32_t));
  reader.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
}

template<typename T>
std::vector<T> read_bin_rows(const fs::path &path, uint64_t start, uint64_t count, uint32_t &dim_out) {
  uint32_t npts = 0, dim = 0;
  read_bin_metadata<T>(path, npts, dim);
  if (start + count > npts) {
    throw std::runtime_error("Requested rows exceed " + path.string());
  }
  std::ifstream reader(path, std::ios::binary);
  reader.seekg(2 * sizeof(uint32_t) + static_cast<std::streamoff>(start * dim * sizeof(T)), std::ios::beg);
  std::vector<T> data(count * dim);
  reader.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(sizeof(T) * data.size()));
  dim_out = dim;
  return data;
}

inline void write_spmat(const fs::path &path, int64_t nrow, int64_t ncol,
                        const std::vector<std::vector<int32_t>> &indices,
                        const std::vector<std::vector<float>> &values) {
  ensure_parent(path);
  int64_t nnz = 0;
  for (const auto &row : indices) {
    nnz += static_cast<int64_t>(row.size());
  }
  std::ofstream writer(path, std::ios::binary);
  if (!writer) {
    throw std::runtime_error("Failed to open " + path.string());
  }
  writer.write(reinterpret_cast<const char *>(&nrow), sizeof(int64_t));
  writer.write(reinterpret_cast<const char *>(&ncol), sizeof(int64_t));
  writer.write(reinterpret_cast<const char *>(&nnz), sizeof(int64_t));
  int64_t pos = 0;
  for (int64_t i = 0; i < nrow; ++i) {
    writer.write(reinterpret_cast<const char *>(&pos), sizeof(int64_t));
    pos += static_cast<int64_t>(indices[i].size());
  }
  writer.write(reinterpret_cast<const char *>(&pos), sizeof(int64_t));
  for (const auto &row : indices) {
    for (int32_t value : row) {
      writer.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));
    }
  }
  for (const auto &row : values) {
    for (float value : row) {
      writer.write(reinterpret_cast<const char *>(&value), sizeof(float));
    }
  }
}

inline uint32_t target_count(uint64_t npoints, double selectivity) {
  auto count = static_cast<uint64_t>(std::llround(static_cast<double>(npoints) * selectivity));
  return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(count, npoints)));
}

}  // namespace oh
