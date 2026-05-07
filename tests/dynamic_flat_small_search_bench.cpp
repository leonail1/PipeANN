#include "dynamic_index.h"
#include "distance.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>

namespace {

struct Config {
  std::string base_bin = "data/sift1m/sift_base.bin";
  std::string index_prefix = "/tmp/pipeann_flat_small_search_bench";
  uint64_t points = 10000;
  uint64_t threshold = 10000;
  uint64_t k = 10;
  uint32_t threads = 1;
  uint64_t qps_repeats = 1000;
  uint64_t rss_query_id = 0;
  uint64_t query_buffer_count = 16;
};

struct BinHeader {
  uint64_t npts = 0;
  uint64_t dim = 0;
};

struct SingleQueryRss {
  uint64_t before_kb = 0;
  uint64_t after_kb = 0;
  uint64_t peak_kb = 0;
  uint64_t delta_kb = 0;
};

struct TimedSearch {
  double elapsed_s = 0.0;
  double qps = 0.0;
  double avg_latency_us = 0.0;
  double p50_latency_us = 0.0;
  double p95_latency_us = 0.0;
  double p99_latency_us = 0.0;
};

[[noreturn]] void usage(const std::string &message) {
  if (!message.empty()) {
    std::cerr << "dynamic_flat_small_search_bench: " << message << "\n";
  }
  std::cerr
      << "usage: dynamic_flat_small_search_bench --base-bin PATH --points N [--threshold N]\n"
      << "       [--index-prefix PREFIX] [--k K] [--threads N] [--qps-repeats N]\n"
      << "       [--rss-query-id ID] [--query-buffer-count N]\n";
  std::exit(2);
}

std::string require_value(int argc, char **argv, int *index, const std::string &flag) {
  if (*index + 1 >= argc) {
    usage("missing value for " + flag);
  }
  return argv[++(*index)];
}

uint64_t parse_u64(const std::string &value, const std::string &flag) {
  try {
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
      usage("invalid integer for " + flag + ": " + value);
    }
    return parsed;
  } catch (const std::exception &) {
    usage("invalid integer for " + flag + ": " + value);
  }
}

Config parse_args(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--base-bin") {
      config.base_bin = require_value(argc, argv, &i, arg);
    } else if (arg == "--index-prefix") {
      config.index_prefix = require_value(argc, argv, &i, arg);
    } else if (arg == "--points") {
      config.points = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--threshold") {
      config.threshold = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--k") {
      config.k = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--threads") {
      config.threads = static_cast<uint32_t>(parse_u64(require_value(argc, argv, &i, arg), arg));
    } else if (arg == "--qps-repeats") {
      config.qps_repeats = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--rss-query-id") {
      config.rss_query_id = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--query-buffer-count") {
      config.query_buffer_count = parse_u64(require_value(argc, argv, &i, arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      usage("");
    } else {
      usage("unknown argument: " + arg);
    }
  }
  if (config.points > config.threshold) {
    usage("--points must be <= --threshold for this flat-only experiment");
  }
  if (config.k == 0 || config.threads == 0 || config.qps_repeats == 0 || config.query_buffer_count == 0) {
    usage("--k, --threads, --qps-repeats and --query-buffer-count must be positive");
  }
  config.query_buffer_count = std::min<uint64_t>(config.query_buffer_count, 16);
  return config;
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

BinHeader read_header(std::ifstream &reader, const std::string &path) {
  int32_t npts_i32 = 0;
  int32_t dim_i32 = 0;
  reader.read(reinterpret_cast<char *>(&npts_i32), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
  if (!reader.good() || npts_i32 < 0 || dim_i32 <= 0) {
    throw std::runtime_error("failed to read valid bin header from " + path);
  }
  return {static_cast<uint64_t>(npts_i32), static_cast<uint64_t>(dim_i32)};
}

uint64_t current_rss_kb() {
  std::ifstream reader("/proc/self/statm");
  uint64_t total_pages = 0;
  uint64_t resident_pages = 0;
  reader >> total_pages >> resident_pages;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (!reader.good() || page_size <= 0) {
    return 0;
  }
  return resident_pages * static_cast<uint64_t>(page_size) / 1024ULL;
}

uint64_t max_rss_kb() {
  rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<uint64_t>(usage.ru_maxrss);
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = p * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

void assert_no_disk_files(const std::string &prefix) {
  const std::vector<std::string> suffixes = {
      "_disk.index", "_disk.index.tags", "_pq_pivots.bin", "_pq_compressed.bin", "_partition.bin.aligned",
      "_flat_build_data.bin", "_flat_build_tags.bin", "_flat_build_labels.spmat"};
  for (const std::string &suffix : suffixes) {
    require(!file_exists(prefix + suffix), "unexpected disk artifact exists: " + prefix + suffix);
  }
}

SingleQueryRss measure_single_query_rss(pipeann::DynamicSSDIndex<float, uint32_t> &index, const float *query,
                                        const Config &config) {
  std::vector<uint32_t> tags(static_cast<size_t>(config.k), std::numeric_limits<uint32_t>::max());
  std::vector<float> distances(static_cast<size_t>(config.k), std::numeric_limits<float>::infinity());
  pipeann::QueryStats stats {};

  std::atomic<bool> sample_rss {true};
  std::atomic<uint64_t> peak_rss {0};
  const uint64_t before = current_rss_kb();
  peak_rss.store(before, std::memory_order_relaxed);
  std::thread sampler([&]() {
    while (sample_rss.load(std::memory_order_relaxed)) {
      const uint64_t current = current_rss_kb();
      uint64_t previous = peak_rss.load(std::memory_order_relaxed);
      while (current > previous && !peak_rss.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  index.search(query, config.k, 0, config.k, 4, tags.data(), distances.data(), &stats);

  sample_rss.store(false, std::memory_order_relaxed);
  sampler.join();
  const uint64_t after = current_rss_kb();
  const uint64_t peak = std::max(peak_rss.load(std::memory_order_relaxed), after);
  return {before, after, peak, peak > before ? peak - before : 0};
}

TimedSearch run_qps_search(pipeann::DynamicSSDIndex<float, uint32_t> &index,
                           const std::vector<std::vector<float>> &queries, const Config &config) {
  std::vector<uint32_t> tags(static_cast<size_t>(config.qps_repeats * config.k),
                             std::numeric_limits<uint32_t>::max());
  std::vector<float> distances(static_cast<size_t>(config.qps_repeats * config.k),
                               std::numeric_limits<float>::infinity());
  std::vector<double> latencies(static_cast<size_t>(config.qps_repeats), 0.0);

  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(config.threads) schedule(static)
  for (int64_t i = 0; i < static_cast<int64_t>(config.qps_repeats); ++i) {
    pipeann::QueryStats stats {};
    const float *query = queries[static_cast<size_t>(i) % queries.size()].data();
    const auto query_start = std::chrono::steady_clock::now();
    index.search(query, config.k, 0, config.k, 4,
                 tags.data() + static_cast<size_t>(i) * static_cast<size_t>(config.k),
                 distances.data() + static_cast<size_t>(i) * static_cast<size_t>(config.k), &stats);
    const auto query_end = std::chrono::steady_clock::now();
    latencies[static_cast<size_t>(i)] =
        std::chrono::duration<double, std::micro>(query_end - query_start).count();
  }
  const auto end = std::chrono::steady_clock::now();

  TimedSearch result;
  result.elapsed_s = std::chrono::duration<double>(end - start).count();
  result.qps = static_cast<double>(config.qps_repeats) / std::max(result.elapsed_s, 1e-9);
  result.avg_latency_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  result.p50_latency_us = percentile(latencies, 0.50);
  result.p95_latency_us = percentile(latencies, 0.95);
  result.p99_latency_us = percentile(latencies, 0.99);
  return result;
}

void print_json(const Config &config, const BinHeader &header, uint64_t live_count, bool flat_mode,
                uint64_t rss_after_insert_kb, double insert_elapsed_s, const SingleQueryRss &single_rss,
                const TimedSearch &timed) {
  const double insert_qps = static_cast<double>(config.points) / std::max(insert_elapsed_s, 1e-9);
  const uint64_t process_max = std::max(max_rss_kb(), single_rss.peak_kb);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{"
            << "\"format\":\"pipeann.flat_small_search.v1\","
            << "\"dataset\":\"sift1m\","
            << "\"points\":" << config.points << ","
            << "\"dim\":" << header.dim << ","
            << "\"threshold\":" << config.threshold << ","
            << "\"flat_mode\":" << (flat_mode ? "true" : "false") << ","
            << "\"live_point_count\":" << live_count << ","
            << "\"rss_query_count\":1,"
            << "\"rss_before_single_query_kb\":" << single_rss.before_kb << ","
            << "\"rss_after_single_query_kb\":" << single_rss.after_kb << ","
            << "\"rss_single_query_peak_kb\":" << single_rss.peak_kb << ","
            << "\"rss_single_query_delta_kb\":" << single_rss.delta_kb << ","
            << "\"rss_after_insert_kb\":" << rss_after_insert_kb << ","
            << "\"process_max_rss_kb\":" << process_max << ","
            << "\"qps_repeats\":" << config.qps_repeats << ","
            << "\"elapsed_s\":" << timed.elapsed_s << ","
            << "\"qps\":" << timed.qps << ","
            << "\"avg_latency_us\":" << timed.avg_latency_us << ","
            << "\"p50_latency_us\":" << timed.p50_latency_us << ","
            << "\"p95_latency_us\":" << timed.p95_latency_us << ","
            << "\"p99_latency_us\":" << timed.p99_latency_us << ","
            << "\"insert_elapsed_s\":" << insert_elapsed_s << ","
            << "\"insert_qps\":" << insert_qps << "}"
            << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config config = parse_args(argc, argv);
    assert_no_disk_files(config.index_prefix);

    std::ifstream reader(config.base_bin, std::ios::binary);
    require(reader.good(), "failed to open base bin: " + config.base_bin);
    const BinHeader header = read_header(reader, config.base_bin);
    require(header.npts >= config.points, "base bin does not contain enough points");
    require(config.rss_query_id < header.npts, "--rss-query-id is outside base bin");

    pipeann::IndexBuildParameters params;
    params.set(64, 96, 384, 1.2f, config.threads, true, 4);
    pipeann::DistanceL2Float dist;
    pipeann::DynamicSSDIndex<float, uint32_t> index(params, config.index_prefix, static_cast<uint32_t>(header.dim),
                                                    &dist, pipeann::Metric::L2, config.threshold, PIPE_SEARCH);

    require(index.is_flat_mode(), "index did not start in flat mode");

    const uint64_t qbuf_target = std::min<uint64_t>(config.query_buffer_count, header.npts);
    const uint64_t rows_to_scan = std::max<uint64_t>(config.points, std::max<uint64_t>(config.rss_query_id + 1, qbuf_target));
    std::vector<float> row(static_cast<size_t>(header.dim));
    std::vector<std::vector<float>> query_buffers;
    query_buffers.reserve(static_cast<size_t>(qbuf_target));
    std::vector<float> rss_query;

    const auto insert_start = std::chrono::steady_clock::now();
    for (uint64_t row_id = 0; row_id < rows_to_scan; ++row_id) {
      reader.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(header.dim * sizeof(float)));
      require(reader.good(), "failed to read vector row " + std::to_string(row_id));

      if (row_id < qbuf_target) {
        query_buffers.push_back(row);
      }
      if (row_id == config.rss_query_id) {
        rss_query = row;
      }
      if (row_id < config.points) {
        const int inserted = index.insert(row.data(), static_cast<uint32_t>(row_id));
        require(inserted >= 0, "insert failed at row " + std::to_string(row_id));
      }
    }
    const auto insert_end = std::chrono::steady_clock::now();
    const double insert_elapsed_s = std::chrono::duration<double>(insert_end - insert_start).count();

    require(!rss_query.empty(), "rss query vector was not loaded");
    require(!query_buffers.empty(), "qps query buffers were not loaded");
    require(index.is_flat_mode(), "index left flat mode for points <= threshold");
    require(index.live_point_count() == config.points, "live point count mismatch");
    assert_no_disk_files(config.index_prefix);

    const uint64_t rss_after_insert_kb = current_rss_kb();
    const SingleQueryRss single_rss = measure_single_query_rss(index, rss_query.data(), config);
    const TimedSearch timed = run_qps_search(index, query_buffers, config);
    assert_no_disk_files(config.index_prefix);

    print_json(config, header, index.live_point_count(), index.is_flat_mode(), rss_after_insert_kb, insert_elapsed_s,
               single_rss, timed);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "dynamic_flat_small_search_bench: " << e.what() << std::endl;
    return 1;
  }
}
