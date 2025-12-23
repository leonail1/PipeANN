/**
 * @file search_prefilter_streaming.cpp
 * @brief 纯顺序流式 Pre-filter 搜索
 *
 * 顺序遍历所有数据点，边读边过滤边计算距离:
 * 1. 顺序读取base数据和bitmap文件
 * 2. 对每个点检查标签是否匹配
 * 3. 匹配的点计算距离并维护top-K
 *
 * 用法:
 *   ./search_prefilter_streaming <type> <base.bin> <query.bin> <metric> <selector>
 *       <bitmap_file> <query_label.spmat> <gt.bin> <K> <runs> [threads] [batch_size]
 */

#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <queue>

#include "utils/log.h"
#include "filter/label.h"
#include "distance.h"
#include "utils.h"

struct BitmapHeader {
  uint64_t base_num;
  uint64_t num_words;
  uint64_t max_label_id;
};

void compute_robust_stats(const std::vector<double> &samples, double &avg, double &std_dev, size_t &valid_count) {
  if (samples.empty()) { avg = 0; std_dev = 0; valid_count = 0; return; }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  double q1 = sorted[n / 4], q3 = sorted[3 * n / 4];
  double iqr = q3 - q1, lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
  std::vector<double> valid;
  for (double s : samples) if (s >= lo && s <= hi) valid.push_back(s);
  valid_count = valid.size();
  if (valid_count == 0) { avg = sorted[n/2]; std_dev = 0; valid_count = 1; return; }
  avg = std::accumulate(valid.begin(), valid.end(), 0.0) / valid_count;
  double sq = 0;
  for (double s : valid) sq += (s - avg) * (s - avg);
  std_dev = valid_count > 1 ? std::sqrt(sq / (valid_count - 1)) : 0;
}

inline bool check_subset(const uint64_t *query_bitmap, const uint64_t *point_bitmap, size_t num_words) {
  for (size_t w = 0; w < num_words; w++) {
    if ((query_bitmap[w] & point_bitmap[w]) != query_bitmap[w]) return false;
  }
  return true;
}

inline bool check_intersect(const uint64_t *query_bitmap, const uint64_t *point_bitmap, size_t num_words) {
  for (size_t w = 0; w < num_words; w++) {
    if (query_bitmap[w] & point_bitmap[w]) return true;
  }
  return false;
}

template<typename T>
int search_streaming(int argc, char **argv) {
  int idx = 2;
  std::string base_file = argv[idx++];
  std::string query_file = argv[idx++];
  std::string dist_metric = argv[idx++];
  std::string selector_type = argv[idx++];
  std::string bitmap_file = argv[idx++];
  std::string query_label_file = argv[idx++];
  std::string gt_file = argv[idx++];
  uint64_t recall_at = std::atoi(argv[idx++]);
  uint32_t num_runs = std::atoi(argv[idx++]);
  int num_threads = (argc > idx) ? std::atoi(argv[idx++]) : 1;
  size_t batch_size = (argc > idx) ? std::atoi(argv[idx++]) : 200000;

  (void)num_threads;

  pipeann::Metric m = pipeann::get_metric(dist_metric);
  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(m);

  std::ifstream base_in(base_file, std::ios::binary);
  uint32_t base_num, base_dim;
  base_in.read(reinterpret_cast<char *>(&base_num), 4);
  base_in.read(reinterpret_cast<char *>(&base_dim), 4);
  LOG(INFO) << "Base: " << base_num << " x " << base_dim;

  T *query = nullptr;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  unsigned *gt_ids = nullptr; float *gt_dists = nullptr; uint32_t *tags = nullptr;
  size_t gt_num, gt_dim;
  bool calc_recall = file_exists(gt_file);
  if (calc_recall) pipeann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim, &tags);

  std::ifstream bitmap_in(bitmap_file, std::ios::binary);
  BitmapHeader bmp_hdr;
  bitmap_in.read(reinterpret_cast<char *>(&bmp_hdr), sizeof(bmp_hdr));
  LOG(INFO) << "Bitmap: base_num=" << bmp_hdr.base_num << ", num_words=" << bmp_hdr.num_words 
            << ", max_label_id=" << bmp_hdr.max_label_id;

  pipeann::SpmatLabel query_labels(query_label_file);
  std::vector<char> query_label_buf(query_labels.label_size(), 0);
  query_labels.write(0, query_label_buf.data());
  
  uint32_t label_count;
  std::memcpy(&label_count, query_label_buf.data(), sizeof(uint32_t));
  const uint32_t *label_ptr = reinterpret_cast<const uint32_t *>(query_label_buf.data() + sizeof(uint32_t));

  std::vector<uint64_t> query_bitmap(bmp_hdr.num_words, 0);
  for (uint32_t i = 0; i < label_count; i++) {
    uint32_t lbl = label_ptr[i];
    if (lbl <= bmp_hdr.max_label_id) {
      query_bitmap[lbl / 64] |= (1ULL << (lbl % 64));
    }
  }

  std::vector<float> query_float(base_dim);
  for (size_t i = 0; i < base_dim; i++) query_float[i] = static_cast<float>(query[i]);

  std::vector<uint32_t> results(recall_at);
  std::vector<double> filter_times, search_times;
  size_t filtered_count = 0;

  size_t base_row_bytes = base_dim * sizeof(T);
  size_t bitmap_row_bytes = bmp_hdr.num_words * sizeof(uint64_t);
  
  std::vector<T> batch_data(batch_size * base_dim);
  std::vector<uint64_t> batch_bitmap(batch_size * bmp_hdr.num_words);
  std::vector<float> point_float(base_dim);

  bool is_subset_mode = (selector_type == "subset");

  for (uint32_t run = 0; run <= num_runs; run++) {
    base_in.clear();
    base_in.seekg(8, std::ios::beg);
    bitmap_in.clear();
    bitmap_in.seekg(sizeof(BitmapHeader), std::ios::beg);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<uint32_t, float>> dists;
    dists.reserve(base_num / 10);
    size_t match_count = 0;

    for (size_t batch_start = 0; batch_start < base_num; batch_start += batch_size) {
      size_t batch_end = std::min(batch_start + batch_size, (size_t)base_num);
      size_t current_batch = batch_end - batch_start;

      base_in.read(reinterpret_cast<char *>(batch_data.data()), current_batch * base_row_bytes);
      bitmap_in.read(reinterpret_cast<char *>(batch_bitmap.data()), current_batch * bitmap_row_bytes);

      for (size_t i = 0; i < current_batch; i++) {
        const uint64_t *point_bmp = batch_bitmap.data() + i * bmp_hdr.num_words;
        
        bool match;
        if (is_subset_mode) {
          match = check_subset(query_bitmap.data(), point_bmp, bmp_hdr.num_words);
        } else {
          match = check_intersect(query_bitmap.data(), point_bmp, bmp_hdr.num_words);
        }

        if (match) {
          for (size_t d = 0; d < base_dim; d++) {
            point_float[d] = static_cast<float>(batch_data[i * base_dim + d]);
          }
          float dist = distance->compare(query_float.data(), point_float.data(), base_dim);
          dists.emplace_back(batch_start + i, dist);
          match_count++;
        }
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    size_t k = std::min((size_t)recall_at, dists.size());
    if (k > 0 && k < dists.size()) {
      std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                       [](auto &a, auto &b) { return a.second < b.second; });
      std::sort(dists.begin(), dists.begin() + k,
                [](auto &a, auto &b) { return a.second < b.second; });
    } else if (k > 0) {
      std::sort(dists.begin(), dists.end(),
                [](auto &a, auto &b) { return a.second < b.second; });
    }

    for (size_t i = 0; i < recall_at; i++) {
      results[i] = (i < k) ? dists[i].first : UINT32_MAX;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    if (run > 0) {
      filter_times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      search_times.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
    }
    filtered_count = match_count;
  }

  double avg_filter, std_filter, avg_search, std_search;
  size_t valid_f, valid_s;
  compute_robust_stats(filter_times, avg_filter, std_filter, valid_f);
  compute_robust_stats(search_times, avg_search, std_search, valid_s);

  float recall = 0;
  if (calc_recall) {
    recall = pipeann::calculate_recall(1, gt_ids, gt_dists, (uint32_t)gt_dim,
                                       results.data(), (uint32_t)recall_at, (uint32_t)recall_at);
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::cout << std::setw(14) << "FilterLat(us)" << std::setw(14) << "FilterStd"
            << std::setw(14) << "SearchLat(us)" << std::setw(14) << "SearchStd"
            << std::setw(14) << "TotalLat(us)" << std::setw(14) << "FilteredCnt"
            << std::setw(10) << "ValidRuns";
  if (calc_recall) std::cout << std::setw(12) << "Recall@" + std::to_string(recall_at);
  std::cout << std::endl << std::string(110, '=') << std::endl;

  std::cout << std::setw(14) << avg_filter << std::setw(14) << std_filter
            << std::setw(14) << avg_search << std::setw(14) << std_search
            << std::setw(14) << (avg_filter + avg_search) << std::setw(14) << filtered_count
            << std::setw(10) << valid_f;
  if (calc_recall) std::cout << std::setw(12) << recall;
  std::cout << std::endl;
  std::cout << "[SEQUENTIAL STREAMING mode, batch=" << batch_size << "]" << std::endl;

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cout << "Usage: " << argv[0] << " <type> <base.bin> <query.bin> <metric> <selector>\n"
              << "       <bitmap_file> <query_label.spmat> <gt.bin> <K> <runs> [threads] [batch_size]\n"
              << "\n"
              << "Sequential streaming: reads base data and bitmap sequentially,\n"
              << "filters and computes distances in a single pass.\n"
              << "Default batch_size is 200000 data points.\n";
    return 1;
  }
  std::string type = argv[1];
  if (type == "float") return search_streaming<float>(argc, argv);
  else if (type == "int8") return search_streaming<int8_t>(argc, argv);
  else if (type == "uint8") return search_streaming<uint8_t>(argc, argv);
  else { std::cout << "Unsupported type: " << type << std::endl; return 1; }
}
