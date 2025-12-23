/**
 * @file bench_exact_knn.cpp
 * @brief 精确KNN搜索性能基准测试程序
 * 
 * 本程序用于测量单个查询向量对不同规模数据集进行精确K近邻(KNN)搜索的耗时。
 * 程序会以指定步长逐步增加数据集大小，记录每次搜索的耗时，直到单次搜索时间超过阈值。
 * 
 * 特性:
 * - 每个数据规模多次运行，使用IQR方法去除异常值
 * - 自动调整运行次数，确保有足够的有效样本
 * - 输出CSV格式数据，可用于绘制曲线图
 * 
 * 使用方法:
 *   ./bench_exact_knn [维度] [K值] [步长] [线程数]
 *   例如: ./bench_exact_knn 128 10 10000 8
 * 
 * 输出格式:
 *   npoints,avg_time_ms,std_dev,valid_samples,total_samples
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
#include <random>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <cblas.h>

#include "omp.h"
#include "utils.h"
#include "distance.h"
#include "utils/kmeans_utils.h"

#define ALIGNMENT 512

using pairIF = std::pair<int, float>;

struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    return l.second < r.second;
  };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

/**
 * @brief 使用IQR方法去除异常值并计算统计量
 * 
 * IQR (Interquartile Range) 方法:
 * 1. 计算Q1(第25百分位)和Q3(第75百分位)
 * 2. IQR = Q3 - Q1
 * 3. 有效范围: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
 * 4. 超出范围的值视为异常值
 * 
 * @param samples       输入样本数组
 * @param avg           输出: 去除异常值后的平均值
 * @param std_dev       输出: 去除异常值后的标准差
 * @param valid_count   输出: 有效样本数量
 */
void compute_robust_stats(const std::vector<double>& samples, 
                          double& avg, double& std_dev, size_t& valid_count) {
  if (samples.empty()) {
    avg = 0;
    std_dev = 0;
    valid_count = 0;
    return;
  }

  // 复制并排序样本
  std::vector<double> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end());

  size_t n = sorted_samples.size();
  
  // 计算Q1和Q3（使用线性插值）
  double q1_pos = 0.25 * (n - 1);
  double q3_pos = 0.75 * (n - 1);
  
  size_t q1_idx = static_cast<size_t>(q1_pos);
  size_t q3_idx = static_cast<size_t>(q3_pos);
  
  double q1 = sorted_samples[q1_idx];
  double q3 = sorted_samples[q3_idx];
  
  // 线性插值
  if (q1_idx + 1 < n) {
    double frac = q1_pos - q1_idx;
    q1 = sorted_samples[q1_idx] * (1 - frac) + sorted_samples[q1_idx + 1] * frac;
  }
  if (q3_idx + 1 < n) {
    double frac = q3_pos - q3_idx;
    q3 = sorted_samples[q3_idx] * (1 - frac) + sorted_samples[q3_idx + 1] * frac;
  }

  // 计算IQR和有效范围
  double iqr = q3 - q1;
  double lower_bound = q1 - 1.5 * iqr;
  double upper_bound = q3 + 1.5 * iqr;

  // 过滤异常值并计算统计量
  std::vector<double> valid_samples;
  for (double s : samples) {
    if (s >= lower_bound && s <= upper_bound) {
      valid_samples.push_back(s);
    }
  }

  valid_count = valid_samples.size();
  
  if (valid_count == 0) {
    // 如果全部被过滤，使用中位数
    avg = sorted_samples[n / 2];
    std_dev = 0;
    valid_count = 1;
    return;
  }

  // 计算平均值
  avg = std::accumulate(valid_samples.begin(), valid_samples.end(), 0.0) / valid_count;

  // 计算标准差
  double sq_sum = 0;
  for (double s : valid_samples) {
    sq_sum += (s - avg) * (s - avg);
  }
  std_dev = valid_count > 1 ? std::sqrt(sq_sum / (valid_count - 1)) : 0;
}

void distsq_to_points(const size_t dim, float *dist_matrix, size_t npoints,
                      const float *const points, const float *const points_l2sq,
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq) {
  float *ones_vec = new float[nqueries > npoints ? nqueries : npoints];
  std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, 1.0f);

  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim,
              -2.0f, points, dim, queries, dim, 0.0f, dist_matrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1,
              1.0f, points_l2sq, npoints, ones_vec, nqueries, 1.0f, dist_matrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1,
              1.0f, ones_vec, npoints, queries_l2sq, nqueries, 1.0f, dist_matrix, npoints);

  delete[] ones_vec;
}

double exact_knn_timed(const size_t dim, const size_t k, int *closest_points,
                       float *dist_closest_points, size_t npoints, float *points,
                       size_t nqueries, float *queries) {
  float *points_l2sq = new float[npoints];
  float *queries_l2sq = new float[nqueries];
  kmeans::compute_vecs_l2sq(points_l2sq, points, npoints, dim);
  kmeans::compute_vecs_l2sq(queries_l2sq, queries, nqueries, dim);

  float *dist_matrix = new float[nqueries * npoints];

  auto start = std::chrono::high_resolution_clock::now();

  distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, nqueries,
                   queries, queries_l2sq);

  for (size_t q = 0; q < nqueries; q++) {
    maxPQIFCS point_dist;
    for (size_t p = 0; p < k; p++)
      point_dist.emplace(p, dist_matrix[p + q * npoints]);
    for (size_t p = k; p < npoints; p++) {
      if (point_dist.top().second > dist_matrix[p + q * npoints])
        point_dist.emplace(p, dist_matrix[p + q * npoints]);
      if (point_dist.size() > k)
        point_dist.pop();
    }
    for (int64_t l = k - 1; l >= 0; --l) {
      closest_points[l + q * k] = point_dist.top().first;
      dist_closest_points[l + q * k] = point_dist.top().second;
      point_dist.pop();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

  delete[] dist_matrix;
  delete[] points_l2sq;
  delete[] queries_l2sq;

  return elapsed_ms;
}

/**
 * @brief 对指定数据规模进行多次测试，返回稳健的统计结果
 * 
 * 算法:
 * 1. 初始运行min_runs次
 * 2. 使用IQR方法去除异常值
 * 3. 如果有效样本数 < min_valid_samples，继续运行更多次
 * 4. 最多运行max_runs次
 * 
 * @param dim             向量维度
 * @param k               K值
 * @param npoints         数据点数量
 * @param points          数据点数组
 * @param query           查询向量
 * @param closest         结果缓冲区
 * @param distances       距离缓冲区
 * @param min_runs        最小运行次数
 * @param min_valid_samples 最小有效样本数
 * @param max_runs        最大运行次数
 * @param avg_time        输出: 平均耗时
 * @param std_dev         输出: 标准差
 * @param valid_count     输出: 有效样本数
 * @param total_count     输出: 总运行次数
 */
void benchmark_with_outlier_removal(
    size_t dim, size_t k, size_t npoints, float* points, float* query,
    int* closest, float* distances,
    size_t min_runs, size_t min_valid_samples, size_t max_runs,
    double& avg_time, double& std_dev, size_t& valid_count, size_t& total_count) {
  
  std::vector<double> samples;
  samples.reserve(max_runs);
  
  // 预热运行（不计入统计）
  exact_knn_timed(dim, k, closest, distances, npoints, points, 1, query);
  
  // 初始运行
  for (size_t i = 0; i < min_runs; i++) {
    double t = exact_knn_timed(dim, k, closest, distances, npoints, points, 1, query);
    samples.push_back(t);
  }
  
  // 计算统计量并检查是否需要更多样本
  compute_robust_stats(samples, avg_time, std_dev, valid_count);
  
  // 如果有效样本不足，继续运行
  while (valid_count < min_valid_samples && samples.size() < max_runs) {
    // 每次额外运行5次
    size_t extra_runs = std::min((size_t)5, max_runs - samples.size());
    for (size_t i = 0; i < extra_runs; i++) {
      double t = exact_knn_timed(dim, k, closest, distances, npoints, points, 1, query);
      samples.push_back(t);
    }
    compute_robust_stats(samples, avg_time, std_dev, valid_count);
  }
  
  total_count = samples.size();
}

int main(int argc, char **argv) {
  // ========== 参数配置 ==========
  size_t dim = 128;
  size_t k = 10;
  size_t step = 10000;            // 步长改为10000
  double max_time_ms = 50.0;     // 阈值改为100ms
  int num_threads = omp_get_max_threads();  // 默认使用最大线程数
  
  // 稳健统计参数
  size_t min_runs = 10;           // 每个数据规模最少运行10次
  size_t min_valid_samples = 5;   // 至少需要5个有效样本
  size_t max_runs = 50;           // 最多运行50次

  if (argc >= 2) dim = std::stoul(argv[1]);
  if (argc >= 3) k = std::stoul(argv[2]);
  if (argc >= 4) step = std::stoul(argv[3]);
  if (argc >= 5) num_threads = std::stoi(argv[4]);

  omp_set_num_threads(num_threads);

  if (num_threads == 1) {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
  }

  std::string output_filename = std::string(argv[0]);
  size_t last_slash = output_filename.find_last_of("/\\");
  std::string output_dir = (last_slash != std::string::npos) ? output_filename.substr(0, last_slash + 1) : "./";
  output_filename = output_dir + "bench_exact_knn_threads_" + std::to_string(num_threads) + ".csv";

  std::ofstream csv_file(output_filename);
  if (!csv_file.is_open()) {
    std::cerr << "Error: cannot open output file " << output_filename << std::endl;
    return 1;
  }

  std::cerr << "Benchmarking exact KNN: dim=" << dim << ", k=" << k 
            << ", step=" << step << ", max_time=" << max_time_ms << "ms"
            << ", threads=" << num_threads << std::endl;
  std::cerr << "Output file: " << output_filename << std::endl;
  std::cerr << "Using IQR method for outlier removal, min_runs=" << min_runs 
            << ", min_valid=" << min_valid_samples << ", max_runs=" << max_runs << std::endl;
  
  // CSV表头
  csv_file << "npoints,avg_time_ms,std_dev,valid_samples,total_samples" << std::endl;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float *query;
  pipeann::alloc_aligned((void **)&query, dim * sizeof(float), ALIGNMENT);
  for (size_t i = 0; i < dim; i++) query[i] = dist(rng);

  int *closest = new int[k];
  float *distances = new float[k];

  size_t max_npoints = 100000000;
  float *points;
  pipeann::alloc_aligned((void **)&points, max_npoints * dim * sizeof(float), ALIGNMENT);

  size_t generated = 0;

  for (size_t npoints = step; ; npoints += step) {
    // 增量生成随机数据点
    while (generated < npoints && generated < max_npoints) {
      for (size_t d = 0; d < dim; d++) {
        points[generated * dim + d] = dist(rng);
      }
      generated++;
    }

    if (npoints > max_npoints) break;

    // 执行多次测试并计算稳健统计量
    double avg_time, std_dev;
    size_t valid_count, total_count;
    
    benchmark_with_outlier_removal(
        dim, k, npoints, points, query, closest, distances,
        min_runs, min_valid_samples, max_runs,
        avg_time, std_dev, valid_count, total_count);
    
    // 输出CSV格式结果
    csv_file << npoints << "," 
              << avg_time << "," 
              << std_dev << ","
              << valid_count << ","
              << total_count << std::endl;

    // 平均耗时超过阈值则停止
    if (avg_time > max_time_ms) {
      std::cerr << "Reached avg=" << avg_time << "ms > " << max_time_ms 
                << "ms, stopping." << std::endl;
      break;
    }
  }

  delete[] closest;
  delete[] distances;
  pipeann::aligned_free(query);
  pipeann::aligned_free(points);
  csv_file.close();

  std::cerr << "Results saved to: " << output_filename << std::endl;

  return 0;
}