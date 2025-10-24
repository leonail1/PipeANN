#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <queue>
#include <cblas.h>
#include <stdlib.h>

#include "omp.h"
#include "utils.h"

/**
 * @file compute_groundtruth.cpp
 * @brief 计算最近邻搜索（KNN）的 ground truth 数据
 * 
 * 该程序用于计算大规模向量数据集的精确最近邻搜索结果，支持多种数据类型（float、int8、uint8）。
 * 通过分块处理大数据集，使用 BLAS 库进行高效的矩阵运算，计算查询向量与数据集中所有向量的距离，
 * 并找出每个查询向量的前 K 个最近邻。
 * 
 * 示例调用：
 * build/tests/utils/compute_groundtruth uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/data/bigann/bigann_query.bbin 1000 /mnt/nvme/data/bigann/100M_gt.bin
 */

// 支持最多 20 亿个数据点（使用 int 而不是 unsigned）
#define PARTSIZE 10000000  ///< 每个分块的大小（1000万个向量）
#define ALIGNMENT 512      ///< 内存对齐大小

/**
 * @brief 显示命令行帮助信息
 * 
 * 打印程序的正确使用方法，包括参数说明和格式。
 */
void command_line_help() {
  std::cerr << "<exact-kann> <int8/uint8/float>   <base bin file> <query bin "
               "file>  <K: # nearest neighbors to compute> "
               "<output-truthset-file> optional:<tag_file>"
            << std::endl;
}

/**
 * @brief 向上取整的除法函数
 * @tparam T 数值类型
 * @param numerator 分子
 * @param denominator 分母
 * @return 向上取整的除法结果
 * 
 * 例如：div_round_up(10, 3) = 4，div_round_up(9, 3) = 3
 */
template<class T>
T div_round_up(const T numerator, const T denominator) {
  return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

using pairIF = std::pair<int, float>;  ///< 整数-浮点数对类型，用于存储索引和距离

/**
 * @brief 最大堆比较器结构体
 * 
 * 用于优先队列，维护一个最大堆，堆顶元素是当前最大的距离值。
 */
struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    return l.second < r.second;  // 按距离值降序排列
  };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;  ///< 最大优先队列类型

/**
 * @brief 对齐内存分配函数
 * @tparam T 数据类型
 * @param n 元素数量
 * @param alignment 对齐大小
 * @return 对齐分配的内存指针
 * 
 * 使用 aligned_alloc 分配对齐内存，提高内存访问效率。
 */
template<class T>
T *aligned_malloc(const size_t n, const size_t alignment) {
  return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
}

/**
 * @brief 自定义距离比较函数
 * @param a 第一个索引-距离对
 * @param b 第二个索引-距离对
 * @return 如果 a 的距离小于 b 的距离则返回 true
 * 
 * 用于 std::sort 对结果按距离升序排序。
 */
inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
  return a.second < b.second;
}

/**
 * @brief 计算向量 L2 范数的平方
 * @param points_l2sq 输出数组，存储每个向量的 L2 范数平方
 * @param matrix 输入矩阵（列主序）
 * @param num_points 向量数量
 * @param dim 向量维度
 * 
 * 使用 BLAS 的 sdot 函数计算每个向量与自身的点积，即 L2 范数的平方。
 * 使用 OpenMP 并行化计算以提高性能。
 */
void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const int dim) {
  assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
  for (int64_t d = 0; d < num_points; ++d)
    points_l2sq[d] =
        cblas_sdot(dim, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1);
}

/**
 * @brief 计算查询向量与数据点之间的平方欧氏距离
 * @param dim 向量维度
 * @param dist_matrix 距离矩阵（列主序，列为查询，行为数据点）
 * @param npoints 数据点数量
 * @param points 数据点矩阵（列主序）
 * @param points_l2sq 数据点的 L2 范数平方数组
 * @param nqueries 查询向量数量
 * @param queries 查询向量矩阵（列主序）
 * @param queries_l2sq 查询向量的 L2 范数平方数组
 * @param ones_vec 全1向量（临时空间），如果为NULL则自动分配
 * 
 * 使用矩阵运算高效计算平方欧氏距离：dist² = ||p||² + ||q||² - 2p·q
 * 通过三次 BLAS 矩阵乘法实现：
 * 1. -2 * points^T * queries
 * 2. + points_l2sq * ones_vec^T
 * 3. + ones_vec * queries_l2sq^T
 */
void distsq_to_points(const size_t dim,
                      float *dist_matrix,  // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq,  // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq,  // queries in Col major
                      float *ones_vec = NULL)           // Scratchspace of num_data size and init to 1.0
{
  bool ones_vec_alloc = false;
  if (ones_vec == NULL) {
    ones_vec = new float[nqueries > npoints ? nqueries : npoints];
    std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
    ones_vec_alloc = true;
  }
  
  // 计算 -2 * points^T * queries
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float) -2.0, points, dim, queries, dim,
              (float) 0.0, dist_matrix, npoints);
  
  // 加上 points_l2sq * ones_vec^T
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, points_l2sq, npoints,
              ones_vec, nqueries, (float) 1.0, dist_matrix, npoints);
  
  // 加上 ones_vec * queries_l2sq^T
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, ones_vec, npoints,
              queries_l2sq, nqueries, (float) 1.0, dist_matrix, npoints);
  
  if (ones_vec_alloc)
    delete[] ones_vec;
}

/**
 * @brief 精确 K 最近邻搜索算法
 * @param dim 向量维度
 * @param k 最近邻数量
 * @param closest_points 输出数组，存储每个查询的最近邻索引（列主序，k * num_queries）
 * @param dist_closest_points 输出数组，存储每个最近邻的距离（列主序，k * num_queries）
 * @param npoints 数据点数量
 * @param points 数据点矩阵（列主序）
 * @param nqueries 查询向量数量
 * @param queries 查询向量矩阵（列主序）
 * 
 * 实现精确的 K 最近邻搜索算法：
 * 1. 计算数据点和查询向量的 L2 范数平方
 * 2. 分批处理查询向量以减少内存使用
 * 3. 对每个查询向量使用最大堆维护前 K 个最近邻
 * 4. 使用 OpenMP 并行化查询处理
 */
void exact_knn(const size_t dim, const size_t k,
               int *const closest_points,         // k * num_queries preallocated, col
                                                  // major, queries columns
               float *const dist_closest_points,  // k * num_queries
                                                  // preallocated, Dist to
                                                  // corresponding closes_points
               size_t npoints,
               const float *const points,  // points in Col major
               size_t nqueries,
               const float *const queries)  // queries in Col major
{
  // 计算数据点和查询向量的 L2 范数平方
  float *points_l2sq = new float[npoints];
  float *queries_l2sq = new float[nqueries];
  compute_l2sq(points_l2sq, points, npoints, dim);
  compute_l2sq(queries_l2sq, queries, nqueries, dim);

  // 设置查询批次大小（512个查询一批）
  size_t q_batch_size = (1 << 9);
  float *dist_matrix = new float[(size_t) q_batch_size * (size_t) npoints];

  // 分批处理查询向量
  for (uint64_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b) {
    int64_t q_b = b * q_batch_size;
    int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

    // 计算当前批次查询向量与所有数据点的距离
    distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                     queries + (ptrdiff_t) q_b * (ptrdiff_t) dim, queries_l2sq + q_b);
    std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

    // 并行处理每个查询向量
#pragma omp parallel for schedule(dynamic, 16)
    for (long long q = q_b; q < q_e; q++) {
      maxPQIFCS point_dist;  // 最大堆，用于维护前 K 个最近邻
      
      // 初始化堆，放入前 K 个数据点
      for (uint64_t p = 0; p < k; p++)
        point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
      
      // 处理剩余的数据点
      for (uint64_t p = k; p < npoints; p++) {
        // 如果当前数据点距离更小，则加入堆
        if (point_dist.top().second > dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints])
          point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
        // 保持堆大小为 K
        if (point_dist.size() > k)
          point_dist.pop();
      }
      
      // 从堆中提取结果（按距离升序排列）
      for (ptrdiff_t l = 0; l < (ptrdiff_t) k; ++l) {
        closest_points[(ptrdiff_t) (k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().first;
        dist_closest_points[(ptrdiff_t) (k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().second;
        point_dist.pop();
      }
      
      // 验证结果是否按距离升序排列
      assert(std::is_sorted(dist_closest_points + (ptrdiff_t) q * (ptrdiff_t) k,
                            dist_closest_points + (ptrdiff_t) (q + 1) * (ptrdiff_t) k));
    }
    std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
  }

  // 释放内存
  delete[] dist_matrix;
  delete[] points_l2sq;
  delete[] queries_l2sq;
}

/**
 * @brief 获取数据集的分块数量
 * @tparam T 数据类型
 * @param filename 二进制文件名
 * @return 数据集需要分成的块数
 * 
 * 读取二进制文件的头部信息，根据总数据点数和分块大小计算需要分成多少块。
 * 二进制文件格式：前两个 int 分别是数据点数量和维度。
 */
template<typename T>
inline int get_num_parts(const char *filename) {
  std::ifstream reader(filename, std::ios::binary);
  std::cout << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  std::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
  reader.close();
  
  // 计算分块数量
  uint32_t num_parts = (npts_i32 % PARTSIZE) == 0 ? (uint32_t) (npts_i32 / PARTSIZE)
                                                  : (uint32_t) std::floor((double) npts_i32 / (double) PARTSIZE) + 1;
  std::cout << "Number of parts: " << num_parts << std::endl;
  return num_parts;
}

/**
 * @brief 加载二进制文件的一个分块并转换为浮点数格式
 * @tparam T 原始数据类型（int8/uint8/float）
 * @param filename 二进制文件名
 * @param data 输出参数，指向转换后的浮点数数据
 * @param npts 输出参数，当前分块的数据点数量
 * @param ndims 输出参数，向量维度
 * @param part_num 分块编号（从0开始）
 * 
 * 读取二进制文件的指定分块，将数据转换为浮点数格式，并使用对齐内存存储。
 * 支持多种原始数据类型到浮点数的转换。
 */
template<typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims, int part_num) {
  std::ifstream reader(filename, std::ios::binary);
  std::cout << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  
  // 计算当前分块的起始和结束位置
  uint64_t start_id = part_num * PARTSIZE;
  uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t) npts_i32);
  npts = end_id - start_id;
  ndims = (unsigned) ndims_i32;
  uint64_t nptsuint64_t = (uint64_t) npts;
  uint64_t ndimsuint64_t = (uint64_t) ndims;
  std::cout << "#pts in part = " << npts << ", #dims = " << ndims
            << ", size = " << nptsuint64_t * ndimsuint64_t * sizeof(T) << "B" << std::endl;

  // 定位到当前分块的起始位置
  reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
  
  // 读取原始数据
  T *data_T = new T[nptsuint64_t * ndimsuint64_t];
  reader.read((char *) data_T, sizeof(T) * nptsuint64_t * ndimsuint64_t);
  std::cout << "Finished reading part of the bin file." << std::endl;
  reader.close();
  
  // 分配对齐内存并转换为浮点数
  data = aligned_malloc<float>(nptsuint64_t * ndimsuint64_t, ALIGNMENT);
  
  // 并行转换数据为浮点数格式
#pragma omp parallel for schedule(dynamic, 32768)
  for (int64_t i = 0; i < (int64_t) nptsuint64_t; i++) {
    for (int64_t j = 0; j < (int64_t) ndimsuint64_t; j++) {
      float cur_val_float = (float) data_T[i * ndimsuint64_t + j];
      std::memcpy((char *) (data + i * ndimsuint64_t + j), (char *) &cur_val_float, sizeof(float));
    }
  }
  
  delete[] data_T;
  std::cout << "Finished converting part data to float." << std::endl;
}

/**
 * @brief 保存数据到二进制文件
 * @tparam T 数据类型
 * @param filename 输出文件名
 * @param data 数据指针
 * @param npts 数据点数量
 * @param ndims 向量维度
 * 
 * 将数据保存为二进制格式，文件格式：
 * - 前两个 int：数据点数量、向量维度
 * - 后续数据：npts * ndims 个 T 类型数据
 */
template<typename T>
inline void save_bin(const std::string filename, T *data, size_t npts, size_t ndims) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  std::cout << "Writing bin: " << filename << "\n";
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
            << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int) << "B" << std::endl;

  //    data = new T[npts_u64 * ndims_u64];
  writer.write((char *) data, npts * ndims * sizeof(T));
  writer.close();
  std::cout << "Finished writing bin" << std::endl;
}

/**
 * @brief 保存 ground truth 数据到单个文件
 * @param filename 输出文件名
 * @param data 最近邻索引数组
 * @param distances 最近邻距离数组
 * @param npts 查询向量数量
 * @param ndims 最近邻数量（K值）
 * @param tags 标签数组（可选）
 * 
 * 将 ground truth 数据保存为二进制格式，文件包含：
 * - 头部：数据点数量、最近邻数量
 * - 最近邻索引矩阵（npts * ndims 个 uint32_t）
 * - 最近邻距离矩阵（npts * ndims 个 float）
 * - 标签矩阵（可选，npts * ndims 个 uint32_t）
 */
inline void save_groundtruth_as_one_file(const std::string filename, int32_t *data, float *distances, size_t npts,
                                         size_t ndims, uint32_t *tags = nullptr) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
               "npts*dim dist-matrix) with npts = "
            << npts << ", dim = " << ndims << ", size = " << 2 * npts * ndims * sizeof(unsigned) + 2 * sizeof(int)
            << "B" << std::endl;

  //    data = new T[npts_u64 * ndims_u64];
  writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  writer.write((char *) distances, npts * ndims * sizeof(float));
  if (tags != nullptr) {
    writer.write((char *) tags, npts * ndims * sizeof(uint32_t));
  } else {
    writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  }

  writer.close();
  std::cout << "Finished writing truthset" << std::endl;
}

/**
 * @brief 辅助主函数，处理特定数据类型的 ground truth 计算
 * @tparam T 数据类型（float/int8/uint8）
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序退出码
 * 
 * 主要处理流程：
 * 1. 解析命令行参数
 * 2. 加载查询数据
 * 3. 分块处理基础数据集
 * 4. 对每个分块执行精确 KNN 搜索
 * 5. 合并所有分块的结果
 * 6. 保存最终的 ground truth 数据
 */
template<typename T>
int aux_main(int argc, char **argv) {
  size_t npoints, nqueries, dim;
  std::string base_file(argv[2]);
  std::string query_file(argv[3]);
  size_t k = atoi(argv[4]);
  std::string gt_file(argv[5]);

  float *base_data;
  float *query_data;

  // 获取数据集分块数量并加载查询数据
  int num_parts = get_num_parts<T>(base_file.c_str());
  load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);

  // 初始化结果存储结构
  std::vector<std::vector<std::pair<uint32_t, float>>> results(nqueries);
  int *closest_points = new int[nqueries * k];
  float *dist_closest_points = new float[nqueries * k];

  // 分块处理基础数据集
  for (int p = 0; p < num_parts; p++) {
    size_t start_id = p * PARTSIZE;
    load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);
    int *closest_points_part = new int[nqueries * k];
    float *dist_closest_points_part = new float[nqueries * k];

    // 对当前分块执行精确 KNN 搜索
    exact_knn(dim, k, closest_points_part, dist_closest_points_part, npoints, base_data, nqueries, query_data);

    // 将当前分块的结果合并到总结果中（考虑全局索引）
    for (uint64_t i = 0; i < nqueries; i++) {
      for (uint64_t j = 0; j < k; j++) {
        results[i].push_back(std::make_pair((uint32_t) (closest_points_part[i * k + j] + start_id),
                                            dist_closest_points_part[i * k + j]));
      }
    }

    delete[] closest_points_part;
    delete[] dist_closest_points_part;
    pipeann::aligned_free(base_data);
  }

  // 对每个查询向量的结果进行排序和截断
  for (uint64_t i = 0; i < nqueries; i++) {
    std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
    std::sort(cur_res.begin(), cur_res.end(), custom_dist);
    for (uint64_t j = 0; j < k; j++) {
      closest_points[i * k + j] = (int32_t) cur_res[j].first;
      dist_closest_points[i * k + j] = cur_res[j].second;
    }
  }
  
  // 处理标签数据（如果提供了标签文件）
  uint32_t *tags = nullptr;
  if (argc == 7) {
    std::cout << "Loading tags from " << argv[6] << "\n";
    tags = new uint32_t[nqueries * k];
    uint32_t *all_tags;
    std::string tag_file = std::string(argv[6]);
    size_t tag_pts, tag_dim;
    pipeann::load_bin(tag_file, all_tags, tag_pts, tag_dim);

    std::cout << "Loaded tags for " << tag_pts << " points.\n";
    for (uint64_t i = 0; i < nqueries * k; i++) {
      tags[i] = all_tags[closest_points[i]];
    }
  }

  // 保存最终的 ground truth 数据
  save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points, nqueries, k, tags);
  
  // 清理内存
  pipeann::aligned_free(query_data);
  delete[] closest_points;
  delete[] dist_closest_points;
  if (tags != nullptr) {
    delete[] tags;
  }

  return 0;
}

/**
 * @brief 主函数，程序入口点
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序退出码
 * 
 * 命令行参数格式：
 * <exact-kann> <int8/uint8/float> <base bin file> <query bin file> <K> <output-truthset-file> [tag_file]
 * 
 * 示例调用：
 * build/tests/utils/compute_groundtruth uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/data/bigann/bigann_query.bbin 1000 /mnt/nvme/data/bigann/100M_gt.bin
 */
int main(int argc, char **argv) {
  // 检查命令行参数数量
  if (argc != 6 && argc != 7) {
    command_line_help();
    return -1;
  }
  
  // 根据数据类型调用相应的处理函数
  if (std::string(argv[1]) == std::string("float"))
    aux_main<float>(argc, argv);
  if (std::string(argv[1]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  if (std::string(argv[1]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
}
