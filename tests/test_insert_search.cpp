// 动态SSD索引相关头文件
#include "ssd_index.h"
#include "v2/dynamic_index.h"

// 系统和标准库头文件
#include <index.h>
#include <cstddef>
#include <future>
#include <numeric>
#include <omp.h>        // OpenMP并行计算
#include <string.h>
#include <time.h>
#include "utils/timer.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

// PipeANN核心功能头文件
#include "aux_utils.h"
#include "utils.h"
#include "utils/log.h"

// 内存映射和系统头文件
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// 全局配置参数
int NUM_INSERT_THREADS = 10;   // 插入操作的并行线程数
int NUM_SEARCH_THREADS = 32;   // 搜索操作的并行线程数
int search_mode = BEAM_SEARCH; // 搜索模式（使用beam search）

// 全局计时变量
int begin_time = 0;             // 开始时间记录
pipeann::Timer globalTimer;    // 全局计时器

/**
 * 显示内存状态和磁盘索引大小
 * @param filename 索引文件前缀名
 * 功能：输出当前内存使用情况（RSS）和磁盘索引文件大小
 */
void ShowMemoryStatus(const std::string &filename) {
  // 计算从开始到当前经过的时间（毫秒）
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  // 从/proc/self/statm读取进程内存使用情况
  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;  // tSize:虚拟内存页数, resident:常驻内存页数, share:共享内存页数
  buffer.close();

  // 获取系统页面大小（KB），处理x86-64可能使用2MB页面的情况
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
  double rss = resident * page_size_kb;  // 计算实际内存使用量（KB）

  // 获取磁盘索引文件大小
  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  // 输出内存状态和索引文件大小
  LOG(INFO) << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB";
}

/**
 * 将浮点数转换为字符串（��持指定精度）
 * @param value 要转换的浮点数值
 * @param precision 小数点后精度（默认为0）
 * @return 转换后的字符串
 */
std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

/**
 * 获取真值文件名
 * @param truthFilePrefix 真值文件目录前缀
 * @param l_start 起始索引值
 * @return 完整的真值文件路径（格式：prefix/gt_l_start.bin）
 */
std::string GetTruthFileName(const std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

/**
 * 将搜索结果保存到二进制文件中（测试用）
 * @tparam T ID的数据类型
 * @param filename 输出文件名
 * @param id 搜索结果ID数组
 * @param dist 搜索结果距离数组
 * @param npts 查询数量
 * @param ndims 每个查询的结果数量
 * @param offset 文件偏移量（默认为0）
 * @return 写入的字节数
 */
template<typename T>
inline uint64_t save_bin_test(const std::string &filename, T *id, float *dist, size_t npts, size_t ndims,
                              size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  LOG(INFO) << "Writing bin: " << filename.c_str();
  writer.seekp(offset, writer.beg);

  // 将size_t转换为int写入文件头
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);

  // 写入文件头：查询数量和维度
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

  // 写入搜索结果：ID和距离交替写入
  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char *) (id + i * ndims + j), sizeof(T));      // 写入ID
      writer.write((char *) (dist + i * ndims + j), sizeof(float)); // 写入距离
    }
  }
  writer.close();
  LOG(INFO) << "Finished writing bin.";
  return bytes_written;
}

/**
 * 同步搜索核心函数
 * @tparam T 向量数据类型
 * @tparam TagT 标签数据类型
 * @param query 查询向量数组
 * @param query_num 查询向量数量
 * @param query_dim 查询向量维度
 * @param recall_at 召回率@K中的K值
 * @param mem_L 内存搜索的候选列表大小
 * @param L 磁盘搜索的候选列表大小
 * @param beam_width beam search的宽度
 * @param sync_index 动态SSD索引对象
 * @param truthset_file 真值集文件路径
 * @param merged 是否已合并索引（未使用）
 * @param calRecall 是否计算召回率
 * @param disk_io 输出磁盘I/O次数
 */
template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, uint32_t mem_L, uint64_t L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  // 如果没有搜索线程，直接返回
  if (NUM_SEARCH_THREADS == 0) {
    return;
  }

  // 真值集变量
  unsigned *gt_ids = NULL;      // 真值ID数组
  float *gt_dists = NULL;       // 真值距离数组
  size_t gt_num, gt_dim;        // 真值集的数量和维度

  // 如果需要计算召回率，则加载真值集
  if (calRecall) {
    LOG(INFO) << "current truthfile: " << truthset_file;
    uint32_t* gt_tags = nullptr;
    size_t rss_before_gt = get_current_rss();
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    size_t rss_after_gt = get_current_rss();

    // 计算真值集的理论内存占用
    size_t gt_theoretical = (gt_num * gt_dim * (sizeof(unsigned) + sizeof(float))) / 1024.0;
    if (gt_tags != nullptr) {
      gt_theoretical += (gt_num * gt_dim * sizeof(uint32_t)) / 1024.0;
    }
    LOG(DEBUG) << "加载真值集 - gt_ids 大小: " << gt_num << " x " << gt_dim << ", 单元素大小: " << sizeof(unsigned)
               << " bytes"
               << ", 理论大小: " << gt_theoretical << " KB"
               << ", 增长前: " << rss_before_gt << " KB"
               << ", 增长后: " << rss_after_gt << " KB"
               << ", 实际增长: " << (rss_after_gt - rss_before_gt) << " KB";
  }

  // 分配搜索结果数组并监控内存使用
  size_t rss_before_results = get_current_rss();
  float *query_result_dists = new float[recall_at * query_num];    // 搜索结果距离数组
  TagT *query_result_tags = new TagT[recall_at * query_num];      // 搜索结果标签数组
  size_t rss_after_results = get_current_rss();
  LOG(DEBUG) << "分配搜索结果数组 - 理论大小: " << (recall_at * query_num * (sizeof(float) + sizeof(TagT)) / 1024.0) << " KB"
             << ", 增长前: " << rss_before_results << " KB"
             << ", 增长后: " << rss_after_results << " KB"
             << ", 实际增长: " << (rss_after_results - rss_before_results) << " KB";

  // 初始化搜索结果数组为最大值
  for (uint32_t q = 0; q < query_num; q++) {
    for (uint32_t r = 0; r < (uint32_t) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();  // 标签设为最大值
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max(); // 距离设为最大值
    }
  }

  // 初始化延迟统计和查询统计数组
  std::vector<double> latency_stats(query_num, 0);     // 延迟统计数组
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num]; // 查询统计数组
  std::string recall_string = "Recall@" + std::to_string(recall_at);

  // 输出性能指标表头
  std::cerr << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cerr << "==============================================================="
               "==============="
            << std::endl;

  // 开始计时搜索性能
  auto s = std::chrono::high_resolution_clock::now();
// 使用OpenMP并行执行搜索查询
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    // 执行单个查询的近似最近邻搜索
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    // 记录查询延迟（微秒转换为毫秒）
    latency_stats[i] = stats[i].total_us / 1000.0;

    // 如果使用beam search模式，添加短暂延迟（遵循原始论文设置）
    if (search_mode == BEAM_SEARCH) {
      // 原论文设置：对于PipeSearch，不sleep更快，但这里为了保持一致性使用了sleep
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  auto e = std::chrono::high_resolution_clock::now();  // 结束计时

  // 计算搜索总耗时和QPS（每秒查询数）
  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());  // 计算QPS
  float recall = 0;

  // 获取当前时间戳
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  // 如果需要计算召回率，则计算并清理真值集内存
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;    // 释放真值ID数组内存
    delete[] gt_dists;  // 释放真值距离数组内存
  }

  // 计算平均磁盘I/O次数
  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

  // 对延迟统计进行排序，用于计算百分位数
  std::sort(latency_stats.begin(), latency_stats.end());

  // 输出性能统计结果（包括延迟分布、QPS、召回率和磁盘I/O）
  std::cerr << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num  // 平均延迟
            << std::setw(12) << (float) latency_stats[(uint64_t) (0.50 * ((double) query_num))] << std::setw(12)  // 50%分位数
            << (float) latency_stats[(uint64_t) (0.90 * ((double) query_num))] << std::setw(12)                 // 90%分位数
            << (float) latency_stats[(uint64_t) (0.95 * ((double) query_num))] << std::setw(12)                 // 95%分位数
            << (float) latency_stats[(uint64_t) (0.99 * ((double) query_num))] << std::setw(12)                 // 99%分位数
            << (float) latency_stats[(uint64_t) (0.999 * ((double) query_num))] << std::setw(12) << recall       // 99.9%分位数
            << std::setw(12) << mean_ios << std::endl;

  LOG(INFO) << "search current time: " << current_time;
  disk_io = mean_ios;  // 返回磁盘I/O统计

  // 清理分配的内存
  delete[] query_result_dists;  // 释放搜索结果距离数组
  delete[] query_result_tags;   // 释放搜索结果标签数组
  delete[] stats;               // 释放查询统计数组
}

/**
 * 合并索引的核心函数
 * @tparam T 向量数据类型
 * @tparam TagT 标签数据类型
 * @param sync_index 动态SSD索引对象
 * 功能：对动态SSD索引执行最终合并操作，将内存中的数据合并到磁盘索引中
 */
template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  sync_index.final_merge(NUM_INSERT_THREADS);  // 执行最终合并，使用指定数量的插入线程
}

/**
 * 批量插入核心函数
 * @tparam T 向量数据类型
 * @tparam TagT 标签数据类型
 * @param data_load 要插入的向量数据数组
 * @param sync_index 动态SSD索引对象
 * @param insert_vec 要插入的标签向量
 * @param dim 向量的维度
 * 功能：并行执行批量向量插入操作，并统计插入性能指标
 */
template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                      size_t dim) {
  pipeann::Timer timer;  // 总插入计时器
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);  // 插入延迟统计数组

  LOG(INFO) << "Begin Insert";

  // 监控插入前的内存使用情况
  size_t rss_before_insert = get_current_rss();
  LOG(DEBUG) << "插入开始前内存: " << rss_before_insert << " KB";
  std::atomic_size_t success(0);  // 原子计数器，记录成功插入的数量

  // 使用OpenMP并行执行插入操作
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (int64_t i = 0; i < (int64_t) insert_vec.size(); i++) {
    pipeann::Timer insert_timer;  // 单个插入操作的计时器

    // 执行单个向量插入：将向量数据和对应标签插入索引
    sync_index.insert(data_load + dim * i, insert_vec[i]);

    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());  // 记录插入延迟（微秒）
  }

  // 计算总插入时间和内存使用变化
  float time_secs = timer.elapsed() / 1.0e6f;  // 转换为秒
  size_t rss_after_insert = get_current_rss();
  std::sort(insert_latencies.begin(), insert_latencies.end());  // 排序延迟数组用于计算百分位数

  // 输出插入性能统计
  LOG(INFO) << "Inserted " << insert_vec.size() << " points in " << time_secs << "s";
  LOG(DEBUG) << "插入完成后内存 - 增长前: " << rss_before_insert << " KB"
             << ", 增长后: " << rss_after_insert << " KB"
             << ", 实际增长: " << (rss_after_insert - rss_before_insert) << " KB";

  // 输出插入延迟分布统计（百分位数）
  LOG(INFO) << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us";
  LOG(INFO) << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us";
  LOG(INFO) << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us";
  LOG(INFO) << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us";
  LOG(INFO) << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us";
}

/**
 * 获取跟踪数据：准备删除和插入的标签，以及要插入的向量数据
 * @tparam T 向量数据类型
 * @tparam TagT 标签数据类型（默认为uint32_t）
 * @param data_bin 数据文件路径
 * @param l_start 要删除的起始标签
 * @param r_start 要插入的起始标签
 * @param n 要处理的向量数量
 * @param delete_tags 输出：要删除的标签向量
 * @param insert_tags 输出：要插入的标签向量
 * @param data_load 输出：要插入的向量数据
 * 功能：为增量更新准备数据，包括删除标签、插入标签和对应的向量数据
 */
template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
  LOG(INFO) << "l_start: " << l_start << " r_start: " << r_start << " n: " << n;

  // 生成要删除的标签序列 [l_start, l_start + n)
  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }

  // 生成要插入的标签序列 [r_start, r_start + n)
  for (uint64_t i = r_start; i < r_start + n; ++i) {
    insert_tags.push_back(i);
  }

  // 从数据文件中加载向量数据，从r_start位置开始加载n个向量
  int npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);
  reader.seekg(0, reader.beg);

  // 读取文件头：总向量数和维度
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));

  size_t data_dim = dim_i32;

  // 监控加载向量数据前的内存使用
  size_t rss_before_data = get_current_rss();
  data_load.resize(n * data_dim);

  // 定位到r_start位置并读取n个向量的数据
  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char *) data_load.data(), sizeof(T) * n * data_dim);

  size_t rss_after_data = get_current_rss();

  // 输出内存使用统计
  LOG(DEBUG) << "加载数据向量 - 数组大小: " << n << " x " << data_dim << ", 单元素大小: " << sizeof(T)
             << " bytes"
             << ", 理论大小: " << (n * data_dim * sizeof(T) / 1024.0) << " KB"
             << ", 增长前: " << rss_before_data << " KB"
             << ", 增长后: " << rss_after_data << " KB"
             << ", 实际增长: " << (rss_after_data - rss_before_data) << " KB";
}

/**
 * 主要的增量更新函数：执行插入-搜索-合并的完整流程
 * @tparam T 向量数据类型
 * @tparam TagT 标签数据类型
 * @param data_bin 训练数据文件路径
 * @param L_disk 磁盘索引的候选列表大小
 * @param vecs_per_step 每次更新的向量数量
 * @param num_steps 更新的总步数
 * @param index_prefix 索引文件前缀
 * @param query_file 查询文件路径
 * @param truthset_file 真值集文件路径前缀
 * @param truthset_l_offset 真值集偏移量
 * @param recall_at 召回率@K中的K值
 * @param Lsearch 搜索候选列表大小数组
 * @param beam_width beam search宽度
 * @param search_beam_width 搜索时的beam宽度
 * @param search_mem_L 内存搜索候选列表大小
 * @param dist_cmp 距离计算函数对象
 * 功能：模拟动态索引的增量更新场景，包括批量插入、实时搜索和最终合并
 */
template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
            const std::string &index_prefix, const std::string &query_file, const std::string &truthset_file,
            size_t truthset_l_offset, const int recall_at, const std::vector<uint64_t> &Lsearch,
            const unsigned beam_width, const uint32_t search_beam_width, const uint32_t search_mem_L,
            pipeann::Distance<T> *dist_cmp) {

  // 初始化索引参数
  pipeann::Parameters paras;
  paras.set(0, L_disk, 384, 1.2, NUM_SEARCH_THREADS + NUM_INSERT_THREADS, true, beam_width);

  std::vector<T> data_load;  // 存储要插入的向量数据
  size_t dim{};              // 向量维度

  pipeann::Timer timer;      // 总计时器

  // 加载查询向量
  LOG(INFO) << "Loading queries";
  T *query = NULL;
  size_t query_num, query_dim;
  size_t rss_before_query = get_current_rss();
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);
  size_t rss_after_query = get_current_rss();

  // 输出查询向量加载的内存统计
  LOG(DEBUG) << "加载查询向量 - 数组大小: " << query_num << " x " << query_dim << ", 单元素大小: " << sizeof(T)
             << " bytes"
             << ", 理论大小: " << (query_num * query_dim * sizeof(T) / 1024.0) << " KB"
             << ", 增长前: " << rss_before_query << " KB"
             << ", 增长后: " << rss_after_query << " KB"
             << ", 实际增长: " << (rss_after_query - rss_before_query) << " KB";

  // 设置维度和距离度量
  dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;  // 使用欧几里得距离

  // 创建动态SSD索引对象
  size_t rss_before_index = get_current_rss();
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, (search_mem_L > 0));
  size_t rss_after_index = get_current_rss();

  // 输出索引对象创建的内存统计
  LOG(DEBUG) << "创建 DynamicSSDIndex 对象 - 增长前: " << rss_before_index << " KB"
             << ", 增长后: " << rss_after_index << " KB"
             << ", 实际增长: " << (rss_after_index - rss_before_index) << " KB";

  // 在插入前执行基线搜索测试
  LOG(INFO) << "Searching before inserts: ";

  uint64_t res = 0;  // 已处理的向量数量

  // 获取初始真值文件名并开始计时
  std::string currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);  // 显示初始内存状态

  // 对每个搜索参数配置执行基线测试，记录磁盘I/O性能
  std::vector<double> ref_diskio;
  for (size_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width, sync_index,
                       currentFileName, false, true, diskio);
    ref_diskio.push_back(diskio);  // 记录基准磁盘I/O数据
  }

  // 初始化增量更新循环的变量
  int inMemorySize = 0;           // 内存中的向量数量
  std::future<void> merge_future; // 异步合并操作句柄
  uint64_t index_npts = sync_index._disk_index->num_points;  // 当前索引中的向量数量

  // 主要的增量更新循环
  for (int i = 0; i < num_steps; i++) {
    LOG(INFO) << "Batch: " << i << " Total Batch : " << num_steps;

    std::vector<unsigned> insert_vec;  // 本次插入的标签向量
    std::vector<unsigned> delete_vec;  // 本次删除的标签向量（本测试中未实际使用）

    /**准备更新数据：确定要插入的向量范围**/
    uint64_t st = vecs_per_step * i;        // 起始位置
    uint64_t ed = st + index_npts;          // 结束位置（相对于索引大小）
    LOG(INFO) << "st: " << st << " ed: " << ed;

    // 获取要插入的向量数据和标签
    get_trace<T, TagT>(data_bin, st, ed, vecs_per_step, delete_vec, insert_vec, data_load);

    // 启动异步插入操作
    std::future<void> insert_future = std::async(std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(insert_vec), dim);

    // 在插入进行期间，执行搜索查询来测试并发性能
    int total_queries = 0;
    std::future_status insert_status;

    do {
      // 等待5秒检查插入操作状态
      insert_status = insert_future.wait_for(std::chrono::seconds(5));

      if (insert_status == std::future_status::deferred) {
        LOG(INFO) << "deferred\n";  // 插入操作被延迟
      } else if (insert_status == std::future_status::timeout) {
        // 插入仍在进行中，此时执行搜索查询测试并发性能
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        LOG(INFO) << "Number of vectors: " << sync_index._disk_index->cur_id;

        double dummy;
        // 在插入过程中执行搜索测试（只使用第一个搜索配置）
        sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[0], search_beam_width,
                           sync_index, currentFileName, false, false, dummy);
        sleep(1);  // 短暂休眠避免过度占用CPU

        total_queries += query_num;
        LOG(INFO) << "Queries processed: " << total_queries;
      }

      if (insert_status == std::future_status::ready) {
        LOG(INFO) << "Insertions complete!\n";  // 插入操作完成
      }
    } while (insert_status != std::future_status::ready);  // 直到插入操作完成

    // 更新内存中的向量数量统计
    inMemorySize += insert_vec.size();

    // 在更新后执行搜索测试
    LOG(INFO) << "Search after update, current vector number: " << res;

    // 更新已处理向量总数和对应的真值文件
    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);

    // 对所有搜索配置执行更新后的性能测试
    std::vector<double> disk_ios;
    for (size_t j = 0; j < Lsearch.size(); ++j) {
      double diskio = 0;
      sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width,
                         sync_index, currentFileName, false, true, diskio);
      disk_ios.push_back(diskio);
    }

    // 如果是最后一步且步数足够多，则执行最终合并并存储索引
    if (i == num_steps - 1 && num_steps >= 10) {  // 存储最终索引，图9和图11使用num_steps < 10的情况
      LOG(INFO) << "Store the last index to disk.";

      // 启动异步合并操作
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index));

      // 等待合并操作完成
      std::future_status merge_status;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(10));
      } while (merge_status != std::future_status::ready);

      LOG(INFO) << "Store finished.";
      exit(0);  // 合并完成后退出程序
    }
  }
}

/**
 * 主函数：程序入口点
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * 功能：解析命令行参数，根据数据类型启动相应的增量更新测试
 */
int main(int argc, char **argv) {
  // 输出程序启动时的内存使用情况
  LOG(DEBUG) << "程序启动 - 当前内存使用: " << get_current_rss() << " KB";

  // 检查命令行参数数量
  if (argc < 18) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk>"
              << " <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode>"
              << " <index_prefix> <query_file> <truthset_prefix> <truthset_l_offset> <recall@>"
              << " <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  // 解析命令行参数（从第3个参数开始，第2个是数据类型）
  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);        // 数据文件路径
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);         // 磁盘索引候选列表大小

  // 每步更新的向量数量（例如：1M个向量）
  int vecs_per_step = (int) std::atoi(argv[arg_no++]);

  // 更新步数（例如：100步用于100M+100M测试，200步用于800M+200M测试）
  int num_steps = (int) std::atoi(argv[arg_no++]);

  // 线程配置
  NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);       // 插入线程数
  NUM_SEARCH_THREADS = (int) std::atoi(argv[arg_no++]);       // 搜索线程数
  LOG(INFO) << "num insert threads: " << NUM_INSERT_THREADS;
  LOG(INFO) << "num search threads: " << NUM_SEARCH_THREADS;

  // 搜索模式配置
  search_mode = std::atoi(argv[arg_no++]);                    // 搜索模式
  LOG(INFO) << "search mode: " << search_mode;

  std::string index_prefix(argv[arg_no++]);                   // 索引文件前缀
  std::string query_file(argv[arg_no++]);                     // 查询文件路径

  // 真值集配置：gt_X.bin表示[0, x + 100M)向量的真值集
  std::string truthset(argv[arg_no++]);

  // 真值集偏移量：100M+100M测试应为0，800M+200M测试应为700M
  size_t truthset_l_offset = (size_t) std::atoll(argv[arg_no++]);

  // 搜索性能参数
  int recall_at = (int) std::atoi(argv[arg_no++]);            // 召回率@K中的K值
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);           // beam search宽度
  unsigned search_beam_width = (unsigned) std::atoi(argv[arg_no++]);     // 搜索时的beam宽度
  unsigned search_mem_L = (unsigned) std::atoi(argv[arg_no++]);          // 内存搜索候选列表大小（0表示不使用内存索引）

  // 搜索参数列表（可以有多个L值用于测试）
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;  // 缓存节点数（未使用）

  // 根据数据类型启动相应的测试
  if (std::string(argv[1]) == std::string("int8")) {
    // 8位有符号整数向量测试
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                             truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                             &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    // 8位无符号整数向量测试
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                              truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                              &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    // 单精度浮点数向量测试
    pipeann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                            truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                            &dist_cmp);
  } else {
    // 不支持的数据类型
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
  }
}
