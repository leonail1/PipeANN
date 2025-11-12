/**
 * @file search_disk_index.cpp
 * @brief 磁盘索引搜索测试程序，评估不同搜索模式与 L 的性能与召回。
 *
 * 该程序加载 PipeANN 的磁盘索引，并对给定查询在多线程下执行搜索，
 * 输出 QPS、平均/分位时延、平均 hops/IOs，以及可选的 Recall@K。
 */
#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <string.h>
#include <time.h>
#include <iomanip>

#include "utils/log.h"
#include "nbr/abstract_nbr.h"
#include "nbr/pq_nbr.h"
#include "nbr/rabitq_nbr.h"
#include "utils/timer.h"
#include "utils.h"
#include "aux_utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

#define WARMUP false

// 使用 utils.h 中的 get_current_rss()

/**
 * @brief 打印统计信息行，包括百分位和对应数值。
 * @param category 统计类别名称，用于前缀显示。
 * @param percentiles 需要展示的百分位列表（例如 50, 95, 99）。
 * @param results 与百分位一一对应的结果数值。
 */
void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results) {
  std::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(8) << percentiles[s] << "%";
  }
  std::cout << std::endl;
  std::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(9) << results[s];
  }
  std::cout << std::endl;
}

/**
 * @brief 在磁盘索引上执行多线程 ANN 搜索基准。
 * @tparam T 向量数据类型（float/int8/uint8）。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 0 表示正常结束，非 0 表示加载或搜索失败。
 *
 * 详细说明：
 * - 从命令行解析索引路径、线程数、I/O 束宽（beamwidth）、查询文件、真值集、K（recall_at）、
 *   距离度量、邻居编码类型（PQ/RaBitQ）、搜索模式（BEAM/PAGE/PIPE/CORO）、内存索引 L（mem_L）以及若干 L 值。
 * - 加载磁盘索引与（可选的）内存索引，按指定搜索模式对所有查询执行检索并统计性能与召回。
 */
template<typename T>
int search_disk_index(int argc, char **argv) {
  LOG(DEBUG) << "程序启动 - 当前内存使用: " << get_current_rss() << " KB";

  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  bool tags_flag = true;

  int index = 2;                                    // 从 argv[2] 开始解析参数（argv[1] 是数据类型）
  std::string index_prefix_path(argv[index++]);     // 索引前缀路径，用于定位磁盘/内存索引文件
  uint32_t num_threads = std::atoi(argv[index++]);  // OpenMP 线程数
  uint32_t beamwidth = std::atoi(argv[index++]);    // I/O 束宽（每次并行读取的扇区数），<=0 表示自动优化
  std::string query_bin(argv[index++]);             // 查询向量的二进制文件路径
  std::string truthset_bin(argv[index++]);          // 召回评估的真值集（可为 "null"）
  uint64_t recall_at = std::atoi(argv[index++]);    // top-K（用于评估与输出）
  std::string dist_metric(argv[index++]);           // 距离度量类型：l2 或 cosine
  std::string nbr_type = argv[index++];             // 邻居编码类型：pq 或 rabitq
  int search_mode = std::atoi(argv[index++]);       // 搜索模式：0=BEAM,1=PAGE,2=PIPE,3=CORO
  bool use_page_search = search_mode != 0;          // 是否加载页布局以支持 page/pipe 模式
  uint32_t mem_L = std::atoi(argv[index++]);        // pipe/page/beam 搜索时可选的内存索引 L

  pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE
                                              : pipeann::Metric::L2;  // 如果指定为 cosine 则使用余弦，否则默认 L2 距离
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {              // 未知度量时退回 L2 并提示
    std::cout << "Unknown distance metric: " << dist_metric << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";  // 兼容旧版标签文件名（当前未直接使用）

  bool calc_recall_flag = false;  // 是否在输出中计算 Recall@K（取决于是否提供真值集）

  for (int ctr = index; ctr < argc; ctr++) {  // 解析命令行尾部的多个 L 值，过滤出 >= recall_at 的有效值
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at" << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  size_t rss_before_query = get_current_rss();
  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);  // 加载查询向量数据（BIN 格式）
  size_t rss_after_query = get_current_rss();
  LOG(DEBUG) << "加载查询向量 - 数组大小: " << query_num << " x " << query_dim << ", 单元素大小: " << sizeof(T)
             << " bytes"
             << ", 理论大小: " << (query_num * query_dim * sizeof(T) / 1024.0) << " KB"
             << ", 增长前: " << rss_before_query << " KB"
             << ", 增长后: " << rss_after_query << " KB"
             << ", 实际增长: " << (rss_after_query - rss_before_query) << " KB";

  if (file_exists(truthset_bin)) {  // 若提供真值集，则加载以用于召回评估
    size_t rss_before_gt = get_current_rss();
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    size_t rss_after_gt = get_current_rss();
    size_t gt_theoretical = (gt_num * gt_dim * (sizeof(unsigned) + sizeof(float))) / 1024.0;
    if (tags != nullptr) {
      gt_theoretical += (gt_num * gt_dim * sizeof(uint32_t)) / 1024.0;
    }
    LOG(DEBUG) << "加载真值集 - gt_ids 大小: " << gt_num << " x " << gt_dim << ", 单元素大小: " << sizeof(unsigned)
               << " bytes"
               << ", 理论大小: " << gt_theoretical << " KB"
               << ", 增长前: " << rss_before_gt << " KB"
               << ", 增长后: " << rss_after_gt << " KB"
               << ", 实际增长: " << (rss_after_gt - rss_before_gt) << " KB";
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;  // 启用召回计算
  }

  // 构建对齐文件读取器（Linux 平台下基于 io_uring 的对齐 I/O）
  size_t rss_before_reader = get_current_rss();
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  size_t rss_after_reader = get_current_rss();
  LOG(DEBUG) << "创建 AlignedFileReader - 增长前: " << rss_before_reader << " KB"
             << ", 增长后: " << rss_after_reader << " KB"
             << ", 实际增长: " << (rss_after_reader - rss_before_reader) << " KB";

  // 选择邻居编码处理器（PQ 或 RaBitQ），用于压缩邻居向量与检索
  pipeann::AbstractNeighbor<T> *nbr_handler = nullptr;
  size_t rss_before_nbr = get_current_rss();
  if (nbr_type == "pq") {
    nbr_handler = new pipeann::PQNeighbor<T>();  // 乘积量化（Product Quantization）
    size_t rss_after_nbr = get_current_rss();
    LOG(DEBUG) << "创建 PQNeighbor - 增长前: " << rss_before_nbr << " KB"
               << ", 增长后: " << rss_after_nbr << " KB"
               << ", 实际增长: " << (rss_after_nbr - rss_before_nbr) << " KB";
  } else if (nbr_type == "rabitq") {
    nbr_handler = new pipeann::RaBitQNeighbor<T>();  // RaBitQ 方案
    size_t rss_after_nbr = get_current_rss();
    LOG(DEBUG) << "创建 RaBitQNeighbor - 增长前: " << rss_before_nbr << " KB"
               << ", 增长后: " << rss_after_nbr << " KB"
               << ", 实际增长: " << (rss_after_nbr - rss_before_nbr) << " KB";
  } else {
    std::cout << "Unknown nbr type: " << nbr_type << std::endl;
    return -1;
  }
  // 创建磁盘索引实例（可选支持标签），并绑定距离度量与文件读取器
  size_t rss_before_ssd = get_current_rss();
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(new pipeann::SSDIndex<T>(m, reader, nbr_handler, tags_flag));
  size_t rss_after_ssd = get_current_rss();
  LOG(DEBUG) << "创建 SSDIndex 对象 - 增长前: " << rss_before_ssd << " KB"
             << ", 增长后: " << rss_after_ssd << " KB"
             << ", 实际增长: " << (rss_after_ssd - rss_before_ssd) << " KB";

  size_t rss_before_load = get_current_rss();
  int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads, true, use_page_search);
  size_t rss_after_load = get_current_rss();
  LOG(DEBUG) << "加载磁盘索引完成 - 增长前: " << rss_before_load << " KB"
             << ", 增长后: " << rss_after_load << " KB"
             << ", 实际增长: " << (rss_after_load - rss_before_load) << " KB";
  if (res != 0) {
    return res;
  }

  if (mem_L != 0) {  // 可选：加载内存索引以辅助磁盘搜索（mem_L 控制内存候选数）
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index " << mem_index_path << " " << query_dim;
    size_t rss_before_memidx = get_current_rss();
    _pFlashIndex->load_mem_index(m, query_dim, mem_index_path);
    size_t rss_after_memidx = get_current_rss();
    LOG(DEBUG) << "加载内存索引完成 - 增长前: " << rss_before_memidx << " KB"
               << ", 增长后: " << rss_after_memidx << " KB"
               << ", 实际增长: " << (rss_after_memidx - rss_before_memidx) << " KB";
  }

  omp_set_num_threads(num_threads);  // 设置 OpenMP 线程并行度

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());   // 结果 ID（保留以对齐结构）
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());  // 每个 L 测试的标签结果
  std::vector<std::vector<float>> query_result_dists(Lvec.size());    // 每个 L 测试的距离结果
  LOG(DEBUG) << "创建结果向量 - 每个向量容量: " << Lvec.size() << ", 当前内存使用: " << get_current_rss() << " KB";

  // 记录每个 L 对应的内存分配信息
  struct MemAllocInfo {
    size_t expected_kb;  // 预计分配的内存大小(KB)
    size_t before_kb;    // 分配前内存(KB)
    size_t after_kb;     // 分配后内存(KB)
  };

  struct LMemoryInfo {
    uint64_t L;                        // L 值
    MemAllocInfo querystats_alloc;     // QueryStats 数组分配
    MemAllocInfo result_arrays_alloc;  // 结果数组分配
    MemAllocInfo temp_tags_alloc;      // 临时标签数组分配
  };

  std::vector<LMemoryInfo> memory_records(Lvec.size());  // 记录所有 L 的内存分配信息

  /* 运行一次 L 设置的测试：执行搜索并统计性能 */
  auto run_tests = [&](uint32_t test_id, bool output) {
    LMemoryInfo &mem_info = memory_records[test_id];
    mem_info.L = Lvec[test_id];

    // 记录 QueryStats 数组分配
    mem_info.querystats_alloc.before_kb = get_current_rss();
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];  // 保存每个查询的统计信息
    mem_info.querystats_alloc.after_kb = get_current_rss();
    mem_info.querystats_alloc.expected_kb = (query_num * sizeof(pipeann::QueryStats)) / 1024.0;

    uint64_t L = Lvec[test_id];  // 当前测试使用的 L 值（磁盘候选数）

    // 记录结果数组分配
    mem_info.result_arrays_alloc.before_kb = get_current_rss();
    query_result_ids[test_id].resize(recall_at * query_num);    // 每个查询保留 recall_at 个结果
    query_result_dists[test_id].resize(recall_at * query_num);  // 距离结果
    query_result_tags[test_id].resize(recall_at * query_num);   // 标签结果（32位）
    mem_info.result_arrays_alloc.after_kb = get_current_rss();
    mem_info.result_arrays_alloc.expected_kb = (recall_at * query_num * (sizeof(uint32_t) * 2 + sizeof(float))) / 1024.0;

    // 记录临时标签数组分配
    mem_info.temp_tags_alloc.before_kb = get_current_rss();
    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);  // 备用 64 位标签缓存
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    mem_info.temp_tags_alloc.after_kb = get_current_rss();
    mem_info.temp_tags_alloc.expected_kb = (recall_at * query_num * (sizeof(uint64_t) + sizeof(uint32_t))) / 1024.0;
    auto s = std::chrono::high_resolution_clock::now();  // 开始计时

    if (search_mode == SearchMode::PIPE_SEARCH) {  // 管线化搜索（使用 SQPOLL 加速 I/O）
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {  // 并行执行 pipe_search
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else if (search_mode == SearchMode::PAGE_SEARCH) {  // 页级搜索
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {  // 并行执行 page_search
        _pFlashIndex->page_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else if (search_mode == SearchMode::CORO_SEARCH) {  // 协程批处理搜索
      constexpr uint64_t kBatchSize = 8;                  // 每批处理的查询数
      T *q[kBatchSize];
      uint32_t *res_tags[kBatchSize];
      float *res_dists[kBatchSize];
      int N;
#pragma omp parallel for schedule(dynamic, 1) private(q, res_tags, res_dists, N)
      for (int64_t i = 0; i < (int64_t) query_num; i += kBatchSize) {  // 分批执行 coro_search
        N = std::min(kBatchSize, query_num - i);
        for (int v = 0; v < N; ++v) {
          q[v] = query + ((i + v) * query_dim);
          res_tags[v] = query_result_tags_32.data() + ((i + v) * recall_at);
          res_dists[v] = query_result_dists[test_id].data() + ((i + v) * recall_at);
        }

        _pFlashIndex->coro_search(q, (uint64_t) recall_at, mem_L, (uint64_t) L, res_tags, res_dists,
                                  (uint64_t) beamwidth, N);
      }
    } else if (search_mode == SearchMode::BEAM_SEARCH) {  // 束搜索（经典方法）
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {  // 并行执行 beam_search
        _pFlashIndex->beam_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth, stats + i,
                                  nullptr, false);
      }
    } else {
      std::cout << "Unknown search mode: " << search_mode << std::endl;
      exit(-1);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));  // 查询每秒（QPS）

    pipeann::convert_types<uint32_t, uint32_t>(query_result_tags_32.data(), query_result_tags[test_id].data(),
                                               (size_t) query_num, (size_t) recall_at);  // 标签类型转换/复制

    float mean_latency = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });  // 平均总时延

    float latency_999 = (float) pipeann::get_percentile_stats(
        stats, query_num, 0.999f, [](const pipeann::QueryStats &stats) { return stats.total_us; });  // P99.9 时延

    float mean_hops = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_hops; });  // 平均跳数

    float mean_ios = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });  // 平均 I/O 次数

    delete[] stats;

    if (output) {  // 打印当前 L 的统计与召回
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same distance,
          which may cause lower than expected recall@1 (?) */
        recall = (float) pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                   query_result_tags[test_id].data(), (uint32_t) recall_at,
                                                   (uint32_t) recall_at);  // 计算 Recall@K
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios;  // 输出指标列
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;  // 追加 Recall 列
      }
    }
  };

  // LOG(INFO) << "Use two ANNS for warming up...";
  // uint32_t prev_L = Lvec[0];
  // Lvec[0] = 200;
  // run_tests(0, false);
  // run_tests(0, false);
  // Lvec[0] = prev_L;
  // LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
            << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "==========================================="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }

//   // 输出所有 L 的内存分配详情
//   std::cout << "\n=============================================="
//                "==========================================="
//             << std::endl;
//   std::cout << "Memory Allocation Details for Each L:" << std::endl;
//   std::cout << "=============================================="
//                "==========================================="
//             << std::endl;

//   for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
//     const LMemoryInfo &mem_info = memory_records[test_id];

//     std::cout << "\nL = " << mem_info.L << ":" << std::endl;

//     // QueryStats 数组分配
//     std::cout << "  [1] QueryStats Array:" << std::endl;
//     std::cout << "      预计分配: " << std::setw(10) << mem_info.querystats_alloc.expected_kb << " KB" << std::endl;
//     std::cout << "      分配前:   " << std::setw(10) << mem_info.querystats_alloc.before_kb << " KB" << std::endl;
//     std::cout << "      分配后:   " << std::setw(10) << mem_info.querystats_alloc.after_kb << " KB" << std::endl;
//     std::cout << "      实际增长: " << std::setw(10)
//               << (mem_info.querystats_alloc.after_kb - mem_info.querystats_alloc.before_kb) << " KB" << std::endl;

//     // 结果数组分配
//     std::cout << "  [2] Result Arrays (ids/dists/tags):" << std::endl;
//     std::cout << "      预计分配: " << std::setw(10) << mem_info.result_arrays_alloc.expected_kb << " KB" << std::endl;
//     std::cout << "      分配前:   " << std::setw(10) << mem_info.result_arrays_alloc.before_kb << " KB" << std::endl;
//     std::cout << "      分配后:   " << std::setw(10) << mem_info.result_arrays_alloc.after_kb << " KB" << std::endl;
//     std::cout << "      实际增长: " << std::setw(10)
//               << (mem_info.result_arrays_alloc.after_kb - mem_info.result_arrays_alloc.before_kb) << " KB" << std::endl;

//     // 临时标签数组分配
//     std::cout << "  [3] Temporary Tag Arrays:" << std::endl;
//     std::cout << "      预计分配: " << std::setw(10) << mem_info.temp_tags_alloc.expected_kb << " KB" << std::endl;
//     std::cout << "      分配前:   " << std::setw(10) << mem_info.temp_tags_alloc.before_kb << " KB" << std::endl;
//     std::cout << "      分配后:   " << std::setw(10) << mem_info.temp_tags_alloc.after_kb << " KB" << std::endl;
//     std::cout << "      实际增长: " << std::setw(10)
//               << (mem_info.temp_tags_alloc.after_kb - mem_info.temp_tags_alloc.before_kb) << " KB" << std::endl;
//   }

//   std::cout << "\n=============================================="
//                "==========================================="
//             << std::endl;

  return 0;
}

/**
 * @brief 程序入口：解析参数并调用对应数据类型的搜索基准。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 *
 * 使用说明（与程序输出的 Usage 对应）：
 *  index_type(float/int8/uint8) index_prefix_path num_threads beamwidth query_file.bin truthset.bin K
 * similarity(cosine/l2) nbr_type(pq/rabitq) search_mode(0/1/2/3) mem_L L1 [L2] ...
 */
int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2)> <nbr_type (pq/rabitq)>"
                 " <search_mode(0 for beam search / 1 for page search / 2 for pipe search)> <mem_L (0 means not "
                 "using mem index)> <L1> [L2] etc."
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8" << std::endl;
}
