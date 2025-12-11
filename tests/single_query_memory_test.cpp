#include <cstring>
#include <iostream>
#include <ssd_index.h>
#include <string.h>
#include <time.h>

#include "utils/log.h"
#include "nbr/nbr.h"
#include "linux_aligned_file_reader.h"
#include "utils.h"

/**
 * 单查询内存测试程序
 * 用于配合 valgrind 监控 PipeANN 在单次查询时的内存占用
 *
 * 用法:
 * ./single_query_memory_test <index_type> <index_prefix> <num_threads> <query_file> <K> <L> <metric> <nbr_type>
 *
 * 示例:
 * valgrind --tool=massif --massif-out-file=massif.out ./single_query_memory_test float ./index 1 query.bin 10 100 l2 pq
 */

template<typename T>
int single_query_test(int argc, char **argv) {
  // 参数解析
  if (argc < 9) {
    std::cerr << "Usage: " << argv[0]
              << " <index_type> <index_prefix> <num_threads> <query_file> "
              << "<K> <L> <metric> <nbr_type>" << std::endl;
    std::cerr << "\nExample: " << argv[0]
              << " float ./index 1 query.bin 10 100 l2 pq" << std::endl;
    return -1;
  }

  int idx = 2;
  std::string index_prefix(argv[idx++]);
  uint32_t num_threads = std::atoi(argv[idx++]);
  std::string query_file(argv[idx++]);
  uint64_t K = std::atoi(argv[idx++]);         // 返回结果数量
  uint64_t L = std::atoi(argv[idx++]);         // 搜索参数 L
  std::string metric_str(argv[idx++]);
  std::string nbr_type(argv[idx++]);

  // 设置参数
  uint64_t beamwidth = 4;  // 默认 beamwidth
  uint32_t mem_L = 0;      // 不使用内存索引
  bool tags_flag = true;

  std::cout << "============================================" << std::endl;
  std::cout << "PipeANN 单查询内存测试" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "索引路径: " << index_prefix << std::endl;
  std::cout << "线程数: " << num_threads << std::endl;
  std::cout << "查询文件: " << query_file << std::endl;
  std::cout << "K (返回结果数): " << K << std::endl;
  std::cout << "L (搜索参数): " << L << std::endl;
  std::cout << "距离度量: " << metric_str << std::endl;
  std::cout << "邻居类型: " << nbr_type << std::endl;
  std::cout << "============================================" << std::endl;

  // 获取度量类型
  pipeann::Metric metric = pipeann::get_metric(metric_str);

  // 加载查询数据
  T *query_data = nullptr;
  size_t query_num, query_dim;

  std::cout << "\n[1/4] 加载查询数据..." << std::endl;
  pipeann::load_bin<T>(query_file, query_data, query_num, query_dim);
  std::cout << "查询数量: " << query_num << ", 维度: " << query_dim << std::endl;

  if (query_num == 0) {
    std::cerr << "错误: 查询文件为空!" << std::endl;
    return -1;
  }

  // 创建文件读取器和邻居处理器
  std::cout << "\n[2/4] 初始化索引..." << std::endl;
  std::shared_ptr<AlignedFileReader> reader;
  reader.reset(new LinuxAlignedFileReader());

  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  std::unique_ptr<pipeann::SSDIndex<T>> index(
      new pipeann::SSDIndex<T>(metric, reader, nbr_handler, tags_flag));

  // 加载索引
  std::cout << "加载磁盘索引..." << std::endl;
  int res = index->load(index_prefix.c_str(), num_threads, false);
  if (res != 0) {
    std::cerr << "错误: 索引加载失败!" << std::endl;
    delete[] query_data;
    return res;
  }
  std::cout << "索引加载成功!" << std::endl;

  // 准备结果缓冲区
  std::vector<uint32_t> result_ids(K);
  std::vector<float> result_dists(K);
  pipeann::QueryStats stats;

  // 执行单次查询
  std::cout << "\n[3/4] 执行单次查询..." << std::endl;
  std::cout << "使用第一个查询向量进行搜索..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // 使用 beam_search 执行查询
  index->beam_search(
      query_data,           // 查询向量 (第一个)
      K,                    // 返回结果数量
      mem_L,                // 内存索引参数 (0 表示不使用)
      L,                    // 搜索参数
      result_ids.data(),    // 结果 ID
      result_dists.data(),  // 结果距离
      beamwidth,            // beamwidth
      &stats,               // 统计信息
      nullptr,              // 删除节点集合
      true                  // 动态搜索 L
  );

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // 输出结果
  std::cout << "\n[4/4] 查询完成!" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "查询耗时: " << elapsed.count() * 1000.0 << " ms" << std::endl;
  std::cout << "延迟: " << stats.total_us << " us" << std::endl;
  std::cout << "跳数: " << stats.n_hops << std::endl;
  std::cout << "I/O 次数: " << stats.n_ios << std::endl;

  std::cout << "\n前 " << std::min((uint64_t)10, K) << " 个最近邻结果:" << std::endl;
  for (uint64_t i = 0; i < std::min((uint64_t)10, K); i++) {
    std::cout << "  [" << i << "] ID: " << result_ids[i]
              << ", Distance: " << result_dists[i] << std::endl;
  }
  std::cout << "============================================" << std::endl;

  // 清理
  delete[] query_data;

  std::cout << "\n测试完成!" << std::endl;
  std::cout << "提示: 使用 ms_print massif.out 查看内存使用情况" << std::endl;

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <index_type> ..." << std::endl;
    std::cerr << "index_type: float, int8, uint8" << std::endl;
    return -1;
  }

  std::string index_type(argv[1]);

  if (index_type == "float") {
    return single_query_test<float>(argc, argv);
  } else if (index_type == "int8") {
    return single_query_test<int8_t>(argc, argv);
  } else if (index_type == "uint8") {
    return single_query_test<uint8_t>(argc, argv);
  } else {
    std::cerr << "不支持的索引类型: " << index_type << std::endl;
    std::cerr << "支持的类型: float, int8, uint8" << std::endl;
    return -1;
  }
}
