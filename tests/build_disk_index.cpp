#include <cstdint>
#include "nbr/abstract_nbr.h"
#include "nbr/pq_nbr.h"
#include "nbr/rabitq_nbr.h"
#include "omp.h"

#include "aux_utils.h"
#include "utils.h"

// Usage:
// build/tests/build_disk_index <data_type (float/int8/uint8)> <data_file.bin> <index_prefix_path> <R>  <L>  <bytes_per_nbr>  <M>  <T> <similarity metric (cosine/l2) case sensitive>. <nbr_type (pq/rabitq)>
// build/tests/build_disk_index uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme2/indices/bigann/100m 96 128 32 256 112 l2 pq

int main(int argc, char **argv) {
  // 检查命令行参数数量
  if (argc < 11) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2) case sensitive> <nbr_type (pq/rabitq)>."
                 " See README for more information on parameters."
              << std::endl;
  } else {
    // 解析相似度度量方法参数
    std::string dist_metric(argv[9]);

    // 设置相似度度量方法：余弦相似度或L2距离
    pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
    if (dist_metric != "l2" && m == pipeann::Metric::L2) {
      std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
    }

    // 解析邻居搜索类型参数
    std::string nbr_type = argv[10];

    // 处理float数据类型
    if (std::string(argv[1]) == std::string("float")) {
      pipeann::AbstractNeighbor<float> *nbr_handler = nullptr;
      // 根据邻居搜索类型创建相应的处理器
      if (nbr_type == "rabitq") {
        nbr_handler = new pipeann::RaBitQNeighbor<float>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<float>();
      }
      // 构建磁盘索引
      pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                       std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    // 处理int8数据类型
    } else if (std::string(argv[1]) == std::string("int8")) {
      pipeann::AbstractNeighbor<int8_t> *nbr_handler = nullptr;
      if (nbr_type == "rabitq") {
        nbr_handler = new pipeann::RaBitQNeighbor<int8_t>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<int8_t>();
      }
      // 构建磁盘索引
      pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                        std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    // 处理uint8数据类型
    } else if (std::string(argv[1]) == std::string("uint8")) {
      pipeann::AbstractNeighbor<uint8_t> *nbr_handler = nullptr;
      if (nbr_type == "rabitq") {
        nbr_handler = new pipeann::RaBitQNeighbor<uint8_t>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<uint8_t>();
      }
      // 构建磁盘索引
      pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                         std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    // 处理不支持的数据类型
    } else {
      std::cout << "Error. wrong file type" << std::endl;
    }
  }
}
