/**
 * @file utils.cpp
 * @brief PipeANN 工具函数实现文件
 * 
 * 本文件包含了 PipeANN 库中的核心工具函数实现，主要包括：
 * - 距离函数工厂方法的模板特化实现
 * - 数据文件归一化处理功能
 * 
 * @author PipeANN Team
 * @version 1.0
 * @date 2024
 * @copyright Copyright (c) 2024 PipeANN Team
 */

#include "utils.h"

#include <stdio.h>

namespace pipeann {
  
  /**
   * @brief 获取 float 类型数据的距离计算函数（模板特化）
   * 
   * 根据指定的距离度量类型，返回相应的 float 类型距离计算函数实例。
   * 支持 L2 欧几里得距离和余弦距离两种度量方式。
   * 
   * @param m 距离度量类型，支持 L2 和 COSINE
   * @return pipeann::Distance<float>* 返回对应的距离计算函数指针
   * @throws 如果传入不支持的度量类型，程序将崩溃退出
   * 
   * @note 返回的指针需要调用者负责释放内存
   * @see pipeann::Metric
   * @see pipeann::Distance
   */
  template<>
  pipeann::Distance<float> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2();  // 编译时分发，创建 L2 距离计算器
    } else if (m == pipeann::Metric::COSINE) {
      return new pipeann::DistanceCosineFloat();  // 创建余弦距离计算器
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now.";
      crash();  // 不支持的度量类型，程序崩溃
      return nullptr;
    }
  }

  /**
   * @brief 获取 int8_t 类型数据的距离计算函数（模板特化）
   * 
   * 根据指定的距离度量类型，返回相应的 int8_t 类型距离计算函数实例。
   * 针对 8 位有符号整数数据进行了优化。
   * 
   * @param m 距离度量类型，支持 L2 和 COSINE
   * @return pipeann::Distance<int8_t>* 返回对应的距离计算函数指针
   * @throws 如果传入不支持的度量类型，程序将崩溃退出
   * 
   * @note 返回的指针需要调用者负责释放内存
   * @see pipeann::Metric
   * @see pipeann::Distance
   */
  template<>
  pipeann::Distance<int8_t> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2Int8();  // 创建 int8 类型的 L2 距离计算器
    } else if (m == pipeann::Metric::COSINE) {
      return new pipeann::DistanceCosineInt8();  // 创建 int8 类型的余弦距离计算器
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now";
      crash();  // 不支持的度量类型，程序崩溃
      return nullptr;
    }
  }

  /**
   * @brief 获取 uint8_t 类型数据的距离计算函数（模板特化）
   * 
   * 根据指定的距离度量类型，返回相应的 uint8_t 类型距离计算函数实例。
   * 针对 8 位无符号整数数据进行了优化，但余弦距离使用较慢的实现版本。
   * 
   * @param m 距离度量类型，支持 L2 和 COSINE
   * @return pipeann::Distance<uint8_t>* 返回对应的距离计算函数指针
   * @throws 如果传入不支持的度量类型，程序将崩溃退出
   * 
   * @warning 对于 uint8_t 类型的余弦距离，由于没有 AVX/AVX2 优化版本，使用较慢的实现
   * @note 返回的指针需要调用者负责释放内存
   * @see pipeann::Metric
   * @see pipeann::Distance
   */
  template<>
  pipeann::Distance<uint8_t> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2UInt8();  // 创建 uint8 类型的 L2 距离计算器
    } else if (m == pipeann::Metric::COSINE) {
      LOG(INFO) << "AVX/AVX2 distance function not defined for Uint8. Using slow version.";
      return new pipeann::SlowDistanceCosineUInt8();  // 使用慢速版本的余弦距离计算器
    } else {
      LOG(ERROR) << "Only L2 and Cosine metric supported as of now.";
      crash();  // 不支持的度量类型，程序崩溃
      return nullptr;
    }
  }

  /**
   * @brief 对数据文件进行归一化处理
   * 
   * 读取输入的二进制向量文件，对每个向量进行 L2 归一化处理（单位化），
   * 然后将归一化后的向量写入输出文件。归一化过程将每个向量的模长缩放为 1。
   * 
   * 文件格式要求：
   * - 前 4 字节：数据点数量（int32_t）
   * - 接下来 4 字节：向量维度（int32_t）
   * - 后续数据：float 类型的向量数据
   * 
   * @param inFileName 输入文件名，包含待归一化的 float 向量数据
   * @param outFileName 输出文件名，存储归一化后的向量数据
   * 
   * @note 
   * - 使用分块处理方式，每块处理 131072 个向量，减少内存占用
   * - 使用 OpenMP 并行化处理，提高归一化速度
   * - 归一化公式：v_normalized = v / ||v||_2
   * - 为避免除零错误，在计算模长时加入了 epsilon 值
   * 
   * @warning 
   * - 输入文件必须是有效的二进制向量文件格式
   * - 确保有足够的磁盘空间存储输出文件
   * - 处理大文件时可能需要较长时间
   * 
   * @see ROUND_UP
   */
  void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
    // 打开输入和输出文件流，使用二进制模式
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    // 读取文件头信息：数据点数量和向量维度
    int npts_s32, ndims_s32;
    readr.read((char *) &npts_s32, sizeof(int32_t));  // 读取数据点数量（32位整数）
    readr.read((char *) &ndims_s32, sizeof(int32_t)); // 读取向量维度（32位整数）

    // 将文件头信息写入输出文件
    writr.write((char *) &npts_s32, sizeof(int32_t));
    writr.write((char *) &ndims_s32, sizeof(int32_t));

    // 转换为64位无符号整数，避免溢出问题
    uint64_t npts = (uint64_t) npts_s32, ndims = (uint64_t) ndims_s32;
    LOG(INFO) << "Normalizing FLOAT vectors in file: " << inFileName;
    LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    // 设置分块处理参数，每块处理131072个向量
    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;  // 计算总块数，向上取整
    LOG(INFO) << "# blks: " << nblks;

    // 分配读取缓冲区，存储一个块的所有向量数据
    float *read_buf = new float[blk_size * ndims];
    
    // 逐块处理数据
    for (uint64_t i = 0; i < nblks; i++) {
      // 计算当前块的实际大小（最后一块可能不满）
      uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);

      // 从输入文件读取当前块的数据
      readr.read((char *) read_buf, cblk_size * ndims * sizeof(float));
      
      // 将维度转换为32位整数，用于循环优化
      uint32_t ndims_u32 = (uint32_t) ndims;
      
      // 使用 OpenMP 并行处理当前块中的每个向量
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t) cblk_size; i++) {
        // 计算向量的L2范数（模长）
        float norm_pt = std::numeric_limits<float>::epsilon();  // 初始化为epsilon，避免除零
        
        // 计算向量各分量的平方和
        for (uint32_t dim = 0; dim < ndims_u32; dim++) {
          norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
        }
        
        // 计算L2范数（平方根）
        norm_pt = std::sqrt(norm_pt);
        
        // 对向量进行归一化：每个分量除以L2范数
        for (uint32_t dim = 0; dim < ndims_u32; dim++) {
          *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
        }
      }
      
      // 将归一化后的数据写入输出文件
      writr.write((char *) read_buf, cblk_size * ndims * sizeof(float));
    }
    
    // 释放缓冲区内存
    delete[] read_buf;

    LOG(INFO) << "Wrote normalized points to file: " << outFileName;
  }
}  // namespace pipeann
