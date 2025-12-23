/**
 * @file bitmap_selector.h
 * @brief 基于位图的高性能标签过滤器
 *
 * 本文件实现了使用位图(Bitmap)数据结构进行标签过滤的功能。
 * 相比原始的双重循环实现，位图方法将标签集合表示为二进制位，
 * 从而可以利用CPU的位运算指令(AND/OR)和SIMD向量化指令(AVX2/AVX512)
 * 来大幅加速集合的交集和子集判断操作。
 *
 * 性能优势：
 * - 原始方法：O(query_count * target_count) 的双重循环
 * - 位图方法：O(num_words) ≈ O(1) 的位运算，其中 num_words = max_label_id / 64
 * - 使用SIMD后，每次可并行处理 256位(AVX2) 或 512位(AVX512)
 *
 * 使用限制：
 * - 标签ID必须小于 MAX_LABELS (65536)
 * - 如果标签ID超出范围，需要回退到原始实现
 */

#ifndef BITMAP_SELECTOR_H_
#define BITMAP_SELECTOR_H_

#include <omp.h>
#include <immintrin.h>  // SIMD 指令集头文件 (AVX2/AVX512)
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "filter/label.h"

namespace pipeann {

  /**
   * @brief 位图过滤上下文类
   *
   * 该类负责：
   * 1. 预计算所有基础数据点的标签位图（在初始化时一次性完成）
   * 2. 提供高效的位图构建和比较操作
   * 3. 根据CPU支持的指令集自动选择最优实现（AVX512 > AVX2 > Scalar）
   */
  class BitmapFilterContext {
   public:
    /**
     * 最大支持的标签数量。
     * 65536个标签需要 65536/64 = 1024 个 uint64_t 来存储位图。
     * 每个位图占用 8KB 内存。
     */
    static constexpr size_t MAX_LABELS = 65536;

    /**
     * 每个 uint64_t 可以存储64个标签的存在性信息。
     * 标签 i 对应位图中的第 (i/64) 个word的第 (i%64) 位。
     */
    static constexpr size_t BITS_PER_WORD = 64;

    /**
     * 位图数组的最大长度（以 uint64_t 为单位）
     */
    static constexpr size_t MAX_WORDS = MAX_LABELS / BITS_PER_WORD;

   private:
    /**
     * 预计算的基础数据点位图。
     * base_bitmaps_[i] 是第 i 个数据点的标签位图。
     * 在 init() 时一次性计算完成，后续过滤操作只需读取。
     */
    std::vector<std::vector<uint64_t>> base_bitmaps_;

    /**
     * 实际使用的位图长度（以 uint64_t 为单位）。
     * 由 max_label_id_ 决定，并向上对齐到8的倍数以适配 AVX512。
     */
    size_t num_words_ = 0;

    /**
     * 数据集中出现的最大标签ID
     */
    size_t max_label_id_ = 0;

    /**
     * 基础数据点的数量
     */
    size_t base_num_ = 0;

    /**
     * 标记位图是否成功初始化。
     * 如果标签ID超出范围，valid_ 将为 false。
     */
    bool valid_ = false;

   public:
    BitmapFilterContext() = default;

    /**
     * @brief 初始化位图过滤上下文
     *
     * 该函数执行以下操作：
     * 1. 扫描所有标签，找出最大标签ID
     * 2. 检查标签ID是否在支持范围内
     * 3. 为每个基础数据点预计算位图
     *
     * 时间复杂度：O(base_num * avg_labels_per_point)
     * 空间复杂度：O(base_num * num_words)
     *
     * @param base_labels 基础数据的标签信息（稀疏矩阵格式）
     * @return true 初始化成功，可以使用位图过滤
     * @return false 初始化失败（标签ID超出范围或数据为空）
     */
    bool init(const SpmatLabel &base_labels) {
      base_num_ = base_labels.labels_.size();
      if (base_num_ == 0) {
        return false;
      }

      // 第一遍扫描：找出最大标签ID
      max_label_id_ = 0;
      for (const auto &labels : base_labels.labels_) {
        for (uint32_t lbl : labels) {
          if (lbl > max_label_id_) {
            max_label_id_ = lbl;
          }
        }
      }

      // 检查标签ID是否在支持范围内
      if (max_label_id_ >= MAX_LABELS) {
        return false;
      }

      // 计算位图长度，向上对齐到8的倍数（适配AVX512的512位 = 8个uint64）
      num_words_ = (max_label_id_ + BITS_PER_WORD) / BITS_PER_WORD;
      num_words_ = ((num_words_ + 7) / 8) * 8;

      // 第二遍扫描：为每个数据点构建位图
      base_bitmaps_.resize(base_num_);
      for (size_t i = 0; i < base_num_; i++) {
        // 分配并清零位图
        base_bitmaps_[i].resize(num_words_, 0);

        // 设置对应标签的位
        for (uint32_t lbl : base_labels.labels_[i]) {
          // 标签 lbl 对应第 (lbl/64) 个word的第 (lbl%64) 位
          base_bitmaps_[i][lbl / BITS_PER_WORD] |= (1ULL << (lbl % BITS_PER_WORD));
        }
      }

      valid_ = true;
      return true;
    }

    // ==================== 状态查询接口 ====================

    bool is_valid() const {
      return valid_;
    }
    size_t num_words() const {
      return num_words_;
    }
    size_t max_label_id() const {
      return max_label_id_;
    }
    size_t base_num() const {
      return base_num_;
    }

    /**
     * @brief 获取指定数据点的位图指针
     * @param idx 数据点索引
     * @return 指向位图数组的常量指针
     */
    const uint64_t *get_base_bitmap(size_t idx) const {
      return base_bitmaps_[idx].data();
    }

    // ==================== 查询位图构建 ====================

    /**
     * @brief 从标签向量构建查询位图
     *
     * @param query_labels 查询的标签ID向量
     * @param out_bitmap 输出位图（调用者需预分配 num_words_ 个 uint64_t）
     */
    void build_query_bitmap(const std::vector<uint32_t> &query_labels, uint64_t *out_bitmap) const {
      // 清零位图
      std::memset(out_bitmap, 0, num_words_ * sizeof(uint64_t));

      // 设置每个标签对应的位
      for (uint32_t lbl : query_labels) {
        if (lbl < MAX_LABELS) {
          out_bitmap[lbl / BITS_PER_WORD] |= (1ULL << (lbl % BITS_PER_WORD));
        }
      }
    }

    /**
     * @brief 从序列化的标签缓冲区构建查询位图
     *
     * 缓冲区格式：[count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
     *
     * @param query_labels_buf 序列化的标签缓冲区
     * @param out_bitmap 输出位图（调用者需预分配 num_words_ 个 uint64_t）
     */
    void build_query_bitmap(const void *query_labels_buf, uint64_t *out_bitmap) const {
      std::memset(out_bitmap, 0, num_words_ * sizeof(uint64_t));

      // 读取标签数量
      uint32_t count;
      std::memcpy(&count, query_labels_buf, sizeof(uint32_t));

      // 获取标签数组指针
      const uint32_t *labels =
          reinterpret_cast<const uint32_t *>(static_cast<const char *>(query_labels_buf) + sizeof(uint32_t));

      // 设置每个标签对应的位
      for (uint32_t i = 0; i < count; i++) {
        uint32_t lbl = labels[i];
        if (lbl < MAX_LABELS) {
          out_bitmap[lbl / BITS_PER_WORD] |= (1ULL << (lbl % BITS_PER_WORD));
        }
      }
    }

    // ==================== AVX512 SIMD 实现 ====================

#ifdef __AVX512F__
    /**
     * @brief 使用 AVX512 检测两个位图是否有交集
     *
     * 算法：计算 (a[i] & b[i]) 的累积OR，如果结果非零则有交集。
     * AVX512 每次处理 512位 = 8个 uint64_t。
     *
     * @param a 位图A
     * @param b 位图B
     * @param n 位图长度（uint64_t数量）
     * @return true 存在交集
     * @return false 无交集
     */
    static bool bitmap_has_intersection_avx512(const uint64_t *a, const uint64_t *b, size_t n) {
      // 累加器，用于收集所有AND结果
      __m512i acc = _mm512_setzero_si512();
      size_t i = 0;

      // 主循环：每次处理8个uint64（512位）
      for (; i + 8 <= n; i += 8) {
        // 加载两个512位向量
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));

        // 累积 OR(AND(va, vb))
        acc = _mm512_or_si512(acc, _mm512_and_si512(va, vb));
      }

      // 检查累加器是否非零
      if (_mm512_test_epi64_mask(acc, acc)) {
        return true;
      }

      // 处理剩余元素（标量方式）
      for (; i < n; i++) {
        if (a[i] & b[i])
          return true;
      }
      return false;
    }

    /**
     * @brief 使用 AVX512 检测 query 是否是 target 的子集
     *
     * 子集条件：对于所有位，如果 query 中某位为1，则 target 中对应位也必须为1。
     * 等价于：(query & target) == query，即 query ^ (query & target) == 0
     *
     * @param query 查询位图
     * @param target 目标位图
     * @param n 位图长度（uint64_t数量）
     * @return true query 是 target 的子集
     * @return false query 不是 target 的子集
     */
    static bool bitmap_is_subset_avx512(const uint64_t *query, const uint64_t *target, size_t n) {
      size_t i = 0;

      for (; i + 8 <= n; i += 8) {
        __m512i vq = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + i));
        __m512i vt = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(target + i));

        // masked = query & target
        __m512i masked = _mm512_and_si512(vq, vt);

        // diff = query ^ masked，如果 query 是子集则 diff 应该为0
        __m512i diff = _mm512_xor_si512(vq, masked);

        // 如果 diff 非零，说明存在 query 有但 target 没有的位
        if (_mm512_test_epi64_mask(diff, diff)) {
          return false;
        }
      }

      // 处理剩余元素
      for (; i < n; i++) {
        if ((query[i] & target[i]) != query[i])
          return false;
      }
      return true;
    }
#endif

    // ==================== AVX2 SIMD 实现 ====================

#ifdef __AVX2__
    /**
     * @brief 使用 AVX2 检测两个位图是否有交集
     *
     * AVX2 每次处理 256位 = 4个 uint64_t。
     *
     * @param a 位图A
     * @param b 位图B
     * @param n 位图长度（uint64_t数量）
     * @return true 存在交集
     * @return false 无交集
     */
    static bool bitmap_has_intersection_avx2(const uint64_t *a, const uint64_t *b, size_t n) {
      __m256i acc = _mm256_setzero_si256();
      size_t i = 0;

      // 主循环：每次处理4个uint64（256位）
      for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
        acc = _mm256_or_si256(acc, _mm256_and_si256(va, vb));
      }

      // _mm256_testz_si256(a,a) 返回1当a全为0，返回0当a有非零位
      if (!_mm256_testz_si256(acc, acc)) {
        return true;
      }

      // 处理剩余元素
      for (; i < n; i++) {
        if (a[i] & b[i])
          return true;
      }
      return false;
    }

    /**
     * @brief 使用 AVX2 检测 query 是否是 target 的子集
     *
     * @param query 查询位图
     * @param target 目标位图
     * @param n 位图长度（uint64_t数量）
     * @return true query 是 target 的子集
     * @return false query 不是 target 的子集
     */
    static bool bitmap_is_subset_avx2(const uint64_t *query, const uint64_t *target, size_t n) {
      size_t i = 0;

      for (; i + 4 <= n; i += 4) {
        __m256i vq = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + i));
        __m256i vt = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(target + i));
        __m256i masked = _mm256_and_si256(vq, vt);
        __m256i diff = _mm256_xor_si256(vq, masked);

        if (!_mm256_testz_si256(diff, diff)) {
          return false;
        }
      }

      for (; i < n; i++) {
        if ((query[i] & target[i]) != query[i])
          return false;
      }
      return true;
    }
#endif

    // ==================== 标量实现（无SIMD回退） ====================

    /**
     * @brief 标量方式检测两个位图是否有交集
     *
     * 当CPU不支持AVX2/AVX512时使用此实现。
     */
    static bool bitmap_has_intersection_scalar(const uint64_t *a, const uint64_t *b, size_t n) {
      for (size_t i = 0; i < n; i++) {
        if (a[i] & b[i])
          return true;
      }
      return false;
    }

    /**
     * @brief 标量方式检测 query 是否是 target 的子集
     */
    static bool bitmap_is_subset_scalar(const uint64_t *query, const uint64_t *target, size_t n) {
      for (size_t i = 0; i < n; i++) {
        if ((query[i] & target[i]) != query[i])
          return false;
      }
      return true;
    }

    // ==================== 自动分派函数 ====================

    /**
     * @brief 检测两个位图是否有交集（自动选择最优实现）
     *
     * 编译时根据CPU支持的指令集选择：AVX512F > AVX2 > Scalar
     */
    static bool bitmap_has_intersection(const uint64_t *a, const uint64_t *b, size_t n) {
#ifdef __AVX512F__
      return bitmap_has_intersection_avx512(a, b, n);
#elif defined(__AVX2__)
      return bitmap_has_intersection_avx2(a, b, n);
#else
      return bitmap_has_intersection_scalar(a, b, n);
#endif
    }

    /**
     * @brief 检测 query 是否是 target 的子集（自动选择最优实现）
     */
    static bool bitmap_is_subset(const uint64_t *query, const uint64_t *target, size_t n) {
#ifdef __AVX512F__
      return bitmap_is_subset_avx512(query, target, n);
#elif defined(__AVX2__)
      return bitmap_is_subset_avx2(query, target, n);
#else
      return bitmap_is_subset_scalar(query, target, n);
#endif
    }

    // ==================== 批量过滤接口 ====================

    /**
     * @brief 过滤出所有与查询标签有交集的数据点
     *
     * 遍历所有基础数据点，使用位图交集检测筛选匹配的点。
     *
     * @param query_bitmap 查询标签的位图表示
     * @param out_matching 输出：匹配的数据点ID列表
     */
    void filter_intersect(const uint64_t *query_bitmap, std::vector<uint32_t> &out_matching) const {
      out_matching.clear();

      std::vector<uint8_t> match_flags(base_num_, 0);

#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < base_num_; i++) {
        if (bitmap_has_intersection(query_bitmap, base_bitmaps_[i].data(), num_words_)) {
          match_flags[i] = 1;
        }
      }

      out_matching.reserve(base_num_ / 10);
      for (size_t i = 0; i < base_num_; i++) {
        if (match_flags[i]) {
          out_matching.push_back(static_cast<uint32_t>(i));
        }
      }
    }

    /**
     * @brief 过滤出所有包含查询标签的数据点（查询是目标的子集）
     *
     * 用于 "数据点必须包含查询指定的所有标签" 的场景。
     *
     * @param query_bitmap 查询标签的位图表示
     * @param out_matching 输出：匹配的数据点ID列表
     */
    void filter_subset(const uint64_t *query_bitmap, std::vector<uint32_t> &out_matching) const {
      out_matching.clear();

      std::vector<uint8_t> match_flags(base_num_, 0);

#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < base_num_; i++) {
        if (bitmap_is_subset(query_bitmap, base_bitmaps_[i].data(), num_words_)) {
          match_flags[i] = 1;
        }
      }

      out_matching.reserve(base_num_ / 10);
      for (size_t i = 0; i < base_num_; i++) {
        if (match_flags[i]) {
          out_matching.push_back(static_cast<uint32_t>(i));
        }
      }
    }

#ifdef __AVX2__
    /**
     * @brief 批量过滤（实验性）
     *
     * 每次处理8个数据点，使用位掩码收集结果。
     * 在某些场景下可能比逐个处理更快。
     *
     * @param query_bitmap 查询标签的位图表示
     * @param out_matching 输出：匹配的数据点ID列表
     */
    void filter_intersect_batch(const uint64_t *query_bitmap, std::vector<uint32_t> &out_matching) const {
      out_matching.clear();
      out_matching.reserve(base_num_ / 10);

      constexpr size_t BATCH_SIZE = 8;
      size_t i = 0;

      // 批量处理
      for (; i + BATCH_SIZE <= base_num_; i += BATCH_SIZE) {
        uint8_t results = 0;  // 8个结果的位掩码

        // 检测这8个数据点
        for (size_t j = 0; j < BATCH_SIZE; j++) {
          if (bitmap_has_intersection_avx2(query_bitmap, base_bitmaps_[i + j].data(), num_words_)) {
            results |= (1 << j);
          }
        }

        // 使用 ctz (count trailing zeros) 快速提取匹配的索引
        while (results) {
          int idx = __builtin_ctz(results);  // 找到最低位的1
          out_matching.push_back(static_cast<uint32_t>(i + idx));
          results &= (results - 1);  // 清除最低位的1
        }
      }

      // 处理剩余的数据点
      for (; i < base_num_; i++) {
        if (bitmap_has_intersection_avx2(query_bitmap, base_bitmaps_[i].data(), num_words_)) {
          out_matching.push_back(static_cast<uint32_t>(i));
        }
      }
    }
#endif
  };

}  // namespace pipeann

#endif  // BITMAP_SELECTOR_H_