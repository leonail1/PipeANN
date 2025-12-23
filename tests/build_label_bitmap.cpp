/**
 * @file build_label_bitmap.cpp
 * @brief 预计算标签位图并保存到磁盘
 *
 * 生成 .bitmap 文件，供低内存版 search_prefilter 使用 mmap 加载。
 *
 * 文件格式：
 * - header: base_num (uint64), num_words (uint64), max_label_id (uint64)
 * - data: base_num * num_words * sizeof(uint64) 字节
 *
 * 用法:
 *   ./build_label_bitmap <base_label.spmat> <output.bitmap>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include "filter/label.h"
#include "utils/log.h"

static constexpr size_t MAX_LABELS = 65536;
static constexpr size_t BITS_PER_WORD = 64;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <base_label.spmat> <output.bitmap>" << std::endl;
    return 1;
  }

  std::string label_file = argv[1];
  std::string output_file = argv[2];

  LOG(INFO) << "Loading labels from: " << label_file;
  pipeann::SpmatLabel labels(label_file);

  uint64_t base_num = labels.labels_.size();
  if (base_num == 0) {
    LOG(ERROR) << "Empty label file";
    return 1;
  }

  uint64_t max_label_id = 0;
  for (const auto &lbl_list : labels.labels_) {
    for (uint32_t lbl : lbl_list) {
      if (lbl > max_label_id) max_label_id = lbl;
    }
  }

  if (max_label_id >= MAX_LABELS) {
    LOG(ERROR) << "Label ID " << max_label_id << " exceeds MAX_LABELS=" << MAX_LABELS;
    return 1;
  }

  uint64_t num_words = (max_label_id + BITS_PER_WORD) / BITS_PER_WORD;
  num_words = ((num_words + 7) / 8) * 8;

  LOG(INFO) << "base_num=" << base_num << ", max_label_id=" << max_label_id << ", num_words=" << num_words;

  std::ofstream out(output_file, std::ios::binary);
  if (!out) {
    LOG(ERROR) << "Failed to open output file: " << output_file;
    return 1;
  }

  out.write(reinterpret_cast<const char *>(&base_num), sizeof(uint64_t));
  out.write(reinterpret_cast<const char *>(&num_words), sizeof(uint64_t));
  out.write(reinterpret_cast<const char *>(&max_label_id), sizeof(uint64_t));

  std::vector<uint64_t> bitmap(num_words, 0);

  for (uint64_t i = 0; i < base_num; i++) {
    std::fill(bitmap.begin(), bitmap.end(), 0);
    for (uint32_t lbl : labels.labels_[i]) {
      bitmap[lbl / BITS_PER_WORD] |= (1ULL << (lbl % BITS_PER_WORD));
    }
    out.write(reinterpret_cast<const char *>(bitmap.data()), num_words * sizeof(uint64_t));

    if ((i + 1) % 100000 == 0) {
      LOG(INFO) << "Processed " << (i + 1) << "/" << base_num << " vectors";
    }
  }

  out.close();

  size_t file_size = 24 + base_num * num_words * sizeof(uint64_t);
  LOG(INFO) << "Bitmap saved to: " << output_file << " (" << file_size / 1024 / 1024 << " MB)";

  return 0;
}
