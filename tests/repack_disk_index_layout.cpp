#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ssd_index_defs.h"
#include "utils.h"

namespace {

constexpr uint64_t kSupersector32K = 8 * SECTOR_LEN;

void read_exact(std::ifstream &in, char *dst, uint64_t len, const std::string &what) {
  in.read(dst, static_cast<std::streamsize>(len));
  if (!in.good()) {
    throw std::runtime_error("short read while reading " + what);
  }
}

template<typename T>
uint64_t record_file_offset(const pipeann::SSDIndexMetadata<T> &meta, uint64_t loc) {
	  if (meta.uses_packed_layout()) {
	    const uint64_t block_id = loc / meta.layout_nodes_per_block;
	    const uint64_t in_block = loc % meta.layout_nodes_per_block;
	    return SECTOR_LEN + block_id * meta.layout_block_bytes + meta.packed_slot_offset(in_block);
	  }
  return SECTOR_LEN + (meta.nnodes_per_sector > 0
                           ? (loc / meta.nnodes_per_sector) * SECTOR_LEN
                                 + (loc % meta.nnodes_per_sector) * meta.max_node_len
                           : loc * ROUND_UP(meta.max_node_len, SECTOR_LEN));
}

template<typename T>
uint64_t packed_4k_pages_to_read(const pipeann::SSDIndexMetadata<T> &meta, uint64_t loc) {
  const uint64_t offset = record_file_offset(meta, loc) % SECTOR_LEN;
  return DIV_ROUND_UP(offset + meta.max_node_len, SECTOR_LEN);
}

void copy_if_exists(const std::string &src_prefix, const std::string &dst_prefix, const std::string &suffix,
                    std::vector<std::string> *copied) {
  const std::filesystem::path src = src_prefix + suffix;
  if (!std::filesystem::exists(src)) {
    return;
  }
  const std::filesystem::path dst = dst_prefix + suffix;
  std::filesystem::create_directories(dst.parent_path());
  std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
  copied->push_back(suffix);
}

void remove_stale_v1_layout_sidecars(const std::string &dst_prefix) {
  for (const std::string &suffix : {"_partition.bin", "_partition.bin.aligned"}) {
    std::error_code ec;
    std::filesystem::remove(dst_prefix + suffix, ec);
  }
}

template<typename T>
int repack_supersector32k(const std::string &src_prefix, const std::string &dst_prefix) {
  const std::string src_disk = src_prefix + "_disk.index";
  const std::string dst_disk = dst_prefix + "_disk.index";

  pipeann::SSDIndexMetadata<T> src_meta;
  src_meta.load_from_disk_index(src_disk);
  if (src_meta.uses_packed_layout()) {
    throw std::runtime_error("source index is already a packed layout");
  }
  if (src_meta.max_node_len == 0 || src_meta.max_node_len > kSupersector32K) {
    throw std::runtime_error("invalid max_node_len for supersector32k packing");
  }

  pipeann::SSDIndexMetadata<T> dst_meta = src_meta;
	  dst_meta.disk_layout_version = pipeann::SSDIndexMetadata<T>::kLayoutV3Supersector32KPageAware;
  dst_meta.layout_block_bytes = kSupersector32K;
  dst_meta.layout_nodes_per_block = kSupersector32K / src_meta.max_node_len;
  dst_meta.layout_read_page_bytes = SECTOR_LEN;
  dst_meta.init_temporary_fields();
  if (dst_meta.layout_nodes_per_block == 0) {
    throw std::runtime_error("node record does not fit in a 32KB packing block");
  }
  if (dst_meta.max_node_len > SECTOR_LEN) {
    throw std::runtime_error("supersector32k v2 currently supports max_node_len <= 4096 so each record needs at most two 4KB reads");
  }

  std::ifstream src(src_disk, std::ios::binary);
  if (!src.is_open()) {
    throw std::runtime_error("failed to open source disk index: " + src_disk);
  }

  std::filesystem::create_directories(std::filesystem::path(dst_disk).parent_path());
  remove_stale_v1_layout_sidecars(dst_prefix);
  std::ofstream dst(dst_disk, std::ios::binary | std::ios::trunc);
  if (!dst.is_open()) {
    throw std::runtime_error("failed to open destination disk index: " + dst_disk);
  }

  std::vector<char> zero_sector(SECTOR_LEN, 0);
  dst.write(zero_sector.data(), static_cast<std::streamsize>(zero_sector.size()));

  std::vector<char> dst_block(kSupersector32K, 0);
  std::vector<char> src_sector(SECTOR_LEN, 0);
  std::vector<char> src_record(src_meta.max_node_len, 0);
  uint64_t cached_src_sector = std::numeric_limits<uint64_t>::max();

  uint64_t straddling_records = 0;
  for (uint64_t block_start = 0; block_start < src_meta.npoints; block_start += dst_meta.layout_nodes_per_block) {
    std::fill(dst_block.begin(), dst_block.end(), 0);
    const uint64_t block_end = std::min(src_meta.npoints, block_start + dst_meta.layout_nodes_per_block);

    for (uint64_t loc = block_start; loc < block_end; ++loc) {
      const uint64_t src_offset = record_file_offset(src_meta, loc);
	      char *record = dst_block.data() + dst_meta.packed_slot_offset(loc - block_start);
      if (src_meta.nnodes_per_sector > 0 && src_meta.max_node_len <= SECTOR_LEN) {
        const uint64_t sector = src_offset / SECTOR_LEN;
        if (sector != cached_src_sector) {
          src.seekg(static_cast<std::streamoff>(sector * SECTOR_LEN), std::ios::beg);
          read_exact(src, src_sector.data(), SECTOR_LEN, "source sector");
          cached_src_sector = sector;
        }
        std::memcpy(record, src_sector.data() + (src_offset % SECTOR_LEN), src_meta.max_node_len);
      } else {
        src.seekg(static_cast<std::streamoff>(src_offset), std::ios::beg);
        read_exact(src, src_record.data(), src_meta.max_node_len, "source node record");
        std::memcpy(record, src_record.data(), src_meta.max_node_len);
      }
      if (packed_4k_pages_to_read(dst_meta, loc) > 1) {
        ++straddling_records;
      }
    }
    dst.write(dst_block.data(), static_cast<std::streamsize>(dst_block.size()));
  }
  dst.close();

  dst_meta.save_to_disk_index(dst_disk);

  std::vector<std::string> copied_sidecars;
  for (const std::string &suffix : {"_pq_compressed.bin", "_pq_pivots.bin", "_labels.densebit", "_hybrid.meta",
                                    "_disk.index.tags", "_mem.index", "_mem.index.tags"}) {
    copy_if_exists(src_prefix, dst_prefix, suffix, &copied_sidecars);
  }

  const uint64_t blocks = DIV_ROUND_UP(src_meta.npoints, dst_meta.layout_nodes_per_block);
  const uint64_t expected_disk_bytes = SECTOR_LEN + blocks * dst_meta.layout_block_bytes;
  const uint64_t actual_disk_bytes = std::filesystem::file_size(dst_disk);
  std::cout << "{"
            << "\"src_prefix\":\"" << src_prefix << "\","
            << "\"dst_prefix\":\"" << dst_prefix << "\","
	            << "\"layout\":\"supersector32k\","
	            << "\"layout_version\":" << dst_meta.disk_layout_version << ","
	            << "\"layout_variant\":\"page_aware_slots\","
	            << "\"npoints\":" << src_meta.npoints << ","
            << "\"max_node_len\":" << src_meta.max_node_len << ","
            << "\"layout_nodes_per_block\":" << dst_meta.layout_nodes_per_block << ","
            << "\"layout_block_bytes\":" << dst_meta.layout_block_bytes << ","
            << "\"read_page_bytes\":" << dst_meta.layout_read_page_bytes << ","
            << "\"expected_disk_bytes\":" << expected_disk_bytes << ","
            << "\"actual_disk_bytes\":" << actual_disk_bytes << ","
	            << "\"straddling_records\":" << straddling_records << ","
	            << "\"straddling_slots_per_block\":" << dst_meta.packed_slot_straddling_count() << ","
            << "\"straddling_fraction\":" << std::fixed << std::setprecision(6)
            << (src_meta.npoints == 0 ? 0.0 : static_cast<double>(straddling_records) / src_meta.npoints) << ","
            << "\"avg_4k_pages_per_record\":" << std::fixed << std::setprecision(6)
            << (src_meta.npoints == 0 ? 0.0 : 1.0 + static_cast<double>(straddling_records) / src_meta.npoints)
            << ",\"copied_sidecars\":[";
  for (size_t i = 0; i < copied_sidecars.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << copied_sidecars[i] << "\"";
  }
  std::cout << "]}" << std::endl;
  return actual_disk_bytes == expected_disk_bytes ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <float|uint8|int8> <src_prefix> <dst_prefix> supersector32k\n";
    return 1;
  }
  const std::string dtype = argv[1];
  const std::string src_prefix = argv[2];
  const std::string dst_prefix = argv[3];
  const std::string layout = argv[4];
  if (layout != "supersector32k") {
    std::cerr << "unsupported layout: " << layout << "\n";
    return 1;
  }

  try {
    if (dtype == "float") {
      return repack_supersector32k<float>(src_prefix, dst_prefix);
    }
    if (dtype == "uint8") {
      return repack_supersector32k<uint8_t>(src_prefix, dst_prefix);
    }
    if (dtype == "int8") {
      return repack_supersector32k<int8_t>(src_prefix, dst_prefix);
    }
    std::cerr << "unsupported dtype: " << dtype << "\n";
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "repack failed: " << ex.what() << "\n";
    return 1;
  }
}
