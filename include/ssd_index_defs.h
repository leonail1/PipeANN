#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "utils.h"
#include "utils/tsl/robin_set.h"

constexpr size_t MAX_N_SECTOR_READS = 128;
constexpr size_t MAX_N_NODE_READ_PAGES = 2;
constexpr size_t MAX_N_EDGES = 1024;
constexpr size_t INDEX_SIZE_FACTOR = 2;  // space amplification during insert.

// Both unaligned and aligned.
// example: a record locates in [300, 500], then
// offset = 0, len = 4096 (aligned read for disk)
// u_offset = 300, u_len = 200 (unaligned read)
// Unaligned read: read u_len from u_offset, read to buf + 0.
struct IORequest {
  uint64_t offset;    // where to read from (page)
  uint64_t len;       // how much to read
  void *buf;          // where to read into
  bool finished;      // for async IO
  uint64_t u_offset;  // where to read from (unaligned)
  uint64_t u_len;     // how much to read (unaligned)
  void *base;         // starting address of this sector scratch

  IORequest() : offset(0), len(0), buf(nullptr) {
  }

  IORequest(uint64_t offset, uint64_t len, void *buf, uint64_t u_offset, uint64_t u_len, void *base = nullptr)
      : offset(offset), len(len), buf(buf), u_offset(u_offset), u_len(u_len), base(base) {
    assert((uint64_t) buf % SECTOR_LEN == 0);
    assert(offset % SECTOR_LEN == 0);
    assert(len % SECTOR_LEN == 0);
  }
};

namespace pipeann {
  template<typename T, typename IdT = uint32_t>
  struct SSDIndexMetadata {
    // The order matches that on SSD.
    uint32_t nr, nc;
    uint64_t npoints;  // size.
    uint64_t data_dim;
    uint64_t entry_point;
    uint64_t max_node_len;  // without data.
    uint64_t nnodes_per_sector;
    uint64_t npts_cur_shard;
	    uint64_t label_size;
	    uint64_t disk_layout_version = 1;
	    uint64_t layout_block_bytes = SECTOR_LEN;
	    uint64_t layout_nodes_per_block = 0;
	    uint64_t layout_read_page_bytes = SECTOR_LEN;
	    uint64_t max_npts;  // capacity.
	    uint64_t range;     // maximum out-degree.

	    /* temporary fields (currently not stored on disk). */
	    IdT entry_point_id;
	    enum DataType : uint64_t { UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3 } data_type;  // currently unused.
		    std::vector<uint32_t> layout_slot_offsets;

    SSDIndexMetadata() {
    }

	    SSDIndexMetadata(uint64_t npoints, uint64_t data_dim, uint64_t entry_point, uint64_t max_node_len,
	                     uint64_t nnodes_per_sector, uint64_t label_size = 0)
	        : npoints(npoints), data_dim(data_dim), entry_point(entry_point), max_node_len(max_node_len),
	          nnodes_per_sector(nnodes_per_sector), npts_cur_shard(npoints), label_size(label_size), data_type(UNDEFINED) {
	      this->init_layout_defaults();
	      this->init_temporary_fields();
	    }

		    static constexpr uint64_t kLayoutV1Sector = 1;
		    static constexpr uint64_t kLayoutV2Supersector32K = 2;
		    static constexpr uint64_t kLayoutV3Supersector32KPageAware = 3;

		    bool uses_packed_layout() const {
		      return disk_layout_version == kLayoutV2Supersector32K ||
		             disk_layout_version == kLayoutV3Supersector32KPageAware;
		    }

		    bool uses_page_aware_packed_slots() const {
		      return disk_layout_version == kLayoutV3Supersector32KPageAware;
		    }

		    uint64_t packed_slot_offset(uint64_t slot) const {
		      if (uses_page_aware_packed_slots() && slot < layout_slot_offsets.size()) {
		        return layout_slot_offsets[slot];
		      }
		      return slot * max_node_len;
		    }

		    uint64_t packed_slot_straddling_count() const {
		      if (!uses_packed_layout() || layout_nodes_per_block == 0 || layout_read_page_bytes == 0) {
		        return 0;
		      }
		      uint64_t count = 0;
		      for (uint64_t slot = 0; slot < layout_nodes_per_block; ++slot) {
		        const uint64_t offset = packed_slot_offset(slot) % layout_read_page_bytes;
		        if (offset + max_node_len > layout_read_page_bytes) {
		          ++count;
		        }
		      }
		      return count;
		    }

		    void init_packed_slot_offsets() {
		      layout_slot_offsets.clear();
		      if (!uses_page_aware_packed_slots() || layout_nodes_per_block == 0 || max_node_len == 0 ||
		          layout_read_page_bytes == 0 || max_node_len > layout_read_page_bytes ||
		          max_node_len > layout_block_bytes || layout_nodes_per_block > layout_block_bytes / max_node_len ||
		          layout_block_bytes > std::numeric_limits<uint32_t>::max()) {
		        return;
		      }

		      const uint64_t dense_bytes = layout_nodes_per_block * max_node_len;
		      const uint64_t padding_budget_u64 = layout_block_bytes - dense_bytes;
		      if (padding_budget_u64 > 4096 || layout_nodes_per_block > 1024) {
		        return;
		      }

		      const size_t nslots = static_cast<size_t>(layout_nodes_per_block);
		      const size_t padding_budget = static_cast<size_t>(padding_budget_u64);
		      auto straddles = [&](size_t slot, size_t prefix_padding) -> uint32_t {
		        const uint64_t offset = static_cast<uint64_t>(slot) * max_node_len + prefix_padding;
		        return (offset % layout_read_page_bytes) + max_node_len > layout_read_page_bytes ? 1U : 0U;
		      };

		      // Choose prefix padding once at load/repack time so hot search paths only do a table lookup.
		      constexpr uint32_t kInf = std::numeric_limits<uint32_t>::max() / 4;
		      std::vector<uint32_t> current(padding_budget + 1, kInf), next(padding_budget + 1, kInf);
		      std::vector<std::vector<uint16_t>> previous(nslots, std::vector<uint16_t>(padding_budget + 1, UINT16_MAX));
		      current[0] = 0;

		      for (size_t slot = 0; slot + 1 < nslots; ++slot) {
		        std::fill(next.begin(), next.end(), kInf);
		        for (size_t pad = 0; pad <= padding_budget; ++pad) {
		          if (current[pad] == kInf) {
		            continue;
		          }
		          const uint32_t cost = current[pad] + straddles(slot, pad);
		          for (size_t next_pad = pad; next_pad <= padding_budget; ++next_pad) {
		            if (cost < next[next_pad]) {
		              next[next_pad] = cost;
		              previous[slot + 1][next_pad] = static_cast<uint16_t>(pad);
		            }
		          }
		        }
		        current.swap(next);
		      }

		      uint32_t best_cost = kInf;
		      size_t best_pad = 0;
		      for (size_t pad = 0; pad <= padding_budget; ++pad) {
		        if (current[pad] == kInf) {
		          continue;
		        }
		        const uint32_t cost = current[pad] + straddles(nslots - 1, pad);
		        if (cost < best_cost) {
		          best_cost = cost;
		          best_pad = pad;
		        }
		      }

		      std::vector<size_t> prefix_padding(nslots, 0);
		      prefix_padding[nslots - 1] = best_pad;
		      for (size_t slot = nslots - 1; slot > 0; --slot) {
		        const uint16_t prev = previous[slot][prefix_padding[slot]];
		        prefix_padding[slot - 1] = prev == UINT16_MAX ? 0 : prev;
		      }

		      layout_slot_offsets.resize(nslots);
		      for (size_t slot = 0; slot < nslots; ++slot) {
		        layout_slot_offsets[slot] = static_cast<uint32_t>(slot * max_node_len + prefix_padding[slot]);
		      }
		    }

	    void init_layout_defaults() {
	      if (disk_layout_version == 0) {
	        disk_layout_version = kLayoutV1Sector;
	      }
	      if (layout_read_page_bytes == 0) {
	        layout_read_page_bytes = SECTOR_LEN;
	      }
	      if (layout_block_bytes == 0) {
	        layout_block_bytes = uses_packed_layout() ? 8 * SECTOR_LEN : SECTOR_LEN;
	      }
	      if (layout_nodes_per_block == 0) {
	        layout_nodes_per_block = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
	      }
	    }

		    void init_temporary_fields() {
		      this->init_layout_defaults();
		      this->init_packed_slot_offsets();
		      this->max_npts = npoints;
		      this->range = (max_node_len - data_dim * sizeof(T) - label_size) / sizeof(unsigned) - 1;
	      this->entry_point_id = static_cast<IdT>(entry_point);
	      assert(entry_point_id == entry_point);
	    }

    void print() const {
      LOG(INFO) << "Max npts: " << max_npts << " Npoints: " << npoints << " Entry point: " << entry_point
                << " Data dim: " << data_dim << " Range: " << range;
	      LOG(INFO) << "Max node len: " << max_node_len << " Nnodes per sector: " << nnodes_per_sector
	                << " Npts cur shard: " << npts_cur_shard << " Label size: " << label_size;
	      LOG(INFO) << "Disk layout version: " << disk_layout_version << " Block bytes: " << layout_block_bytes
	                << " Nodes per block: " << layout_nodes_per_block
	                << " Read page bytes: " << layout_read_page_bytes;
	    }

    void load_from_disk_index(const std::string &filename, bool sharded = false) {
      if (file_exists(filename) == false) {
        LOG(ERROR) << "File " << filename << " does not exist.";
        exit(-1);
      }
      std::ifstream in(filename, std::ios::binary);
      load_from_disk_index(in, sharded);
      in.close();
    }

    void load_from_disk_index(std::ifstream &in, bool sharded = false) {
      LOG(INFO) << "Loading metadata from disk index, sharded: " << sharded;
      in.read((char *) &nr, sizeof(uint32_t));
      in.read((char *) &nc, sizeof(uint32_t));

      in.read((char *) &npoints, sizeof(uint64_t));
      in.read((char *) &data_dim, sizeof(uint64_t));

      in.read((char *) &entry_point, sizeof(uint64_t));
      in.read((char *) &max_node_len, sizeof(uint64_t));
      in.read((char *) &nnodes_per_sector, sizeof(uint64_t));
	      in.read((char *) &npts_cur_shard, sizeof(uint64_t));
	      in.read((char *) &label_size, sizeof(uint64_t));

	      disk_layout_version = kLayoutV1Sector;
	      layout_block_bytes = SECTOR_LEN;
	      layout_nodes_per_block = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
	      layout_read_page_bytes = SECTOR_LEN;
	      if (nr >= 11) {
	        in.read((char *) &disk_layout_version, sizeof(uint64_t));
	        in.read((char *) &layout_block_bytes, sizeof(uint64_t));
	        in.read((char *) &layout_nodes_per_block, sizeof(uint64_t));
	        in.read((char *) &layout_read_page_bytes, sizeof(uint64_t));
	      }

      if (!sharded) {
        this->npts_cur_shard = this->npoints;
      }

      if (nr < 7) {  // backward compatible.
        this->label_size = 0;
      }

      this->init_temporary_fields();
    }

    void save_to_disk_index(const std::string &filename) {
      std::ofstream out(filename, std::ios::in | std::ios::out | std::ios::binary);
      save_to_disk_index(out);
      out.close();
    }

    void save_to_disk_index(std::ofstream &out) {
	      this->init_layout_defaults();
	      nr = uses_packed_layout() ? 11 : 7;  // hard-coded for the number of uint64_t below.
	      nc = 1;
      out.write((char *) &nr, sizeof(uint32_t));
      out.write((char *) &nc, sizeof(uint32_t));

      out.write((char *) &npoints, sizeof(uint64_t));
      out.write((char *) &data_dim, sizeof(uint64_t));

      out.write((char *) &entry_point, sizeof(uint64_t));
      out.write((char *) &max_node_len, sizeof(uint64_t));
	      out.write((char *) &nnodes_per_sector, sizeof(uint64_t));
	      out.write((char *) &npts_cur_shard, sizeof(uint64_t));
	      out.write((char *) &label_size, sizeof(uint64_t));
	      if (nr >= 11) {
	        out.write((char *) &disk_layout_version, sizeof(uint64_t));
	        out.write((char *) &layout_block_bytes, sizeof(uint64_t));
	        out.write((char *) &layout_nodes_per_block, sizeof(uint64_t));
	        out.write((char *) &layout_read_page_bytes, sizeof(uint64_t));
	      }
	    }
	  };

  // The index is stored as fixed-size DiskNodes (records) on disk.
  // Each DiskNode contains: [vector (coords) | nnbrs | nnbrs neighbor IDs | labels (maybe 0 length) ].
  // This struct serves as a reference to a DiskNode<T> in the in-memory page-aligned buffer.
  template<typename T>
  struct DiskNode {
    T *coords;
    uint32_t &nnbrs;
    uint32_t *nbrs;
    void *labels;

		    DiskNode<T>(char *page_buf, uint32_t loc, const SSDIndexMetadata<T> &meta)
		        : coords((T *) (page_buf + (meta.uses_packed_layout()
		                                        ? meta.packed_slot_offset(loc % meta.layout_nodes_per_block)
		                                              % meta.layout_read_page_bytes
		                                        : (meta.nnodes_per_sector == 0
		                                               ? 0
		                                               : (loc % meta.nnodes_per_sector) * meta.max_node_len)))),
	          nnbrs(*(uint32_t *) ((char *) coords + meta.data_dim * sizeof(T))),
	          nbrs((uint32_t *) ((char *) coords + meta.data_dim * sizeof(T) + sizeof(uint32_t))),
	          labels((void *) ((char *) coords + meta.data_dim * sizeof(T) + (1 + meta.range) * sizeof(uint32_t))) {
	    }
  };

  template<typename T>
  struct QueryBuffer {
    T *coord_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim], for current vector in comparison.

    char *sector_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN], for sectors.
    uint64_t sector_idx = 0;         // index of next [SECTOR_LEN] scratch to use

    float *nbr_ctx_scratch = nullptr;       // MUST BE AT LEAST [256 * NCHUNKS], for pq table distance.
    float *aligned_dist_scratch = nullptr;  // MUST BE AT LEAST pipeann MAX_DEGREE, for exact dist.
    uint8_t *nbr_vec_scratch = nullptr;     // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE], for neighbor PQ vectors.
    T *aligned_query_T = nullptr;
    char *update_buf = nullptr;  // Dynamic allocate in insert_in_place.

    tsl::robin_set<uint64_t> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;
	    IORequest reqs[MAX_N_SECTOR_READS * MAX_N_NODE_READ_PAGES];

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };
};  // namespace pipeann
