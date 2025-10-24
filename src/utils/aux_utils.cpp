#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <cblas.h>

#include "aux_utils.h"
#include "utils/cached_io.h"
#include "index.h"
#include "nbr/pq_nbr.h"
#include "omp.h"
#include "partition.h"
#include "utils/percentile_stats.h"
#include "ssd_index.h"
#include "utils.h"

#include "ssd_index.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double total_recall = 0;
    std::set<unsigned> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      unsigned active_points_count = 0;
      unsigned cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
        LOG(INFO) << "Warning: Couldn't find enough closest neighbors " << active_points_count << "/" << recall_at
                  << " from truthset for query # " << i << ". Will result in under-reported value of recall.";
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      LOG(ERROR) << "Error reading idmap file. Check if the file is bin file with 1 dimensional data. Actual: "
                 << actual_file_size << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t);

      crash();
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const uint64_t nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++) {
      vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    uint64_t nnodes = 0;
    uint64_t nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (uint64_t) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (uint64_t shard = 0; shard < nshards; shard++) {
      LOG(INFO) << "Creating inverse map -- shard #" << shard;
      for (uint64_t idx = 0; idx < idmaps[shard].size(); idx++) {
        uint64_t node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((uint32_t) node_id, (uint32_t) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
      return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    LOG(INFO) << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (uint64_t i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    LOG(INFO) << "Max input width: " << max_input_width << ", output width: " << output_width;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = (uint32_t) nshards;
    uint32_t one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    uint64_t vamana_index_frozen = 0;
    for (uint64_t shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(uint64_t));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    diskann_writer.write((char *) &merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    LOG(INFO) << "Starting merge";

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG(INFO) << cur_id << "...";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(), shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (uint64_t j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    LOG(INFO) << "Expected size: " << merged_index_size;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    LOG(INFO) << "Finished merge";
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, unsigned L, unsigned R,
                                double sampling_rate, double ram_budget, std::string mem_index_path,
                                std::string medoids_file, std::string centroids_file, const char *tag_file) {
    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::Parameters paras;
      paras.set(R, L, 750, 1.2, 0, true);

      bool tags_enabled;
      if (tag_file == nullptr)
        tags_enabled = false;
      else
        tags_enabled = true;

      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, base_dim, base_num, false, false, tags_enabled));
      if (tags_enabled)
        _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file);
      else
        _pvamanaIndex->build(base_file.c_str(), base_num, paras);

      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }

    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      pipeann::Parameters paras;
      paras.set(2 * R / 3, L, 750, 1.2, 0, false);
      uint64_t shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, shard_base_dim, shard_base_pts, false, false));
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    pipeann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    uint64_t read_blk_size = 64 * 1024 * 1024;
    uint64_t write_blk_size = read_blk_size;
    cached_ifstream base_reader;
    std::ifstream vamana_reader;
    uint64_t base_offset = 0, vamana_offset = 0, tags_offset = 0;
    bool tags_enabled = false;

    base_reader.open(base_file, read_blk_size);
    vamana_reader.open(mem_index_file, std::ios::binary);
    tags_enabled = tag_file != "";

    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    //    size_t          actual_file_size = get_file_size(mem_index_file);
    std::remove(output_file.c_str());
    cached_ofstream diskann_writer;
    diskann_writer.open(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));

    uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;
    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(uint64_t));
    // compute
    uint64_t medoid, max_node_len, nnodes_per_sector;
    npts_64 = (uint64_t) npts;
    medoid = (uint64_t) medoid_u32;
    if (vamana_frozen_num == 1)
      vamana_frozen_loc = medoid;
    max_node_len = (((uint64_t) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;  // 0 if max_node_len > SECTOR_LEN

    LOG(INFO) << "medoid: " << medoid << "B";
    LOG(INFO) << "max_node_len: " << max_node_len << "B";
    LOG(INFO) << "nnodes_per_sector: " << nnodes_per_sector << "B";

    // SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
    unsigned *nhood_buf = (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(unsigned));

    // number of sectors (1 for meta data)
    uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                               : npts_64 * DIV_ROUND_UP(max_node_len, SECTOR_LEN);
    uint64_t disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;

    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), SECTOR_LEN);  // write out the empty
                                                         // first sector, will
                                                         // be populated at the
                                                         // end.

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    LOG(INFO) << "# sectors: " << n_sectors;
    uint64_t cur_node_id = 0;

    if (nnodes_per_sector > 0) {
      for (uint64_t sector = 0; sector < n_sectors; sector++) {
        if (sector % 100000 == 0) {
          LOG(INFO) << "Sector #" << sector << "written";
        }
        memset(sector_buf.get(), 0, SECTOR_LEN);
        for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
             sector_node_id++) {
          memset(node_buf.get(), 0, max_node_len);
          // read cur node's nnbrs
          vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

          // sanity checks on nnbrs
          if (nnbrs == 0) {
            LOG(INFO) << "ERROR. Found point with no out-neighbors; Point#: " << cur_node_id;
            exit(-1);
          }

          // read node's nhood
          vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));
          if (nnbrs > width_u32) {
            vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned), vamana_reader.cur);
          }

          // write coords of node first
          //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
          base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
          memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

          // write nnbrs
          *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

          // write nhood next
          memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned), nhood_buf,
                 (std::min)(nnbrs, width_u32) * sizeof(unsigned));

          // get offset into sector_buf
          char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

          // copy node buf into sector_node_buf
          memcpy(sector_node_buf, node_buf.get(), max_node_len);
          cur_node_id++;
        }
        // flush sector to disk
        diskann_writer.write(sector_buf.get(), SECTOR_LEN);
      }
    } else {
      uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, SECTOR_LEN);
      for (uint64_t i = 0; i < npts_64; i++) {
        if ((i * nsectors_per_node) % 100000 == 0) {
          LOG(INFO) << "Sector #" << i * nsectors_per_node << "written";
        }
        memset(multisector_buf.get(), 0, nsectors_per_node * SECTOR_LEN);

        memset(node_buf.get(), 0, max_node_len);
        // read cur node's nnbrs
        vamana_reader.read((char *) &nnbrs, sizeof(uint32_t));

        // read node's nhood
        vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
        if (nnbrs > width_u32) {
          vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
        }

        // write coords of node first
        //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(uint32_t *) (multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

        // write nhood next
        memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
               (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

        // flush sector to disk
        diskann_writer.write(multisector_buf.get(), nsectors_per_node * SECTOR_LEN);
      }
    }

    diskann_writer.close();
    size_t tag_bytes_written = 0;

    // frozen point implies dynamic index which must have tags
    if (vamana_frozen_num > 0) {
      std::unique_ptr<TagT[]> mem_index_tags;
      size_t nr, nc;
      pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

      if (nr != npts_64 && nc != 1) {
        LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                   << " tags in 1 dimension (bin format).";

        crash();
      }

      pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
    } else {
      if (tags_enabled) {
        std::unique_ptr<TagT[]> mem_index_tags;
        size_t nr, nc;

        if (!file_exists(tag_file)) {
          LOG(INFO) << "Static vamana index, tag file " << tag_file << "does not exist. Exiting....";
          exit(-1);
        }

        pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

        if (nr != npts_64 && nc != 1) {
          LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                     << " tags in 1 dimension (bin format).";
          crash();
        }

        pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
      }
    }

    output_file_meta.push_back(output_file_meta[output_file_meta.size() - 1] + tag_bytes_written);
    pipeann::save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    LOG(INFO) << "Output file written.";
  }

  /**
   * @brief 构建磁盘索引的主函数
   * 
   * 该函数负责构建一个完整的磁盘索引，包括以下步骤：
   * 1. 数据预处理（如余弦相似度归一化）
   * 2. 构建内存索引
   * 3. 将内存索引转换为磁盘布局
   * 4. 清理临时文件
   * 
   * @tparam T 数据类型（如 float, int8_t, uint8_t）
   * @tparam TagT 标签数据类型（通常为 uint32_t）
   * 
   * @param dataPath 输入数据文件的路径
   * @param indexFilePath 输出索引文件的前缀路径
   * @param R 图的最大出度（每个节点的最大邻居数）
   * @param L 构建时的搜索列表大小
   * @param M 构建索引的RAM预算（GB）
   * @param num_threads 使用的线程数（0表示使用默认值）
   * @param bytes_per_nbr 每个邻居的字节数
   * @param _compareMetric 距离度量类型（L2, COSINE, INNER_PRODUCT等）
   * @param tag_file 标签文件路径（可为nullptr）
   * @param nbr_handler 邻居处理器的抽象接口
   * 
   * @return bool 构建是否成功
   */
  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler) {
    // 将输入参数转换为字符串格式
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    
    // 定义输出文件路径
    std::string mem_index_path = index_prefix_path + "_mem.index";        // 内存索引文件路径
    std::string disk_index_path = index_prefix_path + "_disk.index";      // 磁盘索引文件路径
    std::string medoids_path = disk_index_path + "_medoids.bin";          // 质心文件路径
    std::string centroids_path = disk_index_path + "_centroids.bin";      // 中心点文件路径

    // 设置OpenMP线程数
    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    // 记录构建参数信息
    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Build RAM budget: " << M << "GB T: " << num_threads
              << " bytes per neighbor: " << bytes_per_nbr << " Final index will be in multiple files";

    // 处理余弦相似度度量：需要归一化数据
    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        // 创建归一化数据文件
        normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = pipeann::Metric::L2;  // 归一化后使用L2距离
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    // 开始计时整个构建过程
    auto s = std::chrono::high_resolution_clock::now();
    
    // 构建邻居处理器
    nbr_handler->build(index_prefix_path, normalized_file_path, bytes_per_nbr);

    // 构建Vamana索引
    auto start = std::chrono::high_resolution_clock::now();
    auto p_val = nbr_handler->get_sample_p();  // 获取采样率
    pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, L, R, p_val, M, mem_index_path,
                                          medoids_path, centroids_path, tag_file);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    // 创建磁盘布局
    if (tag_file == nullptr) {
      // 无标签文件的情况
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", disk_index_path);
    } else {
      // 有标签文件的情况
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, disk_index_path);
    }

    // 清理临时文件
    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    
    // 如果创建了归一化文件，则删除它
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    // 记录总构建时间
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;  // 构建成功
  }

  template void create_disk_layout<int8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                     const std::string &tag_file, const std::string &output_file);
  template void create_disk_layout<uint8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                      const std::string &tag_file, const std::string &output_file);
  template void create_disk_layout<float, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                    const std::string &tag_file, const std::string &output_file);

  template bool build_disk_index<int8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                   uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                   pipeann::Metric _compareMetric, const char *tag_file,
                                                   AbstractNeighbor<int8_t> *nbr_handler = nullptr);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                    uint32_t L, uint32_t M, uint32_t num_threads,
                                                    uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                                                    const char *tag_file,
                                                    AbstractNeighbor<uint8_t> *nbr_handler = nullptr);
  template bool build_disk_index<float, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                  uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                  pipeann::Metric _compareMetric, const char *tag_file,
                                                  AbstractNeighbor<float> *nbr_handler = nullptr);

  template int build_merged_vamana_index<int8_t>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                 unsigned R, double sampling_rate, double ram_budget,
                                                 std::string mem_index_path, std::string medoids_path,
                                                 std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<float>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                unsigned R, double sampling_rate, double ram_budget,
                                                std::string mem_index_path, std::string medoids_path,
                                                std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<uint8_t>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                  unsigned R, double sampling_rate, double ram_budget,
                                                  std::string mem_index_path, std::string medoids_path,
                                                  std::string centroids_file, const char *tag_file);
};  // namespace pipeann
