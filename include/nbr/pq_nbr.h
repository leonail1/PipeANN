#pragma once

#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils.h"
#include <immintrin.h>
#include <sstream>
#include <string_view>
#include <shared_mutex>
#include "nbr/pq_table.h"
#include "ssd_index_defs.h"
#include "nbr/abstract_nbr.h"
#include "partition.h"
#include "utils/lock_table.h"
#include "utils/tsl/robin_map.h"
#include "math_utils.h"
#include "utils/cached_io.h"

namespace pipeann {
  template<typename T>
  class PQNeighbor : public AbstractNeighbor<T> {
    static constexpr uint32_t NUM_KMEANS = 15;

   public:
    PQNeighbor() {
    }

    static std::string get_name() {
      return "PQNeighbor";
    }

    // rev_id_map: new_id -> old_id.
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) {
      AbstractNeighbor<T> *abs_nbr_handler;
      PQNeighbor<T> *pq_nbr_handler = new PQNeighbor<T>();
      abs_nbr_handler = pq_nbr_handler;

      pq_nbr_handler->data.resize(new_npoints * this->pq_table.n_chunks);
#pragma omp parallel for num_threads(nthreads)
      for (uint64_t i = 0; i < new_npoints; ++i) {
        memcpy(pq_nbr_handler->data.data() + i * this->pq_table.n_chunks,
               this->data.data() + rev_id_map.find(i) * this->pq_table.n_chunks, this->pq_table.n_chunks);
      }
      pq_nbr_handler->pq_table = std::move(this->pq_table);
      pq_nbr_handler->npoints = new_npoints;
      return abs_nbr_handler;
    }

    // For PQ, out-buffer is filled with PQ table distances.
    void initialize_query(const T *query, QueryBuffer<T> *query_buf) {
      return pq_table.populate_chunk_distances(query, query_buf->nbr_ctx_scratch);
    }
    // call initialize_query first!
    // output to query_buf->aligned_dist_scratch
    void compute_dists(QueryBuffer<T> *query_buf, const uint32_t *ids, const uint64_t n_ids) {
      pq_mu.lock_shared();
      aggregate_coords(ids, n_ids, this->data.data(), pq_table.n_chunks, query_buf->nbr_vec_scratch);
      pq_dist_lookup(query_buf->nbr_vec_scratch, n_ids, pq_table.n_chunks, query_buf->nbr_ctx_scratch,
                     query_buf->aligned_dist_scratch);
      pq_mu.unlock_shared();
    }

    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *aligned_scratch) {
      pq_mu.lock_shared();
      const uint8_t *src_ptr = this->data.data() + (pq_table.n_chunks * query_id);
      // aggregate PQ coords into scratch
      aggregate_coords(ids, n_ids, this->data.data(), pq_table.n_chunks, aligned_scratch);
      // compute distances
      this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, dists_out, n_ids);
      pq_mu.unlock_shared();
    }

    void load(const char *index_prefix) {
      // TODO(gh): possible memory leak for centroid bin during reload.
      std::string iprefix = std::string(index_prefix);
      std::string pq_table_bin = iprefix + "_pq_pivots.bin";
      std::string pq_compressed_vectors = iprefix + "_pq_compressed.bin";

      size_t pq_pivots_offset = 0;
      size_t pq_vectors_offset = 0;

      LOG(INFO) << "PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;

      size_t npts_u64, nchunks_u64;
      size_t rss_before_pq = get_current_rss();
      pipeann::load_bin<uint8_t>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
      size_t rss_after_pq = get_current_rss();
      size_t pq_data_theoretical = (npts_u64 * nchunks_u64) / 1024;

      LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
                << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64;
      LOG(DEBUG) << "[PQNeighbor::load] 加载 PQ 压缩向量 - 理论大小: " << pq_data_theoretical << " KB"
                 << ", 增长前: " << rss_before_pq << " KB"
                 << ", 增长后: " << rss_after_pq << " KB"
                 << ", 实际增长: " << (rss_after_pq - rss_before_pq) << " KB";

      size_t rss_before_table = get_current_rss();
      pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64, pq_pivots_offset);
      size_t rss_after_table = get_current_rss();
      size_t pq_table_theoretical = (
          NUM_PQ_CENTROIDS * nchunks_u64 * sizeof(float) * 4 +  // tables (128 KB)
          NUM_PQ_CENTROIDS * nchunks_u64 * sizeof(float) +      // tables_T (128 KB)
          256 * 256 * nchunks_u64 * sizeof(float) +             // all_to_all_dists (8 MB for 32 chunks)
          nchunks_u64 * 4 * sizeof(float) +                     // centroid
          nchunks_u64 * 4 * sizeof(uint32_t) * 2                // rearrangement + chunk_offsets
      ) / 1024;
      LOG(DEBUG) << "[PQNeighbor::load] 加载 PQ 表 - 理论大小: " << pq_table_theoretical << " KB"
                 << ", 增长前: " << rss_before_table << " KB"
                 << ", 增长后: " << rss_after_table << " KB"
                 << ", 实际增长: " << (rss_after_table - rss_before_table) << " KB";
      this->npoints = npts_u64;
    }

    void save(const char *index_prefix) {
      // write PQ pivots.
      std::string pq_out = std::string(index_prefix) + "_pq_compressed.bin";
      std::string pq_pivot_out = std::string(index_prefix) + "_pq_pivots.bin";
      pipeann::save_bin<uint8_t>(pq_out, this->data.data(), this->npoints, pq_table.n_chunks);
      pq_table.save_pq_pivots(pq_pivot_out.c_str());
    }

    /**
     * @brief 构建PQ（Product Quantization）邻居处理器
     * 
     * 该函数实现了Product Quantization的完整构建流程，包括：
     * 1. 从训练数据中生成PQ聚类中心（pivots）
     * 2. 使用生成的聚类中心对原始数据进行量化压缩
     * 3. 将压缩后的数据保存到磁盘
     * 
     * @param index_prefix_path 索引文件前缀路径，用于生成PQ相关文件名
     * @param data_bin 原始数据的二进制文件路径
     * @param bytes_per_nbr 每个邻居使用的字节数，决定PQ chunks的数量
     * 
     * @note PQ chunks数量受以下限制：
     *       - 不能超过 bytes_per_nbr
     *       - 不能超过 AbstractNeighbor<T>::MAX_BYTES_PER_NBR (128)
     *       - 不能超过数据维度 dim
     * 
     * @details 构建过程：
     *          1. 读取数据元信息（点数和维度）
     *          2. 确定PQ chunks数量
     *          3. 从原始数据中随机采样训练集
     *          4. 使用K-means聚类生成PQ pivots
     *          5. 使用pivots对所有数据进行量化压缩
     *          6. 保存压缩数据到磁盘
     */
    void build(const std::string &index_prefix_path, const std::string &data_bin, uint32_t bytes_per_nbr) {
      // 构造PQ相关文件路径
      std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
      std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";

      size_t points_num, dim;

      // 读取数据文件的元信息（点数和维度）
      pipeann::get_bin_metadata(data_bin, points_num, dim);

      this->npoints = points_num;
      // 确定PQ chunks数量，受多个因素限制
      size_t num_pq_chunks = std::min(bytes_per_nbr, AbstractNeighbor<T>::MAX_BYTES_PER_NBR);
      num_pq_chunks = std::min(num_pq_chunks, dim);  // PQ chunks数量不能超过数据维度
      LOG(INFO) << "File: " << data_bin << " has: " << points_num << " points.";
      LOG(INFO) << "Using " << num_pq_chunks << " PQ chunks.";

      size_t train_size, train_dim;
      float *train_data;  // 训练数据，最大约1GB（256000 * 1024维 * 4字节）

      // 开始计时PQ构建过程
      auto start = std::chrono::high_resolution_clock::now();
      double p_val = this->get_sample_p();
      // 从原始数据中生成随机采样的训练集
      gen_random_slice<T>(data_bin, p_val, train_data, train_size, train_dim);

      // 使用训练数据生成PQ聚类中心（pivots）
      LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size;
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                         pq_pivots_path);
      auto end = std::chrono::high_resolution_clock::now();

      LOG(INFO) << "Pivots generated in " << std::chrono::duration<double>(end - start).count() << "s.";
      start = std::chrono::high_resolution_clock::now();
      // 使用生成的pivots对所有数据进行PQ压缩
      generate_pq_data_from_pivots(data_bin, 256, num_pq_chunks, pq_pivots_path, pq_compressed_vectors_path, 0);
      // 释放训练数据内存
      delete[] train_data;
      train_data = nullptr;
      end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Compressed data written in: " << std::chrono::duration<double>(end - start).count() << "s.";
    }

    void insert(T *point, uint32_t loc) {
      std::vector<uint8_t> pq_coords(pq_table.n_chunks);
      std::vector<float> fp_vec(pq_table.ndims);
      for (uint32_t i = 0; i < pq_table.ndims; i++) {
        fp_vec[i] = (float) point[i];
      }
      pq_table.deflate_vec(fp_vec.data(), pq_coords.data());

      uint64_t pq_offset = loc * pq_table.n_chunks;
      {
        pq_mu.lock();
        if (this->data.size() < pq_offset + pq_table.n_chunks) {
          this->data.resize(1.5 * (pq_offset + pq_table.n_chunks));
        }
        memcpy(this->data.data() + pq_offset, pq_coords.data(), pq_table.n_chunks);
        this->npoints = std::max(this->npoints, (uint64_t) (loc + 1));
        pq_mu.unlock();
      }
    }

   private:
    // PQ data
    // pq_table.n_chunks = # of chunks ndims is split into
    // data: uint8_t * pq_table.n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * pq_table.n_chunks]
    v2::ReaderOptSharedMutex pq_mu;
    std::vector<uint8_t> data;
    FixedChunkPQTable<T> pq_table;

    inline void aggregate_coords(const unsigned *ids, const uint64_t n_ids, const uint8_t *all_coords,
                                 const uint64_t ndims, uint8_t *out) {
      for (uint64_t i = 0; i < n_ids; i++) {
        memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(uint8_t));
      }
    }

    inline void prefetch_chunk_dists(const float *ptr) {
      _mm_prefetch((char *) ptr, _MM_HINT_NTA);
      _mm_prefetch((char *) (ptr + 64), _MM_HINT_NTA);
      _mm_prefetch((char *) (ptr + 128), _MM_HINT_NTA);
      _mm_prefetch((char *) (ptr + 192), _MM_HINT_NTA);
    }

    inline void pq_dist_lookup(const uint8_t *pq_ids, const uint64_t n_pts, const uint64_t pq_nchunks,
                               const float *pq_dists, float *dists_out) {
      _mm_prefetch((char *) dists_out, _MM_HINT_T0);
      _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);

      prefetch_chunk_dists(pq_dists);
      memset(dists_out, 0, n_pts * sizeof(float));
      for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
        const float *chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
          prefetch_chunk_dists(chunk_dists + 256);
        }
        for (uint64_t idx = 0; idx < n_pts; idx++) {
          uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
          dists_out[idx] += chunk_dists[pq_centerid];
        }
      }
    }

    // given training data in train_data of dimensions num_train * dim, generate PQ
    // pivots using k-means algorithm to partition the co-ordinates into
    // num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
    // k-means in each chunk to compute the PQ pivots and stores in bin format in
    // file pq_pivots_path as a s num_centers*dim floating point binary file
    int generate_pq_pivots(const std::unique_ptr<T[]> &passed_train_data, size_t num_train, unsigned dim,
                           unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                           std::string pq_pivots_path) {
      std::unique_ptr<float[]> train_float = std::make_unique<float[]>(num_train * (size_t) (dim));
      float *flt_ptr = train_float.get();
      T *T_ptr = passed_train_data.get();

      for (uint64_t i = 0; i < num_train; i++) {
        for (uint64_t j = 0; j < (uint64_t) dim; j++) {
          flt_ptr[i * (uint64_t) dim + j] = (float) T_ptr[i * (uint64_t) dim + j];
        }
      }
      if (generate_pq_pivots(flt_ptr, num_train, dim, num_centers, num_pq_chunks, max_k_means_reps, pq_pivots_path) !=
          0)
        return -1;
      return 0;
    }

    int generate_pq_pivots(const float *passed_train_data, size_t num_train, unsigned dim, unsigned num_centers,
                           unsigned num_pq_chunks, unsigned max_k_means_reps, std::string pq_pivots_path) {
      if (num_pq_chunks > dim) {
        LOG(ERROR) << " Error: number of chunks more than dimension";
        return -1;
      }

      for (uint64_t i = 0; i < num_train; i++) {
        for (uint64_t j = 0; j < dim; j++) {
          if (std::isnan(passed_train_data[i * dim + j])) {
            LOG(ERROR) << "Error: NaN value found in training data at index [" << i << "][" << j << "]";
          }
        }
      }

      std::unique_ptr<float[]> train_data = std::make_unique<float[]>(num_train * dim);
      std::memcpy(train_data.get(), passed_train_data, num_train * dim * sizeof(float));

      for (uint64_t i = 0; i < num_train; i++) {
        for (uint64_t j = 0; j < dim; j++) {
          if (passed_train_data[i * dim + j] != train_data[i * dim + j]) {
            LOG(ERROR) << "error in copy: passed_train_data[" << i << "][" << j
                       << "] = " << passed_train_data[i * dim + j] << ", train_data[" << i << "][" << j
                       << "] = " << train_data[i * dim + j];
            int val1 = 0, val2 = 0;
            memcpy(&val1, &passed_train_data[i * dim + j], sizeof(int));
            memcpy(&val2, &train_data[i * dim + j], sizeof(int));
            LOG(ERROR) << "Hex: " << "passed_train_data[" << i << "][" << j << "] = " << val1 << ", train_data[" << i
                       << "][" << j << "] = " << val2;
            exit(-1);
          }
        }
      }

      std::unique_ptr<float[]> full_pivot_data;

      // Calculate centroid and center the training data
      std::unique_ptr<float[]> centroid = std::make_unique<float[]>(dim);
      for (uint64_t d = 0; d < dim; d++) {
        centroid[d] = 0;
        for (uint64_t p = 0; p < num_train; p++) {
          centroid[d] += train_data[p * dim + d];
        }
        centroid[d] /= (float) num_train;
      }

      //  std::memset(centroid, 0 , dim*sizeof(float));

      for (uint64_t d = 0; d < dim; d++) {
        for (uint64_t p = 0; p < num_train; p++) {
          train_data[p * dim + d] -= centroid[d];
        }
      }

      std::vector<uint32_t> rearrangement;
      std::vector<uint32_t> chunk_offsets;

      size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
      size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
      size_t max_num_high = dim - (low_val * num_pq_chunks);
      size_t cur_num_high = 0;
      size_t cur_bin_threshold = high_val;

      std::vector<std::vector<uint32_t>> bin_to_dims(num_pq_chunks);
      tsl::robin_map<uint32_t, uint32_t> dim_to_bin;
      std::vector<float> bin_loads(num_pq_chunks, 0);

      // Process dimensions not inserted by previous loop
      for (uint32_t d = 0; d < dim; d++) {
        if (dim_to_bin.find(d) != dim_to_bin.end())
          continue;
        auto cur_best = num_pq_chunks + 1;
        float cur_best_load = std::numeric_limits<float>::max();
        for (uint32_t b = 0; b < num_pq_chunks; b++) {
          if (bin_loads[b] < cur_best_load && bin_to_dims[b].size() < cur_bin_threshold) {
            cur_best = b;
            cur_best_load = bin_loads[b];
          }
        }
        bin_to_dims[cur_best].push_back(d);
        if (bin_to_dims[cur_best].size() == high_val) {
          cur_num_high++;
          if (cur_num_high == max_num_high)
            cur_bin_threshold = low_val;
        }
      }

      rearrangement.clear();
      chunk_offsets.clear();
      chunk_offsets.push_back(0);

      for (uint32_t b = 0; b < num_pq_chunks; b++) {
        for (auto p : bin_to_dims[b]) {
          rearrangement.push_back(p);
        }
        if (b > 0)
          chunk_offsets.push_back(chunk_offsets[b - 1] + (unsigned) bin_to_dims[b - 1].size());
      }
      chunk_offsets.push_back(dim);

      full_pivot_data.reset(new float[num_centers * dim]);

      // DEBUG ONLY
      double kmeans_time = 0.0, lloyds_time = 0.0, copy_time = 0.0;

      for (size_t i = 0; i < num_pq_chunks; i++) {
        size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];

        if (cur_chunk_size == 0)
          continue;
        std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
        std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_train * cur_chunk_size);
        std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_train);

        memset((void *) cur_pivot_data.get(), 0, num_centers * cur_chunk_size * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, 65536)
        for (int64_t j = 0; j < (int64_t) num_train; j++) {
          std::memcpy(cur_data.get() + j * cur_chunk_size, train_data.get() + j * dim + chunk_offsets[i],
                      cur_chunk_size * sizeof(float));
        }
        auto end = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        // kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train,
        // cur_chunk_size,
        //                                  cur_pivot_data.get(), num_centers);
        kmeans::selecting_pivots(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers);

        unsigned k_means_reps = max_k_means_reps;

        kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers, k_means_reps,
                           nullptr, closest_center.get());
        end = std::chrono::high_resolution_clock::now();
        kmeans_time += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        if (num_train > 2 * num_centers) {
          kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers,
                             max_k_means_reps, NULL, closest_center.get());
        }
        end = std::chrono::high_resolution_clock::now();
        lloyds_time += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (uint64_t j = 0; j < num_centers; j++) {
          std::memcpy(full_pivot_data.get() + j * dim + chunk_offsets[i], cur_pivot_data.get() + j * cur_chunk_size,
                      cur_chunk_size * sizeof(float));
        }
        end = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration<double>(end - start).count();
      }
      LOG(INFO) << "Kmeans time: " << kmeans_time << " Lloyds time: " << lloyds_time << " Copy time: " << copy_time;

      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data.get(),
                                                                 (size_t) num_centers, dim, cumul_bytes[0]);
      cumul_bytes[2] = cumul_bytes[1] + pipeann::save_bin<float>(pq_pivots_path.c_str(), centroid.get(), (size_t) dim,
                                                                 1, cumul_bytes[1]);
      cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement.data(),
                                                                    rearrangement.size(), 1, cumul_bytes[2]);
      cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets.data(),
                                                                    chunk_offsets.size(), 1, cumul_bytes[3]);
      pipeann::save_bin<uint64_t>(pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved pq pivot data to " << pq_pivots_path << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                << "B.";

      return 0;
    }

    // streams the base file (data_file), and computes the closest centers in each
    // chunk to generate the compressed data_file and stores it in
    // pq_compressed_vectors_path.
    // If the numbber of centers is < 256, it stores as byte vector, else as 4-byte
    // vector in binary format.
    int generate_pq_data_from_pivots(const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
                                     std::string pq_pivots_path, std::string pq_compressed_vectors_path,
                                     size_t offset) {
      uint64_t read_blk_size = 64 * 1024 * 1024;
      cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
      uint32_t npts32;
      uint32_t basedim32;
      base_reader.read((char *) &npts32, sizeof(uint32_t));
      base_reader.read((char *) &basedim32, sizeof(uint32_t));
      size_t num_points = npts32;
      size_t dim = basedim32;

      size_t BLOCK_SIZE = (std::min)((size_t) 16384, num_points);  // hard-coded max block size.

      std::unique_ptr<float[]> full_pivot_data;
      std::unique_ptr<float[]> centroid;
      std::unique_ptr<uint32_t[]> rearrangement;
      std::unique_ptr<uint32_t[]> chunk_offsets;

      if (!file_exists(pq_pivots_path)) {
        LOG(INFO) << "ERROR: PQ k-means pivot file not found";
        crash();
      } else {
        uint64_t nr, nc;
        std::unique_ptr<uint64_t[]> file_offset_data;

        pipeann::load_bin<uint64_t>(pq_pivots_path.c_str(), file_offset_data, nr, nc, 0);

        if (nr != 5) {
          LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path
                    << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting 5.";
          crash();
        }

        pipeann::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc, file_offset_data[0]);

        if ((nr != num_centers) || (nc != dim)) {
          LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_num_centers  = " << nr
                    << ", file_dim = " << nc << " but expecting " << num_centers << " centers in " << dim
                    << " dimensions.";
          crash();
        }

        pipeann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc, file_offset_data[1]);

        if ((nr != dim) || (nc != 1)) {
          LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr
                    << ", file_cols = " << nc << " but expecting " << dim << " entries in 1 dimension.";
          crash();
        }

        pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc, file_offset_data[2]);

        if ((nr != dim) || (nc != 1)) {
          LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr
                    << ", file_cols = " << nc << " but expecting " << dim << " entries in 1 dimension.";
          crash();
        }

        pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc, file_offset_data[3]);

        if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
          LOG(INFO) << "Error reading pq_pivots file at chunk offsets; file has nr=" << nr << ",nc=" << nc
                    << ", expecting nr=" << num_pq_chunks + 1 << ", nc=1.";
          crash();
        }

        LOG(INFO) << "Loaded PQ pivot information";
      }

      std::ofstream compressed_file_writer(pq_compressed_vectors_path, std::ios::binary);
      uint32_t num_pq_chunks_u32 = num_pq_chunks;

      compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
      compressed_file_writer.write((char *) &num_pq_chunks_u32, sizeof(uint32_t));

      size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
      std::unique_ptr<uint32_t[]> block_compressed_base =
          std::make_unique<uint32_t[]>(block_size * (uint64_t) num_pq_chunks);
      std::memset(block_compressed_base.get(), 0, block_size * (uint64_t) num_pq_chunks * sizeof(uint32_t));

      std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
      std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
      std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

      size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

      for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        pipeann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

        for (uint64_t p = 0; p < cur_blk_size; p++) {
          for (uint64_t d = 0; d < dim; d++) {
            block_data_tmp[p * dim + d] -= centroid[d];
          }
        }

        for (uint64_t p = 0; p < cur_blk_size; p++) {
          for (uint64_t d = 0; d < dim; d++) {
            block_data_float[p * dim + d] = block_data_tmp[p * dim + rearrangement[d]];
          }
        }

        for (size_t i = 0; i < num_pq_chunks; i++) {
          size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
          if (cur_chunk_size == 0)
            continue;

          std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
          std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
          std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(cur_blk_size);

#pragma omp parallel for schedule(static, 8192)
          for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
            for (uint64_t k = 0; k < cur_chunk_size; k++)
              cur_data[j * cur_chunk_size + k] = block_data_float[j * dim + chunk_offsets[i] + k];
          }

#pragma omp parallel for schedule(static, 1)
          for (int64_t j = 0; j < (int64_t) num_centers; j++) {
            std::memcpy(cur_pivot_data.get() + j * cur_chunk_size, full_pivot_data.get() + j * dim + chunk_offsets[i],
                        cur_chunk_size * sizeof(float));
          }

          math_utils::compute_closest_centers(cur_data.get(), cur_blk_size, cur_chunk_size, cur_pivot_data.get(),
                                              num_centers, 1, closest_center.get());
#pragma omp parallel for schedule(static, 8192)
          for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
            block_compressed_base[j * num_pq_chunks + i] = closest_center[j];
          }
        }

        if (num_centers > 256) {
          compressed_file_writer.write((char *) (block_compressed_base.get()),
                                       cur_blk_size * num_pq_chunks * sizeof(uint32_t));
        } else {
          std::unique_ptr<uint8_t[]> pVec = std::make_unique<uint8_t[]>(cur_blk_size * num_pq_chunks);
          pipeann::convert_types<uint32_t, uint8_t>(block_compressed_base.get(), pVec.get(), cur_blk_size,
                                                    num_pq_chunks);
          compressed_file_writer.write((char *) (pVec.get()), cur_blk_size * num_pq_chunks * sizeof(uint8_t));
        }
        // LOG(INFO) << ".done.";
      }
      // Splittng diskann_dll into separate DLLs for search and build.
      // This code should only be available in the "build" DLL.
      compressed_file_writer.close();
      return 0;
    }
  };
}  // namespace pipeann