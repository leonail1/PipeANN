#pragma once

#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils.h"
#include <immintrin.h>
#include "nbr/pq_table.h"
#include "ssd_index_defs.h"
#include "nbr/abstract_nbr.h"
#include "utils/partition.h"
#include "utils/lock_table.h"
#include "utils/tsl/robin_map.h"
#include "utils/kmeans_utils.h"
#include "utils/cached_io.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>

namespace pipeann {
  template<typename T>
  class PQNeighbor : public AbstractNeighbor<T> {
    static constexpr uint32_t NUM_KMEANS = 15;

   public:
    PQNeighbor(pipeann::Metric metric) : AbstractNeighbor<T>(metric), pq_table(metric) {
    }

    ~PQNeighbor() override {
      release_mmap();
    }

    // max size of context needed for a single query.
    uint64_t query_ctx_size() {
      return 256 * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(float);
    }

    std::string get_name() {
      return "PQNeighbor";
    }

    // rev_id_map: new_id -> old_id.
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) {
      AbstractNeighbor<T> *abs_nbr_handler;
      PQNeighbor<T> *pq_nbr_handler = new PQNeighbor<T>(this->metric);
      abs_nbr_handler = pq_nbr_handler;

      pq_nbr_handler->data.resize(new_npoints * this->pq_table.n_chunks);
      std::vector<uint8_t> src_materialized;
      const uint8_t *src_data = this->codes();
      if (src_data == nullptr) {
        src_materialized.resize(this->npoints * this->pq_table.n_chunks);
        read_codes(0, src_materialized.size(), src_materialized.data());
        src_data = src_materialized.data();
      }
#pragma omp parallel for num_threads(nthreads)
      for (uint64_t i = 0; i < new_npoints; ++i) {
        memcpy(pq_nbr_handler->data.data() + i * this->pq_table.n_chunks,
               src_data + rev_id_map.find(i) * this->pq_table.n_chunks, this->pq_table.n_chunks);
      }
      pq_nbr_handler->pq_table = std::move(this->pq_table);
      pq_nbr_handler->npoints = new_npoints;
      return abs_nbr_handler;
    }

    // For PQ, out-buffer is filled with PQ table distances.
    void initialize_query(const T *query, QueryBuffer *query_buf) {
      return pq_table.populate_chunk_distances(query, query_buf->nbr_ctx_scratch);
    }

    // call initialize_query first!
    // output to query_buf->aligned_dist_scratch
    void compute_dists(QueryBuffer *query_buf, const uint32_t *ids, const uint64_t n_ids) {
      pq_mu.lock_shared();
      aggregate_coords_for_ids(ids, n_ids, query_buf->nbr_vec_scratch);
      pq_dist_lookup(query_buf->nbr_vec_scratch, n_ids, pq_table.n_chunks, query_buf->nbr_ctx_scratch,
                     query_buf->aligned_dist_scratch);
      release_mmap_resident_pages();
      pq_mu.unlock_shared();
    }

    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *aligned_scratch) {
      pq_mu.lock_shared();
      std::vector<uint8_t> src_code;
      const uint8_t *src_ptr = nullptr;
      if (this->codes() != nullptr) {
        src_ptr = this->codes() + (pq_table.n_chunks * query_id);
      } else {
        src_code.resize(pq_table.n_chunks);
        read_code(query_id, src_code.data());
        src_ptr = src_code.data();
      }
      // aggregate PQ coords into scratch
      aggregate_coords_for_ids(ids, n_ids, aligned_scratch);
      // compute distances
      this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, dists_out, n_ids);
      release_mmap_resident_pages();
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

      size_t npts_u64 = 0, nchunks_u64 = 0;
      release_mmap();
      if (pq_stream_enabled()) {
        load_codes_stream(pq_compressed_vectors, npts_u64, nchunks_u64);
      } else if (pq_mmap_enabled()) {
        load_codes_mmap(pq_compressed_vectors, npts_u64, nchunks_u64);
      } else {
        pipeann::load_bin<uint8_t>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
      }

      LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
                << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64
                << " mmap: " << (mmap_codes_ != nullptr) << " stream: " << (stream_fd_ >= 0);

      pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64, pq_pivots_offset);
      this->npoints = npts_u64;
    }

    void save(const char *index_prefix) {
      // write PQ pivots.
      std::string pq_out = std::string(index_prefix) + "_pq_compressed.bin";
      std::string pq_pivot_out = std::string(index_prefix) + "_pq_pivots.bin";
      pq_mu.lock();
      materialize_codes_unlocked();
      pipeann::save_bin<uint8_t>(pq_out, data.data(), this->npoints, pq_table.n_chunks);
      pq_mu.unlock();
      pq_table.save_pq_pivots(pq_pivot_out.c_str());
    }

    // During build, we use L2 for both L2 and MIPS.
    void build(const std::string &index_prefix_path, const std::string &data_bin, uint32_t bytes_per_nbr) {
      std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
      std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";

      size_t points_num, dim;

      pipeann::get_bin_metadata(data_bin, points_num, dim);

      this->npoints = points_num;
      size_t num_pq_chunks = std::min(bytes_per_nbr, AbstractNeighbor<T>::MAX_BYTES_PER_NBR);
      num_pq_chunks = std::min(num_pq_chunks, dim);  // cannot have more chunks than dim.
      LOG(INFO) << "File: " << data_bin << " has: " << points_num << " points.";
      LOG(INFO) << "Using " << num_pq_chunks << " PQ chunks.";

      size_t train_size, train_dim;
      float *train_data;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.

      auto start = std::chrono::high_resolution_clock::now();
      double p_val = this->get_sample_p();
      // generates random sample and sets it to train_data and updates train_size
      gen_random_slice<T>(data_bin, p_val, train_data, train_size, train_dim);

      LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size;
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                         pq_pivots_path);
      auto end = std::chrono::high_resolution_clock::now();

      LOG(INFO) << "Pivots generated in " << std::chrono::duration<double>(end - start).count() << "s.";
      start = std::chrono::high_resolution_clock::now();
      generate_pq_data_from_pivots(data_bin, 256, num_pq_chunks, pq_pivots_path, pq_compressed_vectors_path, 0);
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
        materialize_codes_unlocked();
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
    pipeann::ReaderOptSharedMutex pq_mu;
    std::vector<uint8_t> data;
    int mmap_fd_ = -1;
    int stream_fd_ = -1;
    uint8_t *mmap_base_ = nullptr;
    const uint8_t *mmap_codes_ = nullptr;
    size_t mmap_bytes_ = 0;
    uint64_t stream_generation_ = 0;
    FixedChunkPQTable<T> pq_table;

    bool pq_mmap_enabled() const {
      const char *env = std::getenv("PIPEANN_PQ_MMAP_LOAD");
      return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
             std::strcmp(env, "FALSE") != 0;
    }

    bool pq_stream_enabled() const {
      const char *env = std::getenv("PIPEANN_PQ_STREAM_LOAD");
      return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
             std::strcmp(env, "FALSE") != 0;
    }

    const uint8_t *codes() const {
      if (!data.empty()) {
        return data.data();
      }
      return mmap_codes_;
    }

    void release_mmap() {
      if (mmap_base_ != nullptr) {
        munmap(mmap_base_, mmap_bytes_);
        mmap_base_ = nullptr;
        mmap_codes_ = nullptr;
        mmap_bytes_ = 0;
      }
      if (mmap_fd_ >= 0) {
        close(mmap_fd_);
        mmap_fd_ = -1;
      }
      if (stream_fd_ >= 0) {
        close(stream_fd_);
        stream_fd_ = -1;
        stream_generation_++;
      }
    }

    void release_mmap_resident_pages() const {
      if (mmap_base_ != nullptr) {
        const uint64_t interval = mmap_dontneed_interval();
        if (interval == 0) {
          return;
        }
        static thread_local uint64_t calls = 0;
        calls++;
        if (calls % interval == 0) {
          madvise(mmap_base_, mmap_bytes_, MADV_DONTNEED);
        }
      }
    }

    uint64_t mmap_dontneed_interval() const {
      const char *env = std::getenv("PIPEANN_PQ_MMAP_DONTNEED_EVERY");
      if (env == nullptr || *env == '\0') {
        return 0;
      }
      char *end = nullptr;
      uint64_t value = std::strtoull(env, &end, 10);
      return (end == env || *end != '\0') ? 0 : value;
    }

    void load_codes_mmap(const std::string &path, size_t &npts_u64, size_t &nchunks_u64) {
      mmap_fd_ = open(path.c_str(), O_RDONLY);
      if (mmap_fd_ < 0) {
        LOG(ERROR) << "Failed to open PQ compressed file for mmap: " << path;
        crash();
      }
      struct stat st {};
      if (fstat(mmap_fd_, &st) != 0 || st.st_size < 8) {
        LOG(ERROR) << "Invalid PQ compressed file for mmap: " << path;
        crash();
      }
      mmap_bytes_ = static_cast<size_t>(st.st_size);
      void *mapped = mmap(nullptr, mmap_bytes_, PROT_READ, MAP_PRIVATE, mmap_fd_, 0);
      if (mapped == MAP_FAILED) {
        LOG(ERROR) << "Failed to mmap PQ compressed file: " << path;
        crash();
      }
      madvise(mapped, mmap_bytes_, MADV_RANDOM);
      mmap_base_ = static_cast<uint8_t *>(mapped);
      uint32_t npts = 0, nchunks = 0;
      memcpy(&npts, mmap_base_, sizeof(uint32_t));
      memcpy(&nchunks, mmap_base_ + sizeof(uint32_t), sizeof(uint32_t));
      npts_u64 = npts;
      nchunks_u64 = nchunks;
      const size_t expected_bytes = 2 * sizeof(uint32_t) + npts_u64 * nchunks_u64;
      if (expected_bytes > mmap_bytes_) {
        LOG(ERROR) << "PQ compressed mmap size mismatch: expected " << expected_bytes << " got " << mmap_bytes_;
        crash();
      }
      mmap_codes_ = mmap_base_ + 2 * sizeof(uint32_t);
      data.clear();
      data.shrink_to_fit();
    }

    void load_codes_stream(const std::string &path, size_t &npts_u64, size_t &nchunks_u64) {
      stream_fd_ = open(path.c_str(), O_RDONLY);
      if (stream_fd_ < 0) {
        LOG(ERROR) << "Failed to open PQ compressed file for streaming: " << path;
        crash();
      }
      stream_generation_++;
      uint32_t header[2] = {0, 0};
      if (pread(stream_fd_, header, sizeof(header), 0) != (ssize_t) sizeof(header)) {
        LOG(ERROR) << "Failed to read PQ compressed header: " << path;
        crash();
      }
      npts_u64 = header[0];
      nchunks_u64 = header[1];
      data.clear();
      data.shrink_to_fit();
    }

    void materialize_codes_unlocked() {
      if (!data.empty()) {
        return;
      }
      data.resize(this->npoints * pq_table.n_chunks);
      if (mmap_codes_ != nullptr) {
        memcpy(data.data(), mmap_codes_, data.size());
      } else if (stream_fd_ >= 0) {
        read_codes(0, data.size(), data.data());
      } else {
        LOG(ERROR) << "No PQ codes available to materialize.";
        crash();
      }
      release_mmap();
    }

    void read_codes(uint64_t byte_offset, uint64_t n_bytes, uint8_t *out) const {
      uint64_t done = 0;
      const uint64_t file_offset = 2 * sizeof(uint32_t) + byte_offset;
      while (done < n_bytes) {
        ssize_t got = pread(stream_fd_, out + done, n_bytes - done, file_offset + done);
        if (got <= 0) {
          LOG(ERROR) << "Failed to stream PQ codes at byte offset " << (file_offset + done);
          crash();
        }
        done += (uint64_t) got;
      }
    }

    void read_code(uint32_t id, uint8_t *out) const {
      if (stream_fd_ < 0) {
        LOG(ERROR) << "PQ stream fd is not open.";
        crash();
      }
      const uint64_t file_offset = 2 * sizeof(uint32_t) + (uint64_t) id * pq_table.n_chunks;
      const uint64_t page = file_offset / SECTOR_LEN;
      const uint64_t page_off = file_offset % SECTOR_LEN;
      const uint8_t *page_data = stream_page(page);
      if (page_off + pq_table.n_chunks <= SECTOR_LEN) {
        memcpy(out, page_data + page_off, pq_table.n_chunks);
        return;
      }
      const uint64_t first = SECTOR_LEN - page_off;
      memcpy(out, page_data + page_off, first);
      const uint8_t *next_page = stream_page(page + 1);
      memcpy(out + first, next_page, pq_table.n_chunks - first);
    }

    void aggregate_coords_for_ids(const unsigned *ids, const uint64_t n_ids, uint8_t *out) {
      if (this->codes() != nullptr) {
        aggregate_coords(ids, n_ids, this->codes(), pq_table.n_chunks, out);
        return;
      }
      for (uint64_t i = 0; i < n_ids; i++) {
        read_code(ids[i], out + i * pq_table.n_chunks);
      }
    }

    struct StreamPageCache {
      int fd = -1;
      uint64_t generation = 0;
      std::unordered_map<uint64_t, size_t> page_to_slot;
      std::vector<std::array<uint8_t, SECTOR_LEN>> pages;

      void reset_if_needed(int new_fd, uint64_t new_generation) {
        if (fd == new_fd && generation == new_generation) {
          return;
        }
        fd = new_fd;
        generation = new_generation;
        page_to_slot.clear();
        pages.clear();
      }
    };

    uint64_t stream_cache_max_pages() const {
      const char *env = std::getenv("PIPEANN_PQ_STREAM_CACHE_PAGES");
      if (env == nullptr || *env == '\0') {
        return 1024;
      }
      char *end = nullptr;
      uint64_t value = std::strtoull(env, &end, 10);
      return (end == env || *end != '\0' || value == 0) ? 1024 : value;
    }

    const uint8_t *stream_page(uint64_t page) const {
      static thread_local StreamPageCache cache;
      cache.reset_if_needed(stream_fd_, stream_generation_);
      auto found = cache.page_to_slot.find(page);
      if (found != cache.page_to_slot.end()) {
        return cache.pages[found->second].data();
      }
      const uint64_t max_pages = stream_cache_max_pages();
      if (cache.pages.size() >= max_pages) {
        cache.page_to_slot.clear();
        cache.pages.clear();
      }
      size_t slot = cache.pages.size();
      cache.pages.emplace_back();
      ssize_t got = pread(stream_fd_, cache.pages.back().data(), SECTOR_LEN, page * SECTOR_LEN);
      if (got <= 0) {
        LOG(ERROR) << "Failed to read PQ stream page " << page;
        crash();
      }
      if (got < (ssize_t) SECTOR_LEN) {
        memset(cache.pages.back().data() + got, 0, SECTOR_LEN - got);
      }
      cache.page_to_slot[page] = slot;
      return cache.pages[slot].data();
    }

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
            LOG(ERROR) << "Hex: "
                       << "passed_train_data[" << i << "][" << j << "] = " << val1 << ", train_data[" << i << "][" << j
                       << "] = " << val2;
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

      // Simplified: no longer using rearrangement, dimensions are processed in natural order
      std::vector<uint32_t> chunk_offsets;

      size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
      size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
      size_t max_num_high = dim - (low_val * num_pq_chunks);

      // Build chunk_offsets directly - distribute dimensions evenly
      chunk_offsets.push_back(0);
      for (uint32_t b = 0; b < num_pq_chunks; b++) {
        size_t chunk_size = (b < max_num_high) ? high_val : low_val;
        chunk_offsets.push_back(chunk_offsets[b] + chunk_size);
      }

      full_pivot_data.reset(new float[num_centers * dim]);

      // Every thread compute one chunk, fast enough.
      auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, 1)
      for (size_t i = 0; i < num_pq_chunks; i++) {
        size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];

        if (cur_chunk_size == 0)
          continue;
        std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
        std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_train * cur_chunk_size);
        std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_train);

        memset((void *) cur_pivot_data.get(), 0, num_centers * cur_chunk_size * sizeof(float));

        for (int64_t j = 0; j < (int64_t) num_train; j++) {
          memcpy(cur_data.get() + j * cur_chunk_size, train_data.get() + j * dim + chunk_offsets[i],
                 cur_chunk_size * sizeof(float));
        }

        kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers);
        kmeans::run_elkan(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers,
                          max_k_means_reps, nullptr, closest_center.get());

        for (uint64_t j = 0; j < num_centers; j++) {
          memcpy(full_pivot_data.get() + j * dim + chunk_offsets[i], cur_pivot_data.get() + j * cur_chunk_size,
                 cur_chunk_size * sizeof(float));
        }
      }
      auto ed = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Kmeans time: " << std::chrono::duration<double>(ed - st).count() << "s.";

      // Create dummy identity rearrangement for backward compatibility
      std::vector<uint32_t> dummy_rearrangement(dim);
      for (uint32_t d = 0; d < dim; d++) {
        dummy_rearrangement[d] = d;
      }

      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = SECTOR_LEN;
      cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data.get(),
                                                                 (size_t) num_centers, dim, cumul_bytes[0]);
      cumul_bytes[2] = cumul_bytes[1] + pipeann::save_bin<float>(pq_pivots_path.c_str(), centroid.get(), (size_t) dim,
                                                                 1, cumul_bytes[1]);
      cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), dummy_rearrangement.data(),
                                                                    dummy_rearrangement.size(), 1, cumul_bytes[2]);
      cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets.data(),
                                                                    chunk_offsets.size(), 1, cumul_bytes[3]);
      pipeann::save_bin<uint64_t>(pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved pq pivot to " << pq_pivots_path << " of size " << cumul_bytes[cumul_bytes.size() - 1] << "B.";

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
      if (unlikely(num_centers > 256)) {
        LOG(ERROR) << "Error: number of centers is greater than 256";
        return -1;
      }

      this->pq_table.load_pq_centroid_bin(pq_pivots_path.c_str(), num_pq_chunks, 0);

      size_t num_points, dim;
      pipeann::get_bin_metadata(data_file, num_points, dim, offset);
      constexpr size_t BLOCK_SIZE = 131072;  // hard-coded max block size.

      std::ofstream compressed_file_writer(pq_compressed_vectors_path, std::ios::binary);
      uint32_t num_pq_chunks_u32 = num_pq_chunks;

      compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
      compressed_file_writer.write((char *) &num_pq_chunks_u32, sizeof(uint32_t));

      size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
      std::unique_ptr<uint8_t[]> block_compressed_base =
          std::make_unique<uint8_t[]>(block_size * (uint64_t) num_pq_chunks);
      std::memset(block_compressed_base.get(), 0, block_size * (uint64_t) num_pq_chunks);

      std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
      std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

      size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

      std::ifstream base_reader(data_file, std::ios::binary);
      base_reader.seekg(offset + sizeof(uint32_t) * 2, std::ios::beg);  // Skip header and offset
      for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = std::min((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

// Subtract centroid directly from block_data_float (parallel over points)
#pragma omp parallel for schedule(static, 512)
        for (uint64_t p = 0; p < cur_blk_size; p++) {
          for (uint64_t d = 0; d < dim; d++) {
            block_data_float[p * dim + d] -= pq_table.centroid[d];
          }
        }

        // Parallelize over PQ chunks - each chunk is independent
#pragma omp parallel for schedule(static, 1)
        for (size_t i = 0; i < num_pq_chunks; i++) {
          size_t cur_chunk_size = pq_table.chunk_offsets[i + 1] - pq_table.chunk_offsets[i];
          if (cur_chunk_size == 0)
            continue;

          std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
          std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
          std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(cur_blk_size);

          // Extract chunk data from block
          for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
            std::memcpy(cur_data.get() + j * cur_chunk_size,
                        block_data_float.get() + j * dim + pq_table.chunk_offsets[i], cur_chunk_size * sizeof(float));
          }

          // Extract chunk pivots
          for (int64_t j = 0; j < (int64_t) num_centers; j++) {
            std::memcpy(cur_pivot_data.get() + j * cur_chunk_size,
                        pq_table.tables + j * dim + pq_table.chunk_offsets[i], cur_chunk_size * sizeof(float));
          }

          // Compute closest centers for this chunk
          kmeans::compute_closest_centers(cur_data.get(), cur_blk_size, cur_chunk_size, cur_pivot_data.get(),
                                          num_centers, 1, closest_center.get());

          // Write results to output
          for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
            block_compressed_base[j * num_pq_chunks + i] = closest_center[j];
          }
        }
        compressed_file_writer.write((char *) (block_compressed_base.get()),
                                     cur_blk_size * num_pq_chunks * sizeof(uint8_t));
      }
      compressed_file_writer.close();
      return 0;
    }
  };
}  // namespace pipeann
