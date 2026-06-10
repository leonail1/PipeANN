# Valgrind Massif single-query heap breakdown

Run: `full_node4_r96_rd0_st1_fg4_batch32_gt16_save_static_sift100m_3m6m_20260610T100604_valgrind_massif_pq32_L40_aio_nosimd_20260610T154115`

Scope: SIFT1M, `range_s10`, `L=40`, `k=10`, PQ32. The binary was rebuilt in a temporary worktree with `-march=x86-64`, SIMD definitions disabled, and AIO, only so Valgrind 3.18 can decode the instruction stream. The production binary and committed source remain on the uring/native path.

Massif peak heap snapshot:

| Component | Bytes | Share | Evidence |
|---|---:|---:|---|
| PQ compressed codes | 32,000,000 | 69.20% of peak heap | `PQNeighbor<float>::load` -> `load_bin<unsigned char>` for `*_pq_compressed.bin` |
| PQ all-to-all centroid distance table | 8,388,608 | 18.14% | `FixedChunkPQTable<float>::post_load_pq_table` allocates `256 * 256 * n_chunks` floats |
| tag file temporary vector | 4,000,000 | 8.65% | `SSDIndex<float,uint32_t>::load_tags` loads `*_disk.index.tags`, then skips 1M identity tags |
| query buffer aligned scratch | 524,288 | 1.13% | `SSDIndex::init_query_buf` |
| reader IO buffer | 524,288 | 1.13% | `AlignedFileReader::alloc_io_buf` |

Post-load/search steady heap snapshot:

| Component | Bytes | Evidence |
|---|---:|---|
| PQ compressed codes | 32,000,000 | same `PQNeighbor<float>::load` allocation |
| range quantized bucket ids | 1,000,000 | `SortedRangeAttrIndex::load_quantized_buckets` |
| PQ all-to-all centroid distance table | 8,388,608 | same `FixedChunkPQTable::post_load_pq_table` allocation |
| query/IO buffers | about 1,048,576 | two 512KB aligned buffers |
| small allocations and allocator overhead | about 1.4MB useful heap plus allocator/RSS overhead | below Massif threshold |

Interpretation:

- `PQ32` means 32 bytes per vector, not 32 bits. For 1,000,000 vectors the compressed sidecar alone is exactly `1,000,000 * 32 = 32,000,000` bytes.
- The second unavoidable-looking PQ structure in the current implementation is the all-to-all centroid table: `256 * 256 * 32 * 4 = 8,388,608` bytes.
- Hashmap preallocation is no longer visible in the top Massif allocations for the current single-query path. The earlier hashmap/tag optimizations removed the identity-tag and empty-map costs; remaining heap is dominated by PQ code storage and PQ lookup tables.
- Removing the temporary tag vector can reduce load-time peak by up to 4MB, but it will not solve the steady single-query RSS because the PQ code storage plus PQ lookup table already exceed 40MB before allocator/library overhead.
