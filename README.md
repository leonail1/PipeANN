# PipeANN for OpenHarmony

A filtered vector search engine on SSD, built for the OpenHarmony vector+scalar
joint query (向标联合查询) workload.

This project is built on top of [PipeANN](https://github.com/thustorage/PipeANN)
**v0.2.0** (OSDI '25), with selected upstream fixes synced afterwards. It keeps
the PipeANN SSD-resident graph index kernel and adds our own designs for
filtered search routing, dynamic updates from an empty index, and an
OpenHarmony-oriented acceptance suite.

## Our Designs on Top of PipeANN

- **Hybrid filtered search routing.** Every filtered query is routed by an
  auto-calibrated threshold τ on the candidate count: selective queries go
  through a prefilter path (candidate generation + exact rerank), while
  unselective ones stay on the graph search path. Routing decisions are
  per-query and can be forced via overrides.
- **DenseBit label index.** Labels are stored with a bitmap/posting-list dual
  encoding (dense bitmaps for high-selectivity labels, posting lists for
  low-selectivity ones) in an mmap-friendly file format, shrinking a ~30 MB
  sparse-matrix label file for SIFT1M-scale data to under 1 MB.
  `equality` / `subset` / `range` / `match_all` filter semantics are supported.
- **Zero-start dynamic index.** Below a bootstrap threshold (10k vectors),
  queries are served by an in-memory exact path; the disk graph index is built
  and swapped in automatically once the threshold is crossed. Inserts use the
  native dynamic insert path (low-RSS), deletes are cheap mark-deletes with
  background merge/save, and PQ codebooks are retrained after batch inserts to
  suppress quantization drift.
- **Pluggable I/O engines.** io_uring (default), Linux AIO, or SPDK, selected
  at configure time (`-DIO_ENGINE=uring|aio|spdk`).

## OpenHarmony Acceptance Suite

`openharmony_acceptance/` contains a C++ acceptance runner that drives the
index through the six OpenHarmony requirements:

1. Static filtered search: recall@10 ≥ 98% with avg latency < 10 ms
2. 5-cycle dynamic delete/insert: recall ≥ 98% and avg < 10 ms after each cycle
3. Single-query RSS < 30 MB
4. Extra index space expansion < 1× raw vectors
5. Low-cost mark-delete
6. Foreground search latency during background updates < 10 ms

It generates labels, computes exact ground truth (with caching), runs static
and dynamic test matrices, audits space/RSS, and emits a machine-readable
`acceptance_summary.json`. See
[openharmony_acceptance/README.md](openharmony_acceptance/README.md) for usage.
Experiment data and results are kept locally and are **not** part of this
repository (see `.gitignore`).

## Repository Layout

```
include/                 index headers (SSD index, filters, PQ/RaBitQ, utils)
src/                     index kernel (search, update, I/O engines)
openharmony_acceptance/  OpenHarmony acceptance runner (C++ tools + scripts)
tests/                   index build/search tools and unit drivers
scripts/                 benchmark helper scripts
pipeann/, tests_py/      Python bindings and examples
third_party/             bundled dependencies (liburing, etc.)
docs/                    PipeANN documentation
```

## Build

Requirements: CMake ≥ 3.16, a C++17 compiler, OpenMP, BLAS/LAPACK (MKL or
OpenBLAS); tcmalloc is used when available. io_uring requires Linux ≥ 5.1
(AIO fallback works everywhere).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIO_ENGINE=uring
make -j
```

## Quick Start

```bash
# 1. Smoke test the acceptance pipeline on synthetic data
bash openharmony_acceptance/scripts/run_smoke.sh

# 2. Full acceptance (needs base/update/query vectors in .bin format)
BASE_BIN=/path/base_1m.bin UPDATES_BIN=/path/updates_3m.bin QUERY_BIN=/path/query.bin \
PQ_BYTES=16 bash openharmony_acceptance/scripts/run_full_acceptance.sh
```

## 📖 Citation

If you use PipeANN in your research, please cite our papers:

```bibtex
@misc{arxiv26pipeannfilter,
      title={PipeANN-Filter: An Efficient Filtered Vector Search System on SSD}, 
      author={Hao Guo and Jiwu Shu and Youyou Lu},
      year={2026},
      eprint={2605.17992},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2605.17992}, 
}

@inproceedings{fast26odinann,
  author    = {Hao Guo and Youyou Lu},
  title     = {OdinANN: Direct Insert for Consistently Stable Performance 
               in Billion-Scale Graph-Based Vector Search},
  booktitle = {24th USENIX Conference on File and Storage Technologies (FAST 26)},
  year      = {2026},
  address   = {Santa Clara, CA},
  pages     = {133--147},
  publisher = {USENIX Association}
}

@inproceedings{osdi25pipeann,
  author    = {Hao Guo and Youyou Lu},
  title     = {Achieving Low-Latency Graph-Based Vector Search via 
               Aligning Best-First Search Algorithm with SSD},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year      = {2025},
  address   = {Boston, MA},
  pages     = {171--186},
  publisher = {USENIX Association}
}
```
See [Repository Layout](docs/repository-layout.md) for code layout and scripts.
