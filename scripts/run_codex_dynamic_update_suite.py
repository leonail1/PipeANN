#!/usr/bin/env python3
"""Run the Codex dynamic update experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


BUCKETS = [
    ("u1e-03", 0.001),
    ("u3e-03", 0.003),
    ("u1e-02", 0.01),
    ("u5e-02", 0.05),
    ("u1e-01", 0.10),
    ("u25", 0.25),
    ("u30", 0.30),
    ("u50", 0.50),
    ("u75", 0.75),
    ("u100", 1.00),
]

EXP4_L_CANDIDATES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000]
BASELINE_THREADS = [1]
BASELINE_ROUTES = ["prefilter", "graph"]
BASELINE_LATENCY_CUTOFF_US = 100_000.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--base-bin", type=Path, default=Path("data/sift1m/sift_base.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("data/sift1m/sift_query.bin"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--resume-after-exp1",
        action="store_true",
        help="Reuse exp1 results.jsonl and continue with exp2. Useful after manually stopping exp1 early.",
    )
    parser.add_argument(
        "--resume-after-exp2",
        action="store_true",
        help="Reuse exp1/exp2 results and continue with exp3.",
    )
    parser.add_argument(
        "--resume-after-exp3",
        action="store_true",
        help="Reuse exp1/exp2/exp3 results and continue with exp4.",
    )
    parser.add_argument(
        "--rerun-exp4",
        action="store_true",
        help="When resuming after exp3, discard existing exp4 rows and rerun exp4.",
    )
    parser.add_argument(
        "--stop-after-exp4",
        action="store_true",
        help="Stop after exp4, then plot and clean. Useful when only exp4 figures need regeneration.",
    )
    parser.add_argument(
        "--only-exp1",
        action="store_true",
        help="Run only exp1_insert_vs_build_threads, then plot and clean.",
    )
    parser.add_argument(
        "--only-exp2",
        action="store_true",
        help="Run only exp2_stage_recall_build_vs_insert, including the larger-seed recall curves.",
    )
    parser.add_argument(
        "--only-exp3",
        action="store_true",
        help="Run only exp3_search_during_insert, then plot and clean.",
    )
    parser.add_argument(
        "--only-exp4",
        action="store_true",
        help="Run only exp4_intersect_range_selectivity, then plot and clean.",
    )
    parser.add_argument(
        "--only-exp5",
        action="store_true",
        help="Run only exp5_index_bloat_by_size, then plot and clean.",
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Run only exp_baseline on a direct-build 1M index, then plot and clean.",
    )
    parser.add_argument(
        "--only-exp2-seed-sweep",
        action="store_true",
        help="Run only the exp2 250k/500k seed sweep, then plot and clean.",
    )
    parser.add_argument(
        "--rerun-exp2-seed-sweep",
        action="store_true",
        help="Discard existing exp2 seed-sweep rows before running it.",
    )
    parser.add_argument(
        "--exp2-seed-sweep-starts",
        default="250000,500000",
        help="Comma-separated direct-build starting sizes for exp2 seed sweep.",
    )
    parser.add_argument(
        "--exp2-seed-sweep-l",
        type=int,
        default=100,
        help="Fixed search L for exp2 seed sweep, matching the PPT recall curve.",
    )
    parser.add_argument(
        "--exp2-l",
        type=int,
        default=100,
        help="Fixed search L for exp2 direct-build and 10k-seed insert curves.",
    )
    parser.add_argument(
        "--rerun-baseline",
        action="store_true",
        help="Discard existing exp_baseline rows before running it.",
    )
    parser.add_argument(
        "--baseline-query-count",
        type=int,
        default=1000,
        help="Number of SIFT queries to use for exp_baseline. Intended range: 1000 to 10000.",
    )
    parser.add_argument(
        "--baseline-pq-bytes",
        type=int,
        default=32,
        help="PQ bytes for exp_baseline direct-build index. Kept separate from dynamic-suite PQ bytes.",
    )
    parser.add_argument(
        "--exp4-pq-bytes",
        type=int,
        default=32,
        help="PQ bytes for exp4 direct-build starting index. Kept separate from the global dynamic-suite PQ bytes.",
    )
    parser.add_argument(
        "--exp3-pq-bytes",
        type=int,
        default=32,
        help="PQ bytes for exp3 selectivity/during-insert indexes. Matches exp4 by default.",
    )
    parser.add_argument(
        "--exp3-total-n",
        type=int,
        default=0,
        help="Override total vectors for exp3 only. Used for 1M->2M foreground-query insertion runs.",
    )
    parser.add_argument(
        "--exp3-start-n",
        type=int,
        default=0,
        help="Override starting vectors for exp3 only. Defaults to half of --exp3-total-n when set.",
    )
    parser.add_argument(
        "--exp3-no-insert-n",
        type=int,
        default=0,
        help="Override the no-insert baseline size for exp3. Defaults to exp3 total_n.",
    )
    parser.add_argument(
        "--exp4-fixed-graph-l",
        type=int,
        default=100,
        help="Fixed graph-search L for exp4 high-selectivity recall comparison.",
    )
    parser.add_argument("--build-r", type=int, default=64)
    parser.add_argument("--build-l", type=int, default=96)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--gt-numa-node", type=int, default=1)
    parser.add_argument("--gt-threads", type=int, default=0, help="0 means all logical CPUs on --gt-numa-node.")
    return parser.parse_args()


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(line_buffering=True, write_through=True)


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], cwd: Path, timeout: int | None = None, env: dict[str, str] | None = None) -> None:
    log("+ " + " ".join(command))
    run_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env:
        run_env.update(env)
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout, env=run_env)


def run_capture(command: list[str], cwd: Path, timeout: int | None = None) -> float:
    start = time.perf_counter()
    run(command, cwd, timeout)
    return time.perf_counter() - start


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as reader:
        raw = reader.read(8)
    if len(raw) != 8:
        raise ValueError(f"failed to read bin header from {path}")
    return struct.unpack("ii", raw)


def parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for part in cpu_list.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def numa_node_cpus(node: int) -> list[int]:
    cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if cpulist.exists():
        return parse_cpu_list(cpulist.read_text(encoding="utf-8"))
    return list(range(os.cpu_count() or 1))


def gt_command_and_env(args: argparse.Namespace, command: list[str]) -> tuple[list[str], dict[str, str]]:
    cpus = numa_node_cpus(args.gt_numa_node)
    threads = args.gt_threads if args.gt_threads > 0 else len(cpus)
    env = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "threads",
    }
    if shutil.which("numactl") is not None:
        command = ["numactl", f"--cpunodebind={args.gt_numa_node}", f"--membind={args.gt_numa_node}", *command]
    return command, env


def copy_prefix_bin(source: Path, destination: Path, npoints: int, dim: int) -> None:
    if destination.exists() and read_bin_header(destination)[0] == npoints:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("ii", npoints, dim))
        remaining = npoints * dim * 4
        while remaining:
            chunk = src.read(min(32 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"unexpected EOF while slicing {source}")
            dst.write(chunk)
            remaining -= len(chunk)


def read_first_query_csv(query_bin: Path, dim: int) -> str:
    with query_bin.open("rb") as reader:
        npoints, file_dim = struct.unpack("ii", reader.read(8))
        if npoints < 1:
            raise RuntimeError(f"query bin has no vectors: {query_bin}")
        if file_dim != dim:
            raise RuntimeError(f"query dim mismatch for {query_bin}: {file_dim} != {dim}")
        values = struct.unpack(f"{dim}f", reader.read(dim * 4))
    return ",".join(f"{value:.9g}" for value in values)


def read_first_spmat_labels_csv(spmat_path: Path) -> str:
    with spmat_path.open("rb") as reader:
        header = reader.read(24)
        if len(header) != 24:
            raise RuntimeError(f"failed to read spmat header: {spmat_path}")
        nrow, _ncol, nnz = struct.unpack("<qqq", header)
        if nrow < 1:
            return ""
        indptr = struct.unpack(f"<{nrow + 1}q", reader.read(8 * (nrow + 1)))
        indices_data = reader.read(4 * nnz)
        indices = struct.unpack(f"<{nnz}i", indices_data) if nnz else ()
        data_data = reader.read(4 * nnz)
        data = struct.unpack(f"<{nnz}f", data_data) if nnz else ()
    labels = [str(indices[pos]) for pos in range(indptr[0], indptr[1]) if data[pos] != 0.0]
    return ",".join(labels)


def write_spmat(path: Path, nrow: int, ncol: int, rows: Iterable[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    with path.open("wb") as writer:
        writer.write(struct.pack("qqq", nrow, ncol, len(indices)))
        writer.write(struct.pack(f"{len(indptr)}q", *indptr))
        if indices:
            writer.write(struct.pack(f"{len(indices)}i", *indices))
            writer.write(struct.pack(f"{len(indices)}f", *([1.0] * len(indices))))


def write_selectivity_label_file(path: Path, npoints: int, reference_npoints: int) -> Path:
    ncols = len(BUCKETS)
    cutoffs = [max(1, min(reference_npoints, int(math.ceil(reference_npoints * sel)))) for _, sel in BUCKETS]
    if path.exists():
        return path

    def rows() -> Iterable[list[int]]:
        for point_id in range(npoints):
            yield [label_id for label_id, cutoff in enumerate(cutoffs) if point_id < cutoff]

    write_spmat(path, npoints, ncols, rows())
    return path


def ensure_query_label_files(label_dir: Path, query_count: int) -> dict[str, Path]:
    ncols = len(BUCKETS)
    query_labels: dict[str, Path] = {}
    for label_id, (bucket, _) in enumerate(BUCKETS):
        path = label_dir / f"query_{query_count}_{bucket}.spmat"
        if not path.exists():
            write_spmat(path, query_count, ncols, ([label_id] for _ in range(query_count)))
        query_labels[bucket] = path
    return query_labels


def ensure_range_query_label_files(label_dir: Path, query_count: int) -> dict[str, Path]:
    ncols = len(BUCKETS)
    query_labels: dict[str, Path] = {}
    for label_id, (bucket, _) in enumerate(BUCKETS):
        path = label_dir / f"query_{query_count}_range_{bucket}.spmat"
        if not path.exists():
            if label_id == 0:
                row = [0]
            else:
                row = [0, label_id]
            write_spmat(path, query_count, ncols, (row for _ in range(query_count)))
        query_labels[bucket] = path
    return query_labels


def stage_name(npoints: int) -> str:
    return f"{npoints // 1000}k" if npoints < 1_000_000 else f"{npoints // 1_000_000}m"


def prepare_assets(root: Path, args: argparse.Namespace, stages: list[int], seed_n: int, query_count: int) -> dict:
    base_total, dim = read_bin_header(args.base_bin)
    query_total, query_dim = read_bin_header(args.query_bin)
    if dim != query_dim:
        raise ValueError("base/query dimensionality mismatch")
    if max(stages + [seed_n]) > base_total:
        raise ValueError("requested stage exceeds base dataset")
    if query_count > query_total:
        raise ValueError("requested query count exceeds query dataset")

    data_dir = root / "data"
    label_dir = root / "labels"
    query_small = data_dir / f"sift_query_{query_count}.bin"
    copy_prefix_bin(args.query_bin, query_small, query_count, dim)

    base_bins: dict[int, Path] = {}
    label_files: dict[int, Path] = {}
    for npoints in sorted(set(stages + [seed_n])):
        short = stage_name(npoints)
        base_bin = data_dir / f"sift_base_{short}.bin"
        copy_prefix_bin(args.base_bin, base_bin, npoints, dim)
        base_bins[npoints] = base_bin

        label_path = label_dir / f"base_{short}.spmat"
        label_files[npoints] = write_selectivity_label_file(label_path, npoints, npoints)

    return {
        "dim": dim,
        "query_bin": query_small,
        "base_bins": base_bins,
        "label_files": label_files,
        "query_labels": ensure_query_label_files(label_dir, query_count),
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=keys, lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def clean_prefix(prefix: Path) -> None:
    parent = prefix.parent
    if not parent.exists():
        return
    for path in parent.glob(prefix.name + "*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def prefix_exists(prefix: Path) -> bool:
    return Path(str(prefix) + "_disk.index").exists()


def clean_large_files(root: Path) -> None:
    keep = {".json", ".jsonl", ".csv", ".png", ".md", ".sh"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix not in keep:
            path.unlink()


def clear_experiment_dir(path: Path, keep_names: set[str] | None = None) -> None:
    keep_names = keep_names or {"start.sh", "README.md", "ref.png"}
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_tools(repo: Path) -> None:
    if not (repo / "build" / "CMakeCache.txt").exists():
        run(["cmake", "-S", ".", "-B", "build"], repo)
    run([
        "cmake", "--build", "build", "--target", "dynamic_update_suite_driver", "build_disk_index",
        "compute_groundtruth", "calibrate_hybrid_threshold", "-j",
    ], repo)


def build_index(repo: Path, args: argparse.Namespace, base_bin: Path, prefix: Path, labels: Path, threads: int) -> float:
    clean_prefix(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return run_capture([
        str(repo / "build/tests/build_disk_index"),
        "float",
        str(base_bin),
        str(prefix),
        str(args.build_r),
        str(args.build_l),
        str(args.pq_bytes),
        str(args.memory_gb),
        str(threads),
        args.metric,
        args.nbr_type,
        "spmat",
        str(labels),
    ], repo)


def build_index_with_pq_bytes(repo: Path, args: argparse.Namespace, base_bin: Path, prefix: Path, labels: Path,
                              threads: int, pq_bytes: int) -> float:
    original_pq_bytes = args.pq_bytes
    args.pq_bytes = pq_bytes
    try:
        return build_index(repo, args, base_bin, prefix, labels, threads)
    finally:
        args.pq_bytes = original_pq_bytes


def compute_truth(repo: Path, args: argparse.Namespace, base_bin: Path, query_bin: Path, out: Path,
                  base_labels: Path | None = None, query_labels: Path | None = None,
                  selector_type: str = "intersect") -> None:
    if out.exists():
        query_count, _ = read_bin_header(query_bin)
        try:
            truth_count, truth_dim = read_bin_header(out)
            if truth_count == query_count and truth_dim == args.k:
                return
        except Exception:
            pass
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    if base_labels is None:
        command = [
            str(repo / "build/tests/utils/compute_groundtruth"),
            "float", args.metric, str(base_bin), str(query_bin), str(args.k), str(out), "null", "null",
        ]
    else:
        command = [
            str(repo / "build/tests/utils/compute_groundtruth"),
            "float", args.metric, str(base_bin), str(query_bin), str(args.k), str(out), "null",
            "spmat", selector_type, str(base_labels), str(query_labels),
        ]
    command, env = gt_command_and_env(args, command)
    run(command, repo, env=env)


def driver(repo: Path, args: argparse.Namespace, *, mode: str, source: Path, dest: Path | None, jsonl: Path,
           data_bin: Path | None = None, base_label_file: Path | None = None, query_bin: Path | None = None,
           truthset: Path | None = None, query_label_file: Path | None = None, selector_type: str = "none",
           insert_start: int = 0, insert_count: int = 0, delete_start: int = 0, delete_count: int = 0,
           insert_threads: int = 1, search_threads: int = 1, merge_threads: int = 1, search_l: int = 60,
           query_limit: int = 0, route: str = "auto", query_vector_csv: str | None = None,
           query_label_csv: str | None = None, single_query_static_rss: bool = False) -> dict:
    before = len(load_jsonl(jsonl))
    command = [
        str(repo / "build/tests/dynamic_update_suite_driver"),
        "--mode", mode,
        "--source-prefix", str(source),
        "--jsonl-output", str(jsonl),
        "--insert-threads", str(insert_threads),
        "--search-threads", str(search_threads),
        "--merge-threads", str(merge_threads),
        "--build-l", str(args.build_l),
        "--build-r", str(args.build_r),
        "--beamwidth", str(args.beamwidth),
        "--k", str(args.k),
        "--search-l", str(search_l),
        "--metric", args.metric,
        "--route", route,
    ]
    if dest is not None:
        clean_prefix(dest)
        command += ["--dest-prefix", str(dest)]
    if data_bin is not None:
        command += ["--data-bin", str(data_bin)]
    if base_label_file is not None:
        command += ["--base-label-file", str(base_label_file)]
    if query_bin is not None:
        command += ["--query-bin", str(query_bin)]
    if query_vector_csv is not None:
        command += ["--query-vector-csv", query_vector_csv]
    if truthset is not None:
        command += ["--truthset-bin", str(truthset)]
    if query_label_file is not None:
        command += ["--query-label-file", str(query_label_file), "--selector-type", selector_type]
    if query_label_csv is not None:
        command += ["--query-label-csv", query_label_csv, "--selector-type", selector_type]
    if insert_count:
        command += ["--insert-start", str(insert_start), "--insert-count", str(insert_count)]
    if delete_count:
        command += ["--delete-start", str(delete_start), "--delete-count", str(delete_count)]
    if query_limit:
        command += ["--query-limit", str(query_limit)]
    if single_query_static_rss:
        command += ["--single-query-static-rss"]
    env = None
    if single_query_static_rss:
        env = {"PIPEANN_PQ_MMAP": "1", "PIPEANN_PQ_MMAP_DROP_CACHE": "1"}
    run(command, repo, env=env)
    rows = load_jsonl(jsonl)
    if len(rows) <= before:
        raise RuntimeError(f"driver did not append a row to {jsonl}")
    return rows[-1]


def static_hybrid_search(repo: Path, args: argparse.Namespace, *, prefix: Path, jsonl: Path, query_bin: Path | None,
                         truthset: Path | None, query_label_file: Path | None, route: str, threads: int,
                         search_l: int, selector_type: str = "intersect",
                         query_vector_csv: str | None = None, query_label_csv: str | None = None,
                         single_query_static_rss: bool = False) -> dict:
    before = len(load_jsonl(jsonl))
    command = [
        str(repo / "build/tests/search_disk_index_hybrid"),
        "float",
        str(prefix),
        str(threads),
        str(args.beamwidth),
        str(query_bin) if query_bin is not None else "null",
        str(truthset) if truthset is not None else "null",
        str(args.k),
        args.metric,
        args.nbr_type,
        selector_type,
        str(query_label_file) if query_label_file is not None else "null",
        route,
        "0",
        "0",
        str(search_l),
        "--jsonl-output",
        str(jsonl),
    ]
    if query_vector_csv is not None:
        command += ["--query-vector-csv", query_vector_csv]
    if query_label_csv is not None:
        command += ["--query-label-csv", query_label_csv]
    if single_query_static_rss:
        command += ["--single-query-static-rss"]
    env = None
    if single_query_static_rss:
        env = {"PIPEANN_PQ_MMAP": "1", "PIPEANN_PQ_MMAP_DROP_CACHE": "1"}
    run(command, repo, env=env)
    rows = load_jsonl(jsonl)
    if len(rows) <= before:
        raise RuntimeError(f"static hybrid search did not append to {jsonl}")
    row = rows[-1]
    if "recall" in row and "recall@10" not in row:
        row["recall@10"] = row.get("recall")
    if "L" in row and "chosen_L" not in row:
        row["chosen_L"] = row.get("L")
    if "avg_latency_us" in row:
        row["avg_latency_us"] = float(row["avg_latency_us"])
    if "qps" in row:
        row["qps"] = float(row["qps"])
    return row


def calibrate_auto_route(repo: Path, args: argparse.Namespace, prefix: Path, query_bin: Path,
                         query_labels: list[Path], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(repo / "build/tests/calibrate_hybrid_threshold"),
        "float",
        str(prefix),
        "32",
        str(args.beamwidth),
        str(args.k),
        args.metric,
        args.nbr_type,
        "0",
        "100",
    ]
    for label_file in query_labels:
        command += ["intersect", str(query_bin), str(label_file), "200"]
    with log_path.open("w", encoding="utf-8", buffering=1) as writer:
        log("+ " + " ".join(command))
        subprocess.run(
            command,
            cwd=repo,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )


def exp_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_exp1(repo: Path, args: argparse.Namespace, assets: dict, total_n: int, seed_n: int, threads: list[int]) -> list[dict]:
    out = exp_dir(args.out_dir, "exp1_insert_vs_build_threads")
    clear_experiment_dir(out)
    rows: list[dict] = []
    for threads_n in threads:
        seed_prefix = out / "tmp" / f"seed_t{threads_n}"
        seed_build_s = build_index(repo, args, assets["base_bins"][seed_n], seed_prefix, assets["label_files"][seed_n], threads_n)
        inserted_prefix = out / "tmp" / f"inserted_t{threads_n}"
        insert_row = driver(
            repo, args, mode="insert-only", source=seed_prefix, dest=inserted_prefix, jsonl=out / "driver.jsonl",
            data_bin=assets["base_bins"][total_n], base_label_file=assets["label_files"][total_n],
            insert_start=seed_n, insert_count=total_n - seed_n, insert_threads=threads_n, merge_threads=threads_n,
        )
        direct_prefix = out / "tmp" / f"direct_t{threads_n}"
        build_1m_s = build_index(repo, args, assets["base_bins"][total_n], direct_prefix, assets["label_files"][total_n], threads_n)
        row = {
            "status": "ok", "threads": threads_n, "seed_n": seed_n, "points": total_n,
            "seed_build_s": seed_build_s, "insert_remaining_s": insert_row["insert_elapsed_s"],
            "insert_total_s": seed_build_s + insert_row["insert_elapsed_s"], "build_1m_s": build_1m_s,
        }
        rows.append(row)
        append_jsonl(out / "results.jsonl", row)
        clean_prefix(seed_prefix); clean_prefix(inserted_prefix); clean_prefix(direct_prefix)
    write_csv(out / "table.csv", rows)
    return rows


def choose_stage_l(repo: Path, args: argparse.Namespace, assets: dict, direct_prefix: Path, insert_prefix: Path,
                   total_n: int, query_count: int, out: Path) -> tuple[int, str]:
    truth = out / "truth" / f"gt_{stage_name(total_n)}.bin"
    compute_truth(repo, args, assets["base_bins"][total_n], assets["query_bin"], truth)
    for l_value in [40, 60, 80, 100]:
        recalls = []
        for prefix, name in [(direct_prefix, "direct"), (insert_prefix, "insert")]:
            row = driver(repo, args, mode="measure-dynamic-search", source=prefix, dest=None,
                         jsonl=out / "l_sweep.jsonl", query_bin=assets["query_bin"], truthset=truth,
                         search_threads=32, search_l=l_value, query_limit=query_count)
            recalls.append(row["recall@10"])
        if max(recalls) < 99.8 and min(recalls) > 80:
            return l_value, "selected"
    return 60, "fallback"


def run_exp2(repo: Path, args: argparse.Namespace, assets: dict, stages: list[int], seed_n: int, query_count: int) -> list[dict]:
    out = exp_dir(args.out_dir, "exp2_stage_recall_build_vs_insert")
    clear_experiment_dir(out)
    rows: list[dict] = []
    direct_prefixes: dict[int, Path] = {}
    insert_prefixes: dict[int, Path] = {}
    for npoints in stages:
        prefix = out / "tmp" / f"direct_{stage_name(npoints)}"
        build_index(repo, args, assets["base_bins"][npoints], prefix, assets["label_files"][npoints], 32)
        direct_prefixes[npoints] = prefix

    prev_n = seed_n
    prev_prefix = out / "tmp" / "seed"
    build_index(repo, args, assets["base_bins"][seed_n], prev_prefix, assets["label_files"][seed_n], 32)
    for npoints in stages:
        dest = out / "tmp" / f"insert_{stage_name(npoints)}"
        driver(repo, args, mode="insert-only", source=prev_prefix, dest=dest, jsonl=out / "insert_driver.jsonl",
               data_bin=assets["base_bins"][npoints], base_label_file=assets["label_files"][npoints],
               insert_start=prev_n, insert_count=npoints - prev_n, insert_threads=32, merge_threads=32)
        insert_prefixes[npoints] = dest
        prev_n, prev_prefix = npoints, dest

    chosen_l, reason = args.exp2_l, f"fixed_L_{args.exp2_l}"
    for npoints in stages:
        truth = out / "truth" / f"gt_{stage_name(npoints)}.bin"
        compute_truth(repo, args, assets["base_bins"][npoints], assets["query_bin"], truth)
        for path_name, prefix in [("direct_build", direct_prefixes[npoints]), ("incremental_insert", insert_prefixes[npoints])]:
            search_row = driver(repo, args, mode="measure-dynamic-search", source=prefix, dest=None,
                                jsonl=out / "search_driver.jsonl", query_bin=assets["query_bin"], truthset=truth,
                                search_threads=1, search_l=chosen_l, query_limit=query_count)
            row = {"status": "ok", "path": path_name, "points": npoints, "chosen_L": chosen_l,
                   "l_choice": reason, **search_row}
            rows.append(row)
            append_jsonl(out / "results.jsonl", row)
    write_csv(out / "table.csv", rows)
    for prefix in list(direct_prefixes.values()) + list(insert_prefixes.values()) + [out / "tmp" / "seed"]:
        clean_prefix(prefix)
    return rows


def run_exp2_seed_sweep(repo: Path, args: argparse.Namespace, assets: dict, stages: list[int],
                        start_points: list[int], query_count: int) -> list[dict]:
    out = exp_dir(args.out_dir, "exp2_stage_recall_build_vs_insert")
    results_path = out / "seed_sweep_results.jsonl"
    table_path = out / "seed_sweep_table.csv"
    insert_log = out / "seed_sweep_insert_driver.jsonl"
    search_log = out / "seed_sweep_search_driver.jsonl"
    if args.rerun_exp2_seed_sweep:
        for stale in [results_path, table_path, insert_log, search_log, out / "seed_sweep_recall.png"]:
            if stale.exists():
                stale.unlink()
        for prefix in list((out / "tmp").glob("seed_*")) if (out / "tmp").exists() else []:
            clean_prefix(prefix)
    rows: list[dict] = [] if args.rerun_exp2_seed_sweep else load_jsonl(results_path)
    completed = {(int(row.get("start_points", 0)), int(row.get("points", 0))) for row in rows}

    valid_starts = sorted({start for start in start_points if start in stages})
    if not valid_starts:
        raise ValueError(f"no valid --exp2-seed-sweep-starts in stages {stages}: {start_points}")

    for start_n in valid_starts:
        target_points = [stage for stage in stages if stage >= start_n]
        if all((start_n, npoints) in completed for npoints in target_points):
            continue
        prev_n = start_n
        prev_prefix = out / "tmp" / f"seed_{stage_name(start_n)}"
        if not prefix_exists(prev_prefix):
            build_index(repo, args, assets["base_bins"][start_n], prev_prefix, assets["label_files"][start_n], 32)

        for npoints in target_points:
            if npoints == start_n:
                prefix = prev_prefix
            else:
                prefix = out / "tmp" / f"seed_{stage_name(start_n)}_to_{stage_name(npoints)}"
                if not prefix_exists(prefix):
                    driver(repo, args, mode="insert-only", source=prev_prefix, dest=prefix,
                           jsonl=insert_log, data_bin=assets["base_bins"][npoints],
                           base_label_file=assets["label_files"][npoints], insert_start=prev_n,
                           insert_count=npoints - prev_n, insert_threads=32, merge_threads=32)
                prev_n, prev_prefix = npoints, prefix

            if (start_n, npoints) in completed:
                continue
            truth = out / "truth" / f"gt_{stage_name(npoints)}.bin"
            compute_truth(repo, args, assets["base_bins"][npoints], assets["query_bin"], truth)
            search_row = driver(repo, args, mode="measure-dynamic-search", source=prefix, dest=None,
                                jsonl=search_log, query_bin=assets["query_bin"], truthset=truth,
                                search_threads=1, search_l=args.exp2_seed_sweep_l, query_limit=query_count)
            row = {
                "status": "ok",
                "path": f"direct_seed_{stage_name(start_n)}",
                "start_points": start_n,
                "points": npoints,
                "chosen_L": args.exp2_seed_sweep_l,
                **search_row,
            }
            rows.append(row)
            completed.add((start_n, npoints))
            append_jsonl(results_path, row)

        clean_prefix(out / "tmp" / f"seed_{stage_name(start_n)}")
        for npoints in [stage for stage in stages if stage > start_n]:
            clean_prefix(out / "tmp" / f"seed_{stage_name(start_n)}_to_{stage_name(npoints)}")

    write_csv(table_path, rows)
    return rows


def run_exp3(repo: Path, args: argparse.Namespace, assets: dict, total_n: int, start_n: int, query_count: int,
             insert_threads: list[int], query_threads: list[int]) -> list[dict]:
    out = exp_dir(args.out_dir, "exp3_search_during_insert")
    clear_experiment_dir(out)
    del query_threads
    exp3_query_count = min(query_count, 1000)
    exp3_query_bin = args.out_dir / "data" / f"sift_query_{exp3_query_count}.bin"
    copy_prefix_bin(args.query_bin, exp3_query_bin, exp3_query_count, assets["dim"])
    exp3_query_labels = ensure_query_label_files(args.out_dir / "labels", exp3_query_count)

    no_insert_n = args.exp3_no_insert_n or total_n
    if no_insert_n not in assets["base_bins"]:
        raise RuntimeError(f"exp3 no-insert size {no_insert_n} was not prepared")

    initial_no_insert = out / "tmp" / f"initial_{stage_name(no_insert_n)}"
    insert_source = out / "tmp" / f"insert_source_{stage_name(start_n)}"
    if no_insert_n == start_n:
        build_index_with_pq_bytes(repo, args, assets["base_bins"][start_n], insert_source,
                                  assets["label_files"][start_n], 32, args.exp3_pq_bytes)
        initial_no_insert = insert_source
    else:
        build_index_with_pq_bytes(repo, args, assets["base_bins"][no_insert_n], initial_no_insert,
                                  assets["label_files"][no_insert_n], 32, args.exp3_pq_bytes)
        build_index_with_pq_bytes(repo, args, assets["base_bins"][start_n], insert_source,
                                  assets["label_files"][start_n], 32, args.exp3_pq_bytes)

    rows: list[dict] = []
    calibration_cache: dict[tuple[str, str], list[tuple[str, int, dict]]] = {}

    def calibrated_candidates(prefix: Path, npoints: int, bucket: str, selectivity: float,
                              state: str) -> tuple[list[tuple[str, int, dict]], Path]:
        cache_key = (state, bucket)
        truth = out / "truth" / f"gt_{state}_{bucket}.bin"
        if not truth.exists():
            compute_truth(repo, args, assets["base_bins"][npoints], exp3_query_bin, truth,
                          assets["label_files"][npoints], exp3_query_labels[bucket])
        if cache_key not in calibration_cache:
            calibration_cache[cache_key] = calibrate_route_and_l(
                repo, args, prefix, truth, exp3_query_bin, exp3_query_labels[bucket],
                out / "calibration_route_l.jsonl", exp3_query_count, selectivity,
            )
        return calibration_cache[cache_key], truth

    for bucket, selectivity in BUCKETS:
        candidates, truth = calibrated_candidates(
            initial_no_insert, no_insert_n, bucket, selectivity, f"{stage_name(no_insert_n)}_initial_no_insert",
        )
        measured_candidates = []
        for selected_route, chosen_l, calibration_row in candidates:
            measured = driver(repo, args, mode="measure-dynamic-search", source=initial_no_insert, dest=None,
                              jsonl=out / "measure_driver.jsonl", query_bin=exp3_query_bin, truthset=truth,
                              query_label_file=exp3_query_labels[bucket], selector_type="intersect",
                              search_threads=1, search_l=chosen_l, query_limit=exp3_query_count,
                              route=selected_route)
            measured.update({
                "status": "ok" if float(measured.get("recall@10", 0.0)) >= 98.0 else "failed_recall",
                "state": f"{stage_name(no_insert_n)}_initial",
                "bucket": bucket,
                "selected_route": selected_route,
                "chosen_L": chosen_l,
                "insert_threads": 0,
                "query_threads": 1,
                "inserted_during_search": 0,
                "start_points": no_insert_n,
                "total_points": no_insert_n,
                "target_recall@10": 98.0,
                "calibration_recall@10": calibration_row.get("recall@10"),
                "calibration_avg_latency_us": calibration_row.get("avg_latency_us"),
                "exp3_pq_bytes": args.exp3_pq_bytes,
            })
            append_jsonl(out / "route_selection_candidates.jsonl", measured)
            measured_candidates.append(measured)
        passing_measured = [row for row in measured_candidates if float(row.get("recall@10", 0.0)) >= 98.0]
        measured = min(passing_measured or measured_candidates,
                       key=lambda row: float(row.get("avg_latency_us", float("inf"))))
        measured.update({
            "insert_threads": 0,
            "inserted_during_search": 0,
        })
        rows.append(measured)
        append_jsonl(out / "results.jsonl", measured)

    for ins_t in insert_threads:
        for bucket, selectivity in BUCKETS:
            candidates, truth = calibrated_candidates(
                insert_source, start_n, bucket, selectivity, f"{stage_name(start_n)}_insert_source",
            )
            measured_candidates = []
            for candidate_i, (selected_route, chosen_l, calibration_row) in enumerate(candidates):
                dest = out / "tmp" / f"runtime_i{ins_t}_{bucket}_c{candidate_i}"
                measured = driver(repo, args, mode="search-during-insert", source=insert_source, dest=dest,
                                  jsonl=out / "driver.jsonl", data_bin=assets["base_bins"][total_n],
                                  base_label_file=assets["label_files"][total_n], query_bin=exp3_query_bin,
                                  truthset=truth, query_label_file=exp3_query_labels[bucket],
                                  selector_type="intersect", insert_start=start_n,
                                  insert_count=total_n - start_n, insert_threads=ins_t, search_threads=1,
                                  merge_threads=ins_t, search_l=chosen_l, query_limit=exp3_query_count,
                                  route=selected_route)
                measured.update({
                    "status": "ok" if float(measured.get("recall@10", 0.0)) >= 98.0 else "failed_recall",
                    "state": f"during_insert_from_{stage_name(start_n)}",
                    "bucket": bucket,
                    "selected_route": selected_route,
                    "chosen_L": chosen_l,
                    "insert_threads": ins_t,
                    "query_threads": 1,
                    "start_points": start_n,
                    "total_points": total_n,
                    "target_recall@10": 98.0,
                    "calibration_recall@10": calibration_row.get("recall@10"),
                    "calibration_avg_latency_us": calibration_row.get("avg_latency_us"),
                    "exp3_pq_bytes": args.exp3_pq_bytes,
                })
                append_jsonl(out / "route_selection_candidates.jsonl", measured)
                measured_candidates.append(measured)
                clean_prefix(dest)
            passing_measured = [row for row in measured_candidates if float(row.get("recall@10", 0.0)) >= 98.0]
            measured = min(passing_measured or measured_candidates,
                           key=lambda row: float(row.get("avg_latency_us", float("inf"))))
            measured.update({
                "insert_threads": ins_t,
            })
            rows.append(measured)
            append_jsonl(out / "results.jsonl", measured)
    write_csv(out / "table.csv", rows)
    if initial_no_insert != insert_source:
        clean_prefix(initial_no_insert)
    clean_prefix(insert_source)
    return rows


def calibrate_l(repo: Path, args: argparse.Namespace, prefix: Path, truth: Path, query_bin: Path, query_label: Path,
                out_jsonl: Path, query_count: int, route: str = "auto",
                selector_type: str = "intersect") -> tuple[int, dict]:
    best_row: dict | None = None
    for l_value in EXP4_L_CANDIDATES:
        row = driver(repo, args, mode="measure-dynamic-search", source=prefix, dest=None, jsonl=out_jsonl,
                     query_bin=query_bin, truthset=truth, query_label_file=query_label,
                     selector_type=selector_type,
                     search_threads=32, search_l=l_value, query_limit=query_count, route=route)
        best_row = row
        if row["recall@10"] >= 98.0:
            return l_value, row
    return EXP4_L_CANDIDATES[-1], best_row or {}


def calibrate_route_and_l(repo: Path, args: argparse.Namespace, prefix: Path, truth: Path, query_bin: Path,
                          query_label: Path, out_jsonl: Path, query_count: int,
                          selectivity: float, selector_type: str = "intersect") -> list[tuple[str, int, dict]]:
    passing: list[tuple[str, int, dict]] = []
    best_by_recall: tuple[float, str, int, dict] | None = None

    prefilter_l, prefilter_row = calibrate_l(repo, args, prefix, truth, query_bin, query_label,
                                             out_jsonl, query_count, route="prefilter",
                                             selector_type=selector_type)
    prefilter_recall = float(prefilter_row.get("recall@10", 0.0))
    prefilter_candidates = float(prefilter_row.get("mean_candidate_count", 0.0))
    prefilter_points = float(prefilter_row.get("points", 0.0))
    if prefilter_row:
        best_by_recall = (prefilter_recall, "prefilter", prefilter_l, prefilter_row)
        if prefilter_recall >= 98.0:
            passing.append(("prefilter", prefilter_l, prefilter_row))
            if prefilter_points > 0.0 and selectivity < 0.5 and prefilter_candidates < 0.5 * prefilter_points:
                return passing

    for route in ["graph"]:
        for l_value in EXP4_L_CANDIDATES:
            try:
                row = driver(repo, args, mode="measure-dynamic-search", source=prefix, dest=None, jsonl=out_jsonl,
                             query_bin=query_bin, truthset=truth, query_label_file=query_label,
                             selector_type=selector_type,
                             search_threads=32, search_l=l_value, query_limit=query_count, route=route)
            except subprocess.CalledProcessError as error:
                append_jsonl(out_jsonl.parent / "calibration_route_l_errors.jsonl", {
                    "route": route,
                    "search_L": l_value,
                    "returncode": error.returncode,
                    "source": str(prefix),
                    "truthset": str(truth),
                    "query_label_file": str(query_label),
                })
                break
            recall = float(row.get("recall@10", 0.0))
            if best_by_recall is None or recall > best_by_recall[0]:
                best_by_recall = (recall, route, l_value, row)
            if recall >= 98.0:
                passing.append((route, l_value, row))
                break
    if passing:
        return passing
    if best_by_recall is None:
        return [("auto", EXP4_L_CANDIDATES[-1], {})]
    _, route, l_value, row = best_by_recall
    return [(route, l_value, row)]


def run_exp4(repo: Path, args: argparse.Namespace, assets: dict, total_n: int, mid_n: int, query_count: int) -> list[dict]:
    del mid_n
    out = args.out_dir / "exp4_intersect_range_selectivity"
    resume_existing = args.resume_after_exp3 and out.exists() and not args.rerun_exp4
    if out.exists() and not resume_existing:
        clear_experiment_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    exp4_query_bin = args.out_dir / "data" / f"sift_query_{query_count}.bin"
    copy_prefix_bin(args.query_bin, exp4_query_bin, query_count, assets["dim"])
    exp4_single_query_csv = read_first_query_csv(args.query_bin, assets["dim"])
    query_labels_by_selector = {
        "intersect": ensure_query_label_files(args.out_dir / "labels", query_count),
        "range": ensure_range_query_label_files(args.out_dir / "labels", query_count),
    }
    single_query_label_csv = {
        selector: {
            bucket: read_first_spmat_labels_csv(label_file)
            for bucket, label_file in label_files.items()
        }
        for selector, label_files in query_labels_by_selector.items()
    }

    prefix = out / "tmp" / "direct_1m"
    if not prefix_exists(prefix):
        build_index_with_pq_bytes(repo, args, assets["base_bins"][total_n], prefix,
                                  assets["label_files"][total_n], 32, args.exp4_pq_bytes)

    rows: list[dict] = load_jsonl(out / "results.jsonl") if resume_existing else []
    completed = {(str(row.get("selector_type")), str(row.get("bucket"))) for row in rows}
    for selector_type, query_labels in query_labels_by_selector.items():
        for bucket, selectivity in BUCKETS:
            if (selector_type, bucket) in completed:
                continue
            truth = out / "truth" / f"gt_1m_{selector_type}_{bucket}.bin"
            compute_truth(repo, args, assets["base_bins"][total_n], exp4_query_bin, truth,
                          assets["label_files"][total_n], query_labels[bucket],
                          selector_type=selector_type)
            measured_candidates = []
            # Low-selectivity filters should use prefilter directly. Sweeping graph there is
            # both slow and uninformative because graph must find enough filtered hits from a
            # tiny live candidate set.
            route_candidates = ["prefilter", "graph"] if selectivity >= 0.25 else ["prefilter"]
            for selected_route in route_candidates:
                try:
                    chosen_l, calibration_row = calibrate_baseline_l(
                        repo, args, prefix, truth, exp4_query_bin, query_labels[bucket],
                        selected_route, out / "calibration_route_l.jsonl",
                        selector_type=selector_type,
                    )
                except subprocess.CalledProcessError as error:
                    append_jsonl(out / "calibration_route_l_errors.jsonl", {
                        "selector_type": selector_type,
                        "bucket": bucket,
                        "route": selected_route,
                        "returncode": error.returncode,
                        "source": str(prefix),
                        "truthset": str(truth),
                        "query_label_file": str(query_labels[bucket]),
                    })
                    continue
                if chosen_l is None:
                    append_jsonl(out / "skipped.jsonl", {
                        "status": "skipped_latency_cutoff",
                        "selector_type": selector_type,
                        "bucket": bucket,
                        "route": selected_route,
                        "last_L": calibration_row.get("chosen_L"),
                        "last_recall@10": calibration_row.get("recall@10"),
                        "last_avg_latency_us": calibration_row.get("avg_latency_us"),
                    })
                    continue
                measured = static_hybrid_search(
                    repo, args, prefix=prefix, jsonl=out / "measure_driver.jsonl",
                    query_bin=exp4_query_bin, truthset=truth, query_label_file=query_labels[bucket],
                    route=selected_route, threads=1, search_l=chosen_l,
                    selector_type=selector_type,
                )
                status = "ok" if float(measured.get("recall@10", 0.0)) >= 98.0 else "failed_recall"
                measured.update({
                    "status": status,
                    "state": "1m_direct",
                    "bucket": bucket,
                    "selector_type": selector_type,
                    "filter_type": selector_type,
                    "chosen_L": chosen_l,
                    "points": total_n,
                    "selected_route": selected_route,
                    "target_recall@10": 98.0,
                    "exp4_pq_bytes": args.exp4_pq_bytes,
                    "calibration_recall@10": calibration_row.get("recall@10"),
                    "calibration_avg_latency_us": calibration_row.get("avg_latency_us"),
                })
                append_jsonl(out / "route_selection_candidates.jsonl", measured)
                measured_candidates.append(measured)
            if not measured_candidates:
                continue
            passing_measured = [row for row in measured_candidates if float(row.get("recall@10", 0.0)) >= 98.0]
            measured = min(passing_measured or measured_candidates,
                           key=lambda row: float(row.get("avg_latency_us", float("inf"))))
            rss_row = static_hybrid_search(
                repo, args, prefix=prefix, jsonl=out / "rss_single_query_driver.jsonl",
                query_bin=None, truthset=None, query_label_file=None,
                route=str(measured["selected_route"]), threads=1, search_l=int(measured["chosen_L"]),
                selector_type=selector_type, query_vector_csv=exp4_single_query_csv,
                query_label_csv=single_query_label_csv[selector_type][bucket],
                single_query_static_rss=True,
            )
            measured["search_max_rss_kb"] = measured.get("max_rss_kb")
            measured["process_max_rss_kb"] = rss_row.get("process_max_rss_kb", rss_row.get("max_rss_kb"))
            measured["rss_before_query_kb"] = rss_row.get("rss_before_query_kb")
            measured["rss_after_query_kb"] = rss_row.get("rss_after_query_kb")
            measured["query_peak_rss_kb"] = rss_row.get("query_peak_rss_kb")
            measured["query_peak_delta_kb"] = rss_row.get("query_peak_delta_kb")
            measured["max_rss_kb"] = measured["process_max_rss_kb"]
            measured["rss_mode"] = "single_query_process_no_query_files"
            measured["rss_single_query_kb"] = measured["max_rss_kb"]
            measured["rss_single_query_avg_latency_us"] = rss_row.get("avg_latency_us")
            measured["rss_single_query_recall@10"] = rss_row.get("recall@10")
            rows.append(measured)
            completed.add((selector_type, bucket))
            append_jsonl(out / "results.jsonl", measured)

    write_csv(out / "table.csv", rows)
    clean_prefix(prefix)
    return rows


def run_exp5(repo: Path, args: argparse.Namespace, assets: dict, stages: list[int], dim: int) -> list[dict]:
    out = exp_dir(args.out_dir, "exp5_index_bloat_by_size")
    clear_experiment_dir(out)
    rows: list[dict] = []
    suffixes = ["_disk.index", "_disk.index.tags", "_pq_compressed.bin", "_pq_pivots.bin", "_labels.densebit", "_hybrid.meta"]
    for npoints in stages:
        prefix = out / "tmp" / f"bloat_{stage_name(npoints)}"
        build_index(repo, args, assets["base_bins"][npoints], prefix, assets["label_files"][npoints], 32)
        sizes = {suffix: Path(str(prefix) + suffix).stat().st_size if Path(str(prefix) + suffix).exists() else 0 for suffix in suffixes}
        total = sum(sizes.values())
        raw = npoints * dim * 4
        row = {"status": "ok", "points": npoints, "raw_vector_bytes": raw, "total_index_bytes": total,
               "extra_over_raw_ratio": (total - raw) / raw, "total_to_raw_ratio": total / raw, **sizes}
        rows.append(row)
        append_jsonl(out / "results.jsonl", row)
        clean_prefix(prefix)
    write_csv(out / "table.csv", rows)
    return rows


def calibrate_baseline_l(repo: Path, args: argparse.Namespace, prefix: Path, truth: Path, query_bin: Path,
                         query_label: Path, route: str, out_jsonl: Path,
                         selector_type: str = "intersect") -> tuple[int | None, dict]:
    best_row: dict = {}
    for l_value in EXP4_L_CANDIDATES:
        row = static_hybrid_search(repo, args, prefix=prefix, jsonl=out_jsonl, query_bin=query_bin,
                                   truthset=truth, query_label_file=query_label, route=route,
                                   threads=1, search_l=l_value, selector_type=selector_type)
        best_row = row
        recall = float(row.get("recall@10", 0.0))
        avg_latency_us = float(row.get("avg_latency_us", 0.0))
        if recall >= 98.0:
            return l_value, row
        if avg_latency_us > BASELINE_LATENCY_CUTOFF_US:
            break
    return None, best_row


def run_exp_baseline(repo: Path, args: argparse.Namespace, assets: dict, total_n: int) -> list[dict]:
    out = args.out_dir / "exp_baseline"
    if out.exists() and args.rerun_baseline:
        clear_experiment_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    existing = load_jsonl(out / "results.jsonl")
    if existing and not args.rerun_baseline:
        write_csv(out / "table.csv", existing)
        return existing

    for stale in [out / "results.jsonl", out / "table.csv", out / "calibration.jsonl",
                  out / "measure_driver.jsonl", out / "skipped.jsonl"]:
        if stale.exists():
            stale.unlink()

    query_count = max(1, min(args.baseline_query_count, read_bin_header(args.query_bin)[0]))
    query_bin = args.out_dir / "data" / f"sift_query_baseline_{query_count}.bin"
    copy_prefix_bin(args.query_bin, query_bin, query_count, assets["dim"])
    query_labels = ensure_query_label_files(args.out_dir / "labels", query_count)

    prefix = out / "tmp" / "direct_1m"
    if not prefix_exists(prefix):
        build_index_with_pq_bytes(repo, args, assets["base_bins"][total_n], prefix, assets["label_files"][total_n],
                                  32, args.baseline_pq_bytes)

    rows: list[dict] = []
    for bucket, _selectivity in BUCKETS:
        truth = out / "truth" / f"gt_1m_{bucket}.bin"
        compute_truth(repo, args, assets["base_bins"][total_n], query_bin, truth,
                      assets["label_files"][total_n], query_labels[bucket])
        for route in BASELINE_ROUTES:
            chosen_l, calibration_row = calibrate_baseline_l(
                repo, args, prefix, truth, query_bin, query_labels[bucket], route,
                out / "calibration.jsonl",
            )
            if chosen_l is None:
                skipped = {
                    "status": "skipped_latency_cutoff",
                    "points": total_n,
                    "query_count": query_count,
                    "bucket": bucket,
                    "route": route,
                    "baseline_pq_bytes": args.baseline_pq_bytes,
                    "target_recall@10": 98.0,
                    "latency_cutoff_ms": BASELINE_LATENCY_CUTOFF_US / 1000.0,
                    "last_L": calibration_row.get("chosen_L"),
                    "last_recall@10": calibration_row.get("recall@10"),
                    "last_avg_latency_us": calibration_row.get("avg_latency_us"),
                }
                append_jsonl(out / "skipped.jsonl", skipped)
                continue
            for threads in BASELINE_THREADS:
                measured = static_hybrid_search(repo, args, prefix=prefix, jsonl=out / "measure_driver.jsonl",
                                                query_bin=query_bin, truthset=truth,
                                                query_label_file=query_labels[bucket], route=route,
                                                threads=threads, search_l=chosen_l)
                status = "ok" if float(measured.get("recall@10", 0.0)) >= 98.0 else "failed_recall"
                measured.update({
                    "status": status,
                    "points": total_n,
                    "query_count": query_count,
                    "bucket": bucket,
                    "route": route,
                    "threads": threads,
                    "baseline_pq_bytes": args.baseline_pq_bytes,
                    "chosen_L": chosen_l,
                    "target_recall@10": 98.0,
                    "latency_cutoff_ms": BASELINE_LATENCY_CUTOFF_US / 1000.0,
                    "calibration_recall@10": calibration_row.get("recall@10"),
                    "calibration_avg_latency_us": calibration_row.get("avg_latency_us"),
                })
                rows.append(measured)
                append_jsonl(out / "results.jsonl", measured)

    write_csv(out / "table.csv", rows)
    clean_prefix(prefix)
    return rows


def finish_run(repo: Path, args: argparse.Namespace, summary: dict) -> None:
    summary["status"] = "ok"
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run([sys.executable, "scripts/plot_codex_dynamic_update_suite.py", "--out-dir", str(args.out_dir)], repo)
    clean_large_files(args.out_dir)


def main() -> int:
    configure_stdio()
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out_dir = args.out_dir.resolve()
    args.base_bin = args.base_bin.resolve()
    args.query_bin = args.query_bin.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        build_tools(repo)

    if args.smoke:
        stages = [10_000, 20_000, 30_000]
        total_n, mid_n, seed_n, start_n, query_count = 30_000, 20_000, 1_000, 20_000, 100
        exp1_threads, exp3_insert_threads, exp3_query_threads = [2, 1], [1], [2, 1]
        args.memory_gb = min(args.memory_gb, 8)
    else:
        stages = [250_000, 500_000, 750_000, 1_000_000]
        total_n, mid_n, seed_n, start_n, query_count = 1_000_000, 750_000, 10_000, 500_000, 10_000
        exp1_threads, exp3_insert_threads, exp3_query_threads = [32, 16, 8, 4], [4, 2, 1], [32, 16, 8, 4, 2, 1]

    if args.only_exp3 and args.exp3_total_n:
        total_n = args.exp3_total_n
        start_n = args.exp3_start_n or total_n // 2
        no_insert_n = args.exp3_no_insert_n or start_n
        stages = sorted({start_n, total_n, no_insert_n})
        seed_n = min(seed_n, start_n)
        if seed_n not in stages:
            stages.append(seed_n)
            stages = sorted(set(stages))

    assets = prepare_assets(args.out_dir, args, stages, seed_n, query_count)
    summary = {
        "status": "running", "smoke": args.smoke, "stages": stages, "seed_n": seed_n,
        "query_count": query_count, "experiments": {},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.only_exp1:
        summary["experiments"]["exp1_insert_vs_build_threads"] = run_exp1(
            repo, args, assets, total_n, seed_n, exp1_threads,
        )
        finish_run(repo, args, summary)
        return 0

    if args.only_exp2:
        summary["experiments"]["exp2_stage_recall_build_vs_insert"] = run_exp2(
            repo, args, assets, stages, seed_n, query_count,
        )
        summary["experiments"]["exp2_seed_sweep"] = run_exp2_seed_sweep(
            repo, args, assets, stages, parse_int_list(args.exp2_seed_sweep_starts), query_count,
        )
        finish_run(repo, args, summary)
        return 0

    if args.only_exp3:
        summary["experiments"]["exp3_search_during_insert"] = run_exp3(
            repo, args, assets, total_n, start_n, query_count, exp3_insert_threads, exp3_query_threads,
        )
        finish_run(repo, args, summary)
        return 0

    if args.only_exp4:
        summary["experiments"]["exp4_intersect_range_selectivity"] = run_exp4(
            repo, args, assets, total_n, mid_n, min(query_count, 1000),
        )
        finish_run(repo, args, summary)
        return 0

    if args.only_exp5:
        summary["experiments"]["exp5_index_bloat_by_size"] = run_exp5(repo, args, assets, stages, assets["dim"])
        finish_run(repo, args, summary)
        return 0

    if args.only_baseline:
        summary["experiments"]["exp_baseline"] = run_exp_baseline(repo, args, assets, total_n)
        finish_run(repo, args, summary)
        return 0

    if args.only_exp2_seed_sweep:
        start_points = parse_int_list(args.exp2_seed_sweep_starts)
        summary["experiments"]["exp2_seed_sweep"] = run_exp2_seed_sweep(
            repo, args, assets, stages, start_points, query_count,
        )
        finish_run(repo, args, summary)
        return 0

    if args.resume_after_exp1 or args.resume_after_exp2 or args.resume_after_exp3:
        exp1_rows = load_jsonl(args.out_dir / "exp1_insert_vs_build_threads/results.jsonl")
        if not exp1_rows:
            raise RuntimeError("resume was requested, but exp1 results.jsonl is empty")
        summary["experiments"]["exp1_insert_vs_build_threads"] = exp1_rows
        write_csv(args.out_dir / "exp1_insert_vs_build_threads/table.csv", exp1_rows)
    else:
        summary["experiments"]["exp1_insert_vs_build_threads"] = run_exp1(repo, args, assets, total_n, seed_n, exp1_threads)
    if args.resume_after_exp2 or args.resume_after_exp3:
        exp2_rows = load_jsonl(args.out_dir / "exp2_stage_recall_build_vs_insert/results.jsonl")
        if not exp2_rows:
            raise RuntimeError("resume after exp2/exp3 was requested, but exp2 results.jsonl is empty")
        summary["experiments"]["exp2_stage_recall_build_vs_insert"] = exp2_rows
        write_csv(args.out_dir / "exp2_stage_recall_build_vs_insert/table.csv", exp2_rows)
    else:
        summary["experiments"]["exp2_stage_recall_build_vs_insert"] = run_exp2(repo, args, assets, stages, seed_n, query_count)
    summary["experiments"]["exp2_seed_sweep"] = run_exp2_seed_sweep(
        repo, args, assets, stages, parse_int_list(args.exp2_seed_sweep_starts), query_count,
    )
    if args.resume_after_exp3:
        exp3_rows = load_jsonl(args.out_dir / "exp3_search_during_insert/results.jsonl")
        if not exp3_rows:
            raise RuntimeError("--resume-after-exp3 was requested, but exp3 results.jsonl is empty")
        summary["experiments"]["exp3_search_during_insert"] = exp3_rows
        write_csv(args.out_dir / "exp3_search_during_insert/table.csv", exp3_rows)
    else:
        summary["experiments"]["exp3_search_during_insert"] = run_exp3(repo, args, assets, total_n, start_n, query_count,
                                                                         exp3_insert_threads, exp3_query_threads)
    summary["experiments"]["exp4_intersect_range_selectivity"] = run_exp4(repo, args, assets, total_n, mid_n, min(query_count, 1000))
    if args.stop_after_exp4:
        existing_exp5 = load_jsonl(args.out_dir / "exp5_index_bloat_by_size/results.jsonl")
        if existing_exp5:
            summary["experiments"]["exp5_index_bloat_by_size"] = existing_exp5
        finish_run(repo, args, summary)
        return 0
    summary["experiments"]["exp5_index_bloat_by_size"] = run_exp5(repo, args, assets, stages, assets["dim"])
    finish_run(repo, args, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
