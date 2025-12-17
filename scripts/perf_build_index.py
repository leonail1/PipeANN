#!/usr/bin/env python3

import os
import sys
import subprocess
import glob
import json
import time
from pathlib import Path
from datetime import datetime


BASIC_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
]

DETAILED_EVENTS = [
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L1-dcache-stores",
    "L1-icache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
    "LLC-stores",
    "LLC-store-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "dTLB-stores",
    "dTLB-store-misses",
    "iTLB-loads",
    "iTLB-load-misses",
]

MEMORY_EVENTS = [
    "mem-loads",
    "mem-stores",
]


def delete_index_files(index_prefix: str, verbose: bool = True) -> bool:
    deleted_any = False
    patterns = [
        f"{index_prefix}_disk.index*",
        f"{index_prefix}_mem.index*",
        f"{index_prefix}_pq*",
        f"{index_prefix}_SAMPLE_RATE_*",
        f"{index_prefix}*.bin",
        f"{index_prefix}*.txt",
    ]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            if os.path.exists(filepath):
                if verbose:
                    print(f"  删除: {filepath}")
                os.remove(filepath)
                deleted_any = True
    return deleted_any


def get_perf_stat_cmd(events: list, output_file: str = None) -> list:
    cmd = ["perf", "stat", "-e", ",".join(events)]
    if output_file:
        cmd.extend(["-o", output_file])
    cmd.append("--")
    return cmd


def get_perf_record_cmd(output_file: str, frequency: int = 99) -> list:
    return ["perf", "record", "-g", "-F", str(frequency), "-o", output_file, "--"]


def run_with_perf_stat(cmd: list, events: list, output_file: str,
                       description: str, verbose: bool = True) -> bool:
    if verbose:
        print(f"\n{'='*60}")
        print(f"[perf stat] {description}")
        print(f"事件: {', '.join(events[:5])}{'...' if len(events) > 5 else ''}")
        print(f"{'='*60}")

    perf_cmd = get_perf_stat_cmd(events, output_file) + cmd

    try:
        start_time = time.time()
        result = subprocess.run(perf_cmd, check=True)
        elapsed = time.time() - start_time
        if verbose:
            print(f"✓ 完成 (耗时: {elapsed:.2f}s)")
            print(f"  结果保存至: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        return False


def run_with_perf_record(cmd: list, output_file: str,
                         description: str, verbose: bool = True) -> bool:
    if verbose:
        print(f"\n{'='*60}")
        print(f"[perf record] {description}")
        print(f"{'='*60}")

    perf_cmd = get_perf_record_cmd(output_file) + cmd

    try:
        start_time = time.time()
        result = subprocess.run(perf_cmd, check=True)
        elapsed = time.time() - start_time
        if verbose:
            print(f"✓ 完成 (耗时: {elapsed:.2f}s)")
            print(f"  perf.data 保存至: {output_file}")
            print(f"  使用 'perf report -i {output_file}' 查看热点函数")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        return False


def build_disk_index_cmd(data_dir: str, index_prefix: str, project_root: Path,
                         num_threads: int = 64, R: int = 64, L_build: int = 96,
                         pq_bytes: int = 32, memory_gb: int = 32, metric: str = "l2",
                         nbr_type: str = "pq", data_type: str = "uint8",
                         with_labels: bool = True, data_file: str = None,
                         label_file: str = None) -> list:
    if data_file is None:
        data_file = f"{data_dir}/bigann_1m.bin"
    if label_file is None:
        label_file = f"{data_dir}/data_labels.spmat"

    cmd = [
        str(project_root / "build/tests/build_disk_index"), data_type,
        data_file, index_prefix,
        str(R), str(L_build), str(pq_bytes), str(memory_gb), str(num_threads),
        metric, nbr_type
    ]
    if with_labels and os.path.exists(label_file):
        cmd.extend(["spmat", label_file])
    return cmd


def gen_random_slice_cmd(data_dir: str, index_prefix: str, project_root: Path,
                         data_file: str = None) -> list:
    if data_file is None:
        data_file = f"{data_dir}/bigann_1m.bin"
    return [
        str(project_root / "build/tests/utils/gen_random_slice"), "uint8",
        data_file, f"{index_prefix}_SAMPLE_RATE_0.01", "0.01"
    ]


def build_memory_index_cmd(index_prefix: str, project_root: Path,
                           num_threads: int = 64, metric: str = "l2") -> list:
    return [
        str(project_root / "build/tests/build_memory_index"), "uint8",
        f"{index_prefix}_SAMPLE_RATE_0.01_data.bin",
        f"{index_prefix}_SAMPLE_RATE_0.01_ids.bin",
        f"{index_prefix}_mem.index",
        "32", "64", "1.2", str(num_threads), metric
    ]


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    DATA_DIR = os.getenv("SIFT1M_DATA_DIR", "/data/lzg/sift-pipeann/sift1m_pq")
    INDEX_DIR = f"{DATA_DIR}/indices"
    INDEX_PREFIX = f"{INDEX_DIR}/sift1m_filtered"

    PERF_OUTPUT_DIR = SCRIPT_DIR / "perf_results"

    R = 64
    L = 96
    pq_bytes = 32
    memory_gb = 32
    threads = 64
    metric = "l2"
    nbr_type = "pq"
    data_type = "uint8"
    with_labels = True
    data_file = None
    label_file = None

    do_detailed = True
    do_record = True
    do_memory = True
    force = True
    verbose = True

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(PERF_OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if verbose:
        print("=" * 60)
        print("PipeANN 索引构建 - Perf性能统计")
        print("=" * 60)
        print(f"数据目录: {DATA_DIR}")
        print(f"索引前缀: {INDEX_PREFIX}")
        print(f"Perf输出: {PERF_OUTPUT_DIR}")
        print(f"时间戳: {timestamp}")
        print()
        print("统计项目:")
        print(f"  - 基本统计: 是")
        print(f"  - 详细缓存/TLB: {'是' if do_detailed else '否'}")
        print(f"  - 热点函数记录: {'是' if do_record else '否'}")
        print(f"  - 内存访问统计: {'是' if do_memory else '否'}")
        print()

    REQUIRED_EXECUTABLES = [
        "build/tests/build_disk_index",
        "build/tests/build_memory_index",
        "build/tests/utils/gen_random_slice"
    ]
    for exe in REQUIRED_EXECUTABLES:
        if not os.path.exists(PROJECT_ROOT / exe):
            print(f"✗ 可执行文件不存在: {exe}")
            print("  请先编译项目")
            sys.exit(1)

    actual_data_file = data_file or f"{DATA_DIR}/bigann_1m.bin"
    if not os.path.exists(actual_data_file):
        print(f"✗ 数据文件不存在: {actual_data_file}")
        sys.exit(1)

    if force:
        if verbose:
            print("[准备] 删除现有索引...")
        delete_index_files(INDEX_PREFIX, verbose)

    results_summary = {
        "timestamp": timestamp,
        "config": {
            "data_dir": DATA_DIR,
            "R": R,
            "L": L,
            "pq_bytes": pq_bytes,
            "threads": threads,
        },
        "stages": {}
    }

    stages = [
        ("build_disk_index", "构建磁盘索引",
         build_disk_index_cmd(DATA_DIR, INDEX_PREFIX, PROJECT_ROOT,
                              num_threads=threads, R=R, L_build=L,
                              pq_bytes=pq_bytes, memory_gb=memory_gb,
                              metric=metric, nbr_type=nbr_type,
                              data_type=data_type, with_labels=with_labels,
                              data_file=data_file, label_file=label_file)),
    ]

    for stage_name, stage_desc, cmd in stages:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# 阶段: {stage_desc}")
            print(f"# 命令: {' '.join(cmd[:3])}...")
            print(f"{'#'*60}")

        stage_results = {"outputs": []}

        output_file = f"{PERF_OUTPUT_DIR}/{timestamp}_{stage_name}_basic.txt"
        if run_with_perf_stat(cmd, BASIC_EVENTS, output_file, f"{stage_desc} - 基本统计", verbose):
            stage_results["outputs"].append(("basic", output_file))

        if force:
            delete_index_files(INDEX_PREFIX, verbose=False)

        if do_detailed:
            output_file = f"{PERF_OUTPUT_DIR}/{timestamp}_{stage_name}_detailed.txt"
            if run_with_perf_stat(cmd, DETAILED_EVENTS, output_file, f"{stage_desc} - 详细统计", verbose):
                stage_results["outputs"].append(("detailed", output_file))
            if force:
                delete_index_files(INDEX_PREFIX, verbose=False)

        if do_memory:
            output_file = f"{PERF_OUTPUT_DIR}/{timestamp}_{stage_name}_memory.txt"
            if run_with_perf_stat(cmd, MEMORY_EVENTS, output_file, f"{stage_desc} - 内存统计", verbose):
                stage_results["outputs"].append(("memory", output_file))
            if force:
                delete_index_files(INDEX_PREFIX, verbose=False)

        if do_record:
            output_file = f"{PERF_OUTPUT_DIR}/{timestamp}_{stage_name}.perf.data"
            if run_with_perf_record(cmd, output_file, f"{stage_desc} - 热点函数", verbose):
                stage_results["outputs"].append(("record", output_file))

        results_summary["stages"][stage_name] = stage_results

    summary_file = f"{PERF_OUTPUT_DIR}/{timestamp}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("统计完成!")
        print(f"{'='*60}")
        print(f"结果目录: {PERF_OUTPUT_DIR}")
        print(f"摘要文件: {summary_file}")
        print()
        print("查看结果:")
        print(f"  cat {PERF_OUTPUT_DIR}/{timestamp}_*_basic.txt")
        if do_record:
            print(f"  perf report -i {PERF_OUTPUT_DIR}/{timestamp}_build_disk_index.perf.data")

    return 0


if __name__ == "__main__":
    sys.exit(main())