#!/usr/bin/env python3
"""
删除现有索引并重新构建索引的脚本

用法:
    python3 rebuild_index.py                    # 如果索引不存在则构建（带标签）
    python3 rebuild_index.py --force            # 强制删除并重建索引
    python3 rebuild_index.py --no-labels        # 构建不带标签的索引
    python3 rebuild_index.py --delete-only      # 仅删除索引，不重建
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path


def delete_index_files(index_prefix: str, verbose: bool = True) -> bool:
    """
    删除索引相关的所有文件
    
    Args:
        index_prefix: 索引文件前缀
        verbose: 是否输出详细信息
    
    Returns:
        True if any files were deleted
    """
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


def build_index(data_dir: str, index_prefix: str, project_root: Path,
                num_threads: int = 64, R: int = 64, L_build: int = 96,
                pq_bytes: int = 32, memory_gb: int = 32, metric: str = "l2",
                nbr_type: str = "pq", data_type: str = "uint8",
                with_labels: bool = True, data_file: str = None,
                label_file: str = None, verbose: bool = True) -> bool:
    """
    构建磁盘索引和内存索引
    
    Args:
        data_dir: 数据目录
        index_prefix: 索引前缀路径
        project_root: 项目根目录
        num_threads: 线程数
        R: 图度数
        L_build: 构建时搜索列表大小
        pq_bytes: PQ压缩字节数
        memory_gb: 内存限制(GB)
        metric: 距离度量 (l2/cosine/mips)
        nbr_type: 邻居类型 (pq/rabitq)
        data_type: 数据类型 (float/int8/uint8)
        with_labels: 是否构建带标签的索引
        data_file: 数据文件路径（如果为None则使用默认路径）
        label_file: 标签文件路径（如果为None则使用默认路径）
        verbose: 是否输出详细信息
    
    Returns:
        True if build succeeded
    """
    if data_file is None:
        data_file = f"{data_dir}/bigann_1m.bin"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    if with_labels:
        if label_file is None:
            label_file = f"{data_dir}/data_labels.spmat"
        if not os.path.exists(label_file):
            print(f"❌ 标签文件不存在: {label_file}")
            return False
    
    index_dir = os.path.dirname(index_prefix)
    os.makedirs(index_dir, exist_ok=True)
    
    if verbose:
        label_info = "带标签" if with_labels else "无标签"
        print(f"  构建磁盘索引 ({label_info}, R={R}, L={L_build}, PQ={pq_bytes}B)...")
    
    cmd = [
        str(project_root / "build/tests/build_disk_index"), data_type,
        data_file, index_prefix,
        str(R), str(L_build), str(pq_bytes), str(memory_gb), str(num_threads),
        metric, nbr_type
    ]
    
    if with_labels:
        cmd.extend(["spmat", label_file])
    
    try:
        subprocess.run(cmd,
           stdout=subprocess.DEVNULL if not verbose else None,
           stderr=subprocess.DEVNULL if not verbose else None,
           check=True)
        if verbose:
            print("  ✓ 磁盘索引构建完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 磁盘索引构建失败: {e}")
        return False
    
    if verbose:
        print("  构建内存索引...")
    
    try:
        subprocess.run([
            str(project_root / "build/tests/utils/gen_random_slice"), "uint8",
            data_file, f"{index_prefix}_SAMPLE_RATE_0.01", "0.01"
        ], stdout=subprocess.DEVNULL if not verbose else None,
           stderr=subprocess.DEVNULL if not verbose else None,
           check=True)
        
        subprocess.run([
            str(project_root / "build/tests/build_memory_index"), "uint8",
            f"{index_prefix}_SAMPLE_RATE_0.01_data.bin",
            f"{index_prefix}_SAMPLE_RATE_0.01_ids.bin",
            f"{index_prefix}_mem.index",
            "32", "64", "1.2", str(num_threads), metric
        ], stdout=subprocess.DEVNULL if not verbose else None,
           stderr=subprocess.DEVNULL if not verbose else None,
           check=True)
        if verbose:
            print("  ✓ 内存索引构建完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 内存索引构建失败: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='删除并重建SIFT1M过滤搜索索引')
    parser.add_argument('--force', action='store_true',
                        help='强制删除并重建索引（即使已存在）')
    parser.add_argument('--delete-only', action='store_true',
                        help='仅删除索引，不重建')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，减少输出')
    parser.add_argument('--data-dir', type=str,
                        default=os.getenv("SIFT1M_DATA_DIR", "/data/lzg/sift-pipeann/sift1m_pq"),
                        help='数据目录路径')
    parser.add_argument('--with-labels', dest='with_labels', action='store_true',
                        default=True, help='构建带标签的索引（默认）')
    parser.add_argument('--no-labels', dest='with_labels', action='store_false',
                        help='构建不带标签的索引')
    parser.add_argument('--data-type', type=str, default='uint8',
                        choices=['float', 'int8', 'uint8'],
                        help='数据类型 (float/int8/uint8)')
    parser.add_argument('--data-file', type=str, default=None,
                        help='数据文件路径（默认使用 data-dir/bigann_1m.bin）')
    parser.add_argument('--label-file', type=str, default=None,
                        help='标签文件路径（默认使用 data-dir/data_labels.spmat）')
    parser.add_argument('--metric', type=str, default='l2',
                        choices=['l2', 'cosine', 'mips'],
                        help='距离度量 (l2/cosine/mips)')
    parser.add_argument('--nbr-type', type=str, default='pq',
                        choices=['pq', 'rabitq'],
                        help='邻居类型 (pq/rabitq)')
    parser.add_argument('-R', type=int, default=64, help='图度数')
    parser.add_argument('-L', type=int, default=96, help='构建时搜索列表大小')
    parser.add_argument('--pq-bytes', type=int, default=32, help='PQ压缩字节数')
    parser.add_argument('--memory-gb', type=int, default=32, help='内存限制(GB)')
    parser.add_argument('--threads', type=int, default=64, help='线程数')
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    DATA_DIR = args.data_dir
    INDEX_DIR = f"{DATA_DIR}/indices"
    INDEX_PREFIX = f"{INDEX_DIR}/sift1m_filtered"
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    if verbose:
        print("========================================")
        print("SIFT1M 索引管理工具")
        print("========================================")
        print(f"数据目录: {DATA_DIR}")
        print(f"索引前缀: {INDEX_PREFIX}")
        print()
    
    REQUIRED_EXECUTABLES = [
        "build/tests/build_disk_index",
        "build/tests/build_memory_index",
        "build/tests/utils/gen_random_slice"
    ]
    
    if not args.delete_only:
        for exe in REQUIRED_EXECUTABLES:
            if not os.path.exists(PROJECT_ROOT / exe):
                print(f"❌ 可执行文件不存在: {exe}")
                print("   请先编译项目")
                sys.exit(1)
    
    index_exists = os.path.exists(f"{INDEX_PREFIX}_disk.index")
    
    if args.force or args.delete_only:
        if verbose:
            print("[1] 删除现有索引...")
        if delete_index_files(INDEX_PREFIX, verbose):
            if verbose:
                print("  ✓ 索引删除完成")
        else:
            if verbose:
                print("  ✓ 没有找到需要删除的索引文件")
        
        if args.delete_only:
            if verbose:
                print()
                print("索引删除完成（--delete-only 模式）")
            return 0
    elif index_exists:
        if verbose:
            print("索引已存在，跳过构建")
            print("使用 --force 强制重建")
        return 0
    
    if verbose:
        step = "[2]" if args.force else "[1]"
        print()
        print(f"{step} 构建索引...")
    if not build_index(DATA_DIR, INDEX_PREFIX, PROJECT_ROOT,
                       num_threads=args.threads, R=args.R, L_build=args.L,
                       pq_bytes=args.pq_bytes, memory_gb=args.memory_gb,
                       metric=args.metric, nbr_type=args.nbr_type,
                       data_type=args.data_type, with_labels=args.with_labels,
                       data_file=args.data_file, label_file=args.label_file,
                       verbose=verbose):
        return 1
    
    if verbose:
        print()
        print("========================================")
        print("索引构建完成！")
        print("========================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())