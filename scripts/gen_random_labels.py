#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 PipeANN 生成标签文件和计算 Ground Truth，支持 filtered search 功能

支持的命令：
  data-labels    - 生成数据标签（spmat格式）
  query-labels   - 生成查询标签（单个标签）
  compute-gt     - 计算单个标签的 Ground Truth
  compute-gt-all - 批量计算所有标签的 Ground Truth
"""

import struct
import os
import sys
import subprocess
import argparse
import numpy as np
from typing import Dict
from pathlib import Path


def generate_spmat_labels_with_distribution(output_file: str, num_vectors: int, 
                                             label_percentages: Dict[int, float], seed: int = 42):
    """
    生成稀疏矩阵标签（Spmat Labels），可指定每个标签在向量中的占比

    Spmat 格式用于表示每个向量的标签集合，使用 CSR 稀疏矩阵格式存储。
    格式说明：
      - 矩阵维度：nrow x ncol，其中 nrow = 向量数，ncol = 标签空间大小
      - matrix[i][j] != 0 表示向量 i 包含标签 j

    用途：
      - LabelIntersectionSelector: 查询标签集合与数据标签集合有交集
      - LabelSubsetSelector: 查询标签集合是数据标签集合的子集

    Args:
        output_file: 输出文件路径（.spmat 格式）
        num_vectors: 向量数量（矩阵行数）
        label_percentages: 字典，键为标签ID（0-based），值为该标签的占比（0-100）
        seed: 随机种子
    """
    np.random.seed(seed)
    
    num_labels = max(label_percentages.keys()) + 1 if label_percentages else 0
    
    vector_labels = [set() for _ in range(num_vectors)]
    
    for label_id, percentage in label_percentages.items():
        num_vectors_with_label = int(num_vectors * percentage / 100.0)
        chosen_vectors = np.random.choice(num_vectors, size=num_vectors_with_label, replace=False)
        for vec_id in chosen_vectors:
            vector_labels[vec_id].add(label_id)
    
    indptr = [0]
    indices = []
    data = []
    
    for labels in vector_labels:
        sorted_labels = sorted(labels)
        for label_id in sorted_labels:
            indices.append(label_id)
            data.append(1.0)
        indptr.append(len(indices))
    
    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)
    
    with open(output_file, 'wb') as f:
        nrow = num_vectors
        ncol = num_labels
        nnz = len(indices)
        
        f.write(struct.pack('<q', nrow))
        f.write(struct.pack('<q', ncol))
        f.write(struct.pack('<q', nnz))
        f.write(indptr.tobytes())
        f.write(indices.tobytes())
        f.write(data.tobytes())
    
    avg_labels = nnz / num_vectors if num_vectors > 0 else 0
    
    print(f"✓ 成功生成稀疏矩阵标签")
    print(f"  - 向量数量 (nrow): {nrow:,}")
    print(f"  - 标签空间 (ncol): {ncol:,}")
    print(f"  - 非零元素 (nnz): {nnz:,}")
    print(f"  - 平均标签数: {avg_labels:.2f}")
    print(f"  - 输出文件: {output_file}")
    
    file_size = (3 * 8 + len(indptr) * 8 + len(indices) * 4 + len(data) * 4) / 1024 / 1024
    print(f"  - 文件大小: {file_size:.2f} MB")


def generate_query_spmat_single_label(output_file: str, num_queries: int, 
                                       num_labels: int, label_id: int):
    """
    为所有查询生成单个标签的 spmat 文件

    Args:
        output_file: 输出文件路径
        num_queries: 查询数量
        num_labels: 标签空间大小
        label_id: 要生成的标签ID（0-based）
    """
    indptr = [0]
    indices = []
    data = []
    
    for _ in range(num_queries):
        indices.append(label_id)
        data.append(1.0)
        indptr.append(len(indices))
    
    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)
    
    with open(output_file, 'wb') as f:
        nrow = num_queries
        ncol = num_labels
        nnz = len(indices)
        
        f.write(struct.pack('<q', nrow))
        f.write(struct.pack('<q', ncol))
        f.write(struct.pack('<q', nnz))
        f.write(indptr.tobytes())
        f.write(indices.tobytes())
        f.write(data.tobytes())
    
    print(f"✓ 成功生成查询标签")
    print(f"  - 查询数量: {num_queries:,}")
    print(f"  - 标签空间: {num_labels:,}")
    print(f"  - 标签ID: {label_id}")
    print(f"  - 输出文件: {output_file}")


def compute_gt_for_label(label_id: int, data_dir: str, project_root: Path, 
                         k: int = 10, metric: str = "l2", num_queries: int = 10000,
                         num_labels: int = 100, force: bool = False, quiet: bool = False,
                         threads: int = None):
    """计算单个标签的GT文件"""
    query_labels_file = f"{data_dir}/query_labels_{label_id}.spmat"
    gt_file = f"{data_dir}/groundtruth_1m_filtered_label_{label_id}.bin"
    
    if not force and os.path.exists(query_labels_file) and os.path.exists(gt_file):
        if not quiet:
            print(f"  [Label {label_id}] 已存在，跳过")
        return True
    
    if not quiet:
        print(f"  [Label {label_id}] 生成中...", flush=True)
    
    try:
        generate_query_spmat_single_label(query_labels_file, num_queries, num_labels, label_id - 1)
        
        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
        
        subprocess.run([
            str(project_root / "build/tests/utils/compute_groundtruth"), "uint8",
            metric, f"{data_dir}/bigann_1m.bin", f"{data_dir}/bigann_query.bin",
            str(k), gt_file,
            "null", "spmat", "subset",
            f"{data_dir}/data_labels.spmat", query_labels_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, env=env)
        
        if not quiet:
            print(f"  [Label {label_id}] 完成")
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"  [Label {label_id}] 失败: {e}")
        return False


def cmd_compute_gt(args):
    """计算单个标签的 Ground Truth"""
    data_dir = args.data_dir or os.getenv("SIFT1M_DATA_DIR", "/mnt/ext4/lzg/sift1m_pq")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    print("========================================")
    print("运行 compute_groundtruth")
    print(f"选择性: {args.label_id}%")
    print("========================================")
    print()
    
    if not os.path.exists(project_root / "build/tests/utils/compute_groundtruth"):
        print("❌ compute_groundtruth 未编译")
        sys.exit(1)
    
    if not os.path.exists(f"{data_dir}/bigann_1m.bin"):
        print(f"❌ 数据文件不存在: {data_dir}/bigann_1m.bin")
        sys.exit(1)
    
    if not os.path.exists(f"{data_dir}/data_labels.spmat"):
        print(f"❌ 标签文件不存在: {data_dir}/data_labels.spmat")
        sys.exit(1)
    
    success = compute_gt_for_label(
        args.label_id, data_dir, project_root,
        k=args.k, metric=args.metric,
        num_queries=args.num_queries, num_labels=args.num_labels,
        force=args.force, threads=args.threads
    )
    
    if success:
        print()
        print("========================================")
        print("完成！")
        print("========================================")
    else:
        sys.exit(1)


def cmd_compute_gt_all(args):
    """批量计算所有标签的 Ground Truth"""
    data_dir = args.data_dir or os.getenv("SIFT1M_DATA_DIR", "/mnt/ext4/lzg/sift1m_pq")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    print("========================================")
    print("预计算 Ground Truth 文件")
    print("========================================")
    print(f"数据目录: {data_dir}")
    
    if not os.path.exists(f"{data_dir}/bigann_1m.bin"):
        print("❌ 数据文件不存在")
        sys.exit(1)
    
    if not os.path.exists(f"{data_dir}/data_labels.spmat"):
        print("❌ 标签文件不存在")
        sys.exit(1)
    
    if not os.path.exists(project_root / "build/tests/utils/compute_groundtruth"):
        print("❌ compute_groundtruth 未编译")
        sys.exit(1)
    
    if args.labels:
        label_list = [int(x.strip()) for x in args.labels.split(',')]
    else:
        label_list = list(range(args.start, args.end + 1))
    
    print(f"待计算标签: {label_list[0]}-{label_list[-1]} (共 {len(label_list)} 个)")
    print()
    
    success_count = 0
    for label_id in label_list:
        if compute_gt_for_label(
            label_id, data_dir, project_root,
            k=args.k, metric=args.metric,
            num_queries=args.num_queries, num_labels=args.num_labels,
            force=args.force, threads=args.threads
        ):
            success_count += 1
    
    print()
    print("========================================")
    print(f"完成: {success_count}/{len(label_list)} 个标签")
    print("========================================")


def main():
    parser = argparse.ArgumentParser(
        description='为 PipeANN 生成标签文件和计算 Ground Truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 生成数据标签（标签i有i%的向量包含）:
   python gen_random_labels.py data-labels \\
       --output /data/sift1m_pq/data_labels.spmat \\
       --num-vectors 1000000 --num-labels 100

2. 生成查询标签:
   python gen_random_labels.py query-labels \\
       --output /data/sift1m_pq/query_labels.spmat \\
       --num-queries 10000 --num-labels 100 --label-id 5

3. 计算单个标签的 Ground Truth:
   python gen_random_labels.py compute-gt --label-id 60

4. 批量计算所有标签的 Ground Truth:
   python gen_random_labels.py compute-gt-all --start 1 --end 100
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='命令类型')

    parser_data = subparsers.add_parser('data-labels', help='生成数据标签')
    parser_data.add_argument('--output', required=True, help='输出文件路径')
    parser_data.add_argument('--num-vectors', type=int, required=True, help='向量数量')
    parser_data.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_data.add_argument('--seed', type=int, default=42, help='随机种子')
    parser_data.add_argument('--percentages', type=str, default=None, 
                            help='标签占比，格式: "0:10,1:20"。默认为标签i占(i+1)%%')

    parser_query = subparsers.add_parser('query-labels', help='生成查询标签')
    parser_query.add_argument('--output', required=True, help='输出文件路径')
    parser_query.add_argument('--num-queries', type=int, required=True, help='查询数量')
    parser_query.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_query.add_argument('--label-id', type=int, required=True, help='标签ID（0-based）')

    parser_gt = subparsers.add_parser('compute-gt', help='计算单个标签的 Ground Truth')
    parser_gt.add_argument('--label-id', type=int, required=True, help='标签ID（1-based，对应选择性百分比）')
    parser_gt.add_argument('--data-dir', type=str, default=None, help='数据目录 (默认: $SIFT1M_DATA_DIR)')
    parser_gt.add_argument('--k', type=int, default=10, help='Top-K (默认: 10)')
    parser_gt.add_argument('--metric', type=str, default='l2', help='距离度量 (默认: l2)')
    parser_gt.add_argument('--num-queries', type=int, default=10000, help='查询数量 (默认: 10000)')
    parser_gt.add_argument('--num-labels', type=int, default=100, help='标签空间大小 (默认: 100)')
    parser_gt.add_argument('--force', action='store_true', help='强制重新计算')
    parser_gt.add_argument('--threads', type=int, default=None, help='限制最大线程数 (通过OMP_NUM_THREADS)')

    parser_gt_all = subparsers.add_parser('compute-gt-all', help='批量计算所有标签的 Ground Truth')
    parser_gt_all.add_argument('--start', type=int, default=1, help='起始标签ID (默认: 1)')
    parser_gt_all.add_argument('--end', type=int, default=100, help='结束标签ID (默认: 100)')
    parser_gt_all.add_argument('--labels', type=str, default=None, help='指定标签列表，逗号分隔')
    parser_gt_all.add_argument('--data-dir', type=str, default=None, help='数据目录 (默认: $SIFT1M_DATA_DIR)')
    parser_gt_all.add_argument('--k', type=int, default=10, help='Top-K (默认: 10)')
    parser_gt_all.add_argument('--metric', type=str, default='l2', help='距离度量 (默认: l2)')
    parser_gt_all.add_argument('--num-queries', type=int, default=10000, help='查询数量 (默认: 10000)')
    parser_gt_all.add_argument('--num-labels', type=int, default=100, help='标签空间大小 (默认: 100)')
    parser_gt_all.add_argument('--force', action='store_true', help='强制重新计算')
    parser_gt_all.add_argument('--threads', type=int, default=None, help='限制最大线程数 (通过OMP_NUM_THREADS)')

    args = parser.parse_args()

    if args.command == 'data-labels':
        if args.percentages:
            label_percentages = {}
            for pair in args.percentages.split(','):
                label_id, percentage = pair.split(':')
                label_percentages[int(label_id)] = float(percentage)
        else:
            label_percentages = {i: i + 1 for i in range(args.num_labels)}
        
        generate_spmat_labels_with_distribution(args.output, args.num_vectors, 
                                                 label_percentages, args.seed)
    elif args.command == 'query-labels':
        generate_query_spmat_single_label(args.output, args.num_queries, 
                                          args.num_labels, args.label_id)
    elif args.command == 'compute-gt':
        cmd_compute_gt(args)
    elif args.command == 'compute-gt-all':
        cmd_compute_gt_all(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()