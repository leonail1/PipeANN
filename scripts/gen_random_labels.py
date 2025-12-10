#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 PipeANN 生成标签文件，支持 filtered search 功能

支持 spmat 格式标签：每个向量对应一组标签集合，用于集合过滤（交集、子集等）
"""

import struct
import numpy as np
import argparse
from typing import List, Dict
from tqdm import tqdm


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


def main():
    parser = argparse.ArgumentParser(
        description='为 PipeANN 生成标签文件（支持 filtered search）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 为 SIFT1M 数据集生成 spmat 标签（数据标签），标签i有i%的向量包含：
   python gen_random_labels.py data-labels \\
       --output /data/lzg/sift-pipeann/sift1m_pq/data_labels.spmat \\
       --num-vectors 1000000 \\
       --num-labels 100 \\
       --seed 42

2. 为查询生成单个标签的 spmat 文件：
   python gen_random_labels.py query-labels \\
       --output /data/lzg/sift-pipeann/sift1m_pq/query_labels.spmat \\
       --num-queries 10000 \\
       --num-labels 100 \\
       --label-id 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='标签类型')

    parser_data = subparsers.add_parser('data-labels', help='生成数据标签（可指定每个标签的占比）')
    parser_data.add_argument('--output', required=True, help='输出文件路径')
    parser_data.add_argument('--num-vectors', type=int, required=True, help='向量数量')
    parser_data.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_data.add_argument('--seed', type=int, default=42, help='随机种子')
    parser_data.add_argument('--percentages', type=str, default=None, 
                            help='标签占比，格式: "0:10,1:20,2:30" 表示标签0占10%%，标签1占20%%，标签2占30%%。默认为标签i占(i+1)%%')

    parser_query = subparsers.add_parser('query-labels', help='生成查询标签（单个标签）')
    parser_query.add_argument('--output', required=True, help='输出文件路径')
    parser_query.add_argument('--num-queries', type=int, required=True, help='查询数量')
    parser_query.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_query.add_argument('--label-id', type=int, required=True, help='标签ID（0-based）')

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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()