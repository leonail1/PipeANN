#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 PipeANN 生成随机标签文件，支持 filtered search 功能

支持两种标签格式：
1. range 格式：每个向量对应一个 uint32 值，用于范围过滤
2. spmat 格式：每个向量对应一组标签集合，用于集合过滤（交集、子集等）
"""

import struct
import numpy as np
import argparse
from typing import List, Set
from tqdm import tqdm


def generate_range_labels(output_file: str, num_vectors: int,
                          min_value: int, max_value: int, seed: int = 42):
    """
    生成范围标签（Range Labels）

    范围标签格式：每个向量对应一个 uint32_t 值
    用途：使用 RangeSelector 进行范围过滤，例如查询标签在 [100, 200] 范围内的向量

    Args:
        output_file: 输出文件路径（.bin 格式）
        num_vectors: 向量数量
        min_value: 标签最小值
        max_value: 标签最大值（不包含）
        seed: 随机种子
    """
    np.random.seed(seed)

    # 生成随机标签值
    labels = np.random.randint(min_value, max_value, size=num_vectors, dtype=np.uint32)

    # 写入二进制文件
    with open(output_file, 'wb') as f:
        # 每个向量写入一个 uint32_t 值
        f.write(labels.tobytes())

    print(f"✓ 成功生成范围标签")
    print(f"  - 向量数量: {num_vectors:,}")
    print(f"  - 值域范围: [{min_value}, {max_value})")
    print(f"  - 输出文件: {output_file}")
    print(f"  - 文件大小: {num_vectors * 4 / 1024 / 1024:.2f} MB")


def generate_spmat_labels(output_file: str, num_vectors: int, num_labels: int,
                          min_labels_per_vector: int, max_labels_per_vector: int,
                          seed: int = 42):
    """
    生成稀疏矩阵标签（Spmat Labels）

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
        num_labels: 标签空间大小（矩阵列数）
        min_labels_per_vector: 每个向量最少标签数
        max_labels_per_vector: 每个向量最多标签数
        seed: 随机种子
    """
    np.random.seed(seed)

    # 初始化 CSR 格式的稀疏矩阵数据结构
    indptr = [0]  # 行指针数组，长度为 nrow + 1
    indices = []  # 列索引数组，存储非零元素的列号
    data = []     # 数据数组，存储非零元素的值

    # 为每个向量生成随机标签
    for i in range(num_vectors):
        # 随机决定这个向量有多少个标签
        num_labels_for_vector = np.random.randint(min_labels_per_vector,
                                                   max_labels_per_vector + 1)

        # 随机选择标签（不重复）
        vector_labels = np.random.choice(num_labels,
                                        size=min(num_labels_for_vector, num_labels),
                                        replace=False)
        vector_labels = sorted(vector_labels)  # 排序以提高查询效率

        # 添加到稀疏矩阵结构
        for label_id in vector_labels:
            indices.append(label_id)
            data.append(1.0)  # 非零值表示该标签存在

        # 更新行指针
        indptr.append(len(indices))

    # 转换为 numpy 数组
    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    # 写入 spmat 文件
    # 文件格式：
    #   1. Header: [nrow:int64][ncol:int64][nnz:int64]
    #   2. indptr: (nrow+1) 个 int64
    #   3. indices: nnz 个 int32
    #   4. data: nnz 个 float32
    with open(output_file, 'wb') as f:
        # 写入 header
        nrow = num_vectors
        ncol = num_labels
        nnz = len(indices)

        f.write(struct.pack('<q', nrow))   # int64
        f.write(struct.pack('<q', ncol))   # int64
        f.write(struct.pack('<q', nnz))    # int64

        # 写入 indptr
        f.write(indptr.tobytes())

        # 写入 indices
        f.write(indices.tobytes())

        # 写入 data
        f.write(data.tobytes())

    avg_labels = nnz / num_vectors if num_vectors > 0 else 0

    print(f"✓ 成功生成稀疏矩阵标签")
    print(f"  - 向量数量 (nrow): {nrow:,}")
    print(f"  - 标签空间 (ncol): {ncol:,}")
    print(f"  - 非零元素 (nnz): {nnz:,}")
    print(f"  - 平均标签数: {avg_labels:.2f}")
    print(f"  - 标签范围: [{min_labels_per_vector}, {max_labels_per_vector}]")
    print(f"  - 输出文件: {output_file}")

    # 计算文件大小
    file_size = (3 * 8 + len(indptr) * 8 + len(indices) * 4 + len(data) * 4) / 1024 / 1024
    print(f"  - 文件大小: {file_size:.2f} MB")


def generate_query_range_labels(output_file: str, num_queries: int,
                                min_value: int, max_value: int,
                                range_size: int, seed: int = 42):
    """
    生成查询范围标签（用于 RangeSelector）

    每个查询包含一个范围 [low, high]，用于过滤数据

    Args:
        output_file: 输出文件路径
        num_queries: 查询数量
        min_value: 最小值
        max_value: 最大值
        range_size: 范围大小（high - low）
        seed: 随机种子
    """
    np.random.seed(seed)

    with open(output_file, 'wb') as f:
        for i in range(num_queries):
            # 随机生成范围的起始点
            low = np.random.randint(min_value, max_value - range_size + 1)
            high = low + range_size

            # 写入 [low, high] 两个 uint32_t
            f.write(struct.pack('<I', low))
            f.write(struct.pack('<I', high))

    print(f"✓ 成功生成查询范围标签")
    print(f"  - 查询数量: {num_queries:,}")
    print(f"  - 值域范围: [{min_value}, {max_value})")
    print(f"  - 范围大小: {range_size}")
    print(f"  - 输出文件: {output_file}")


def load_spmat_labels(spmat_file: str) -> List[Set[int]]:
    """
    从 spmat 文件加载标签数据

    Returns:
        List[Set[int]]: 每个向量的标签集合列表
    """
    with open(spmat_file, 'rb') as f:
        # 读取 header
        nrow = struct.unpack('<q', f.read(8))[0]
        ncol = struct.unpack('<q', f.read(8))[0]
        nnz = struct.unpack('<q', f.read(8))[0]

        # 读取 indptr
        indptr = np.frombuffer(f.read((nrow + 1) * 8), dtype=np.int64)

        # 读取 indices
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)

        # 读取 data (不使用)
        _ = np.frombuffer(f.read(nnz * 4), dtype=np.float32)

    # 构建标签集合列表
    labels_list = []
    for i in range(nrow):
        start = indptr[i]
        end = indptr[i + 1]
        labels = set(indices[start:end])
        labels_list.append(labels)

    return labels_list


def save_spmat_labels(spmat_file: str, labels_list: List[Set[int]], num_labels: int):
    """
    保存标签数据到 spmat 文件

    Args:
        spmat_file: 输出文件路径
        labels_list: 每个向量的标签集合列表
        num_labels: 标签空间大小
    """
    # 构建 CSR 格式
    indptr = [0]
    indices = []
    data = []

    for labels in tqdm(labels_list, desc="保存数据标签", unit="vector"):
        # 排序标签以提高查询效率
        sorted_labels = sorted(labels)
        for label_id in sorted_labels:
            indices.append(label_id)
            data.append(1.0)
        indptr.append(len(indices))

    # 转换为 numpy 数组
    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    # 写入 spmat 文件
    with open(spmat_file, 'wb') as f:
        nrow = len(labels_list)
        ncol = num_labels
        nnz = len(indices)

        f.write(struct.pack('<q', nrow))
        f.write(struct.pack('<q', ncol))
        f.write(struct.pack('<q', nnz))
        f.write(indptr.tobytes())
        f.write(indices.tobytes())
        f.write(data.tobytes())


def count_subset_matches(query_labels: Set[int], data_labels_list: List[Set[int]]) -> int:
    """
    计算有多少个数据点满足 subset 条件（query_labels ⊆ data_labels）
    """
    count = 0
    for data_labels in data_labels_list:
        if query_labels.issubset(data_labels):
            count += 1
    return count


def count_intersection_matches(query_labels: Set[int], data_labels_list: List[Set[int]]) -> int:
    """
    计算有多少个数据点满足 intersection 条件（query_labels ∩ data_labels ≠ ∅）
    """
    count = 0
    for data_labels in data_labels_list:
        if len(query_labels & data_labels) > 0:
            count += 1
    return count


def generate_query_spmat_labels(output_file: str, num_queries: int, num_labels: int,
                                min_labels_per_query: int, max_labels_per_query: int,
                                seed: int = 42, data_labels_file: str = None,
                                min_matches: int = 10, selector_type: str = 'subset'):
    """
    生成查询标签集合（用于 Intersection/Subset Selector）

    格式与数据的 spmat 格式相同

    Args:
        output_file: 输出文件路径
        num_queries: 查询数量
        num_labels: 标签空间大小
        min_labels_per_query: 每个查询最少标签数
        max_labels_per_query: 每个查询最多标签数
        seed: 随机种子
        data_labels_file: 数据标签文件路径（用于确保足够的匹配数）
        min_matches: 每个查询至少需要的匹配数量
        selector_type: 过滤选择器类型 ('subset' 或 'intersect')
    """
    # 如果没有提供数据标签文件，使用原始逻辑
    if data_labels_file is None:
        generate_spmat_labels(output_file, num_queries, num_labels,
                            min_labels_per_query, max_labels_per_query, seed)
        return

    # 检查数据标签文件是否存在
    import os
    if not os.path.isfile(data_labels_file):
        raise FileNotFoundError(f"数据标签文件不存在: {data_labels_file}")

    print(f"加载数据标签: {data_labels_file}")
    data_labels_list = load_spmat_labels(data_labels_file)
    print(f"  已加载 {len(data_labels_list):,} 个数据点的标签")

    np.random.seed(seed)

    # 初始化 CSR 格式
    indptr = [0]
    indices = []
    data = []

    augmented_count = 0  # 记录修改了多少个数据向量的标签

    print(f"开始生成查询标签（确保每个查询至少有 {min_matches} 个匹配）...")

    for query_idx in tqdm(range(num_queries), desc="生成查询标签", unit="query"):
        # 随机生成查询标签
        num_labels_for_query = np.random.randint(min_labels_per_query,
                                                 max_labels_per_query + 1)
        query_labels = set(np.random.choice(num_labels,
                                            size=min(num_labels_for_query, num_labels),
                                            replace=False))

        # 计算当前匹配数量
        if selector_type == 'subset':
            match_count = count_subset_matches(query_labels, data_labels_list)
        else:  # intersect
            match_count = count_intersection_matches(query_labels, data_labels_list)

        # 如果匹配数不足，直接修改数据标签来增加匹配
        if match_count < min_matches:
            needed = min_matches - match_count

            # 随机选择 needed 个数据向量，将查询标签添加到它们的标签集合中
            available_indices = list(range(len(data_labels_list)))
            np.random.shuffle(available_indices)

            for i in range(needed):
                data_idx = available_indices[i]
                # 将查询标签添加到数据向量的标签集合中
                data_labels_list[data_idx].update(query_labels)
                augmented_count += 1

        # 保存查询标签
        query_labels_sorted = sorted(query_labels)
        for label_id in query_labels_sorted:
            indices.append(label_id)
            data.append(1.0)
        indptr.append(len(indices))

    # 如果修改了数据标签，需要写回数据标签文件
    if augmented_count > 0:
        print(f"\n💡 已向 {augmented_count} 个数据向量添加标签以确保足够匹配")
        print(f"   正在更新数据标签文件: {data_labels_file}")
        save_spmat_labels(data_labels_file, data_labels_list, num_labels)
        print(f"   ✓ 数据标签文件已更新")

    # 转换为 numpy 数组并写入查询标签文件
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

    avg_labels = nnz / num_queries if num_queries > 0 else 0

    print(f"\n✓ 成功生成查询标签")
    print(f"  - 查询数量: {num_queries:,}")
    print(f"  - 标签空间: {num_labels:,}")
    print(f"  - 非零元素: {nnz:,}")
    print(f"  - 平均标签数: {avg_labels:.2f}")
    print(f"  - 标签范围: [{min_labels_per_query}, {max_labels_per_query}]")
    print(f"  - 选择器类型: {selector_type}")
    print(f"  - 最小匹配数: {min_matches}")
    print(f"  - 增强的数据向量数: {augmented_count}")
    print(f"  - 输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='为 PipeANN 生成随机标签文件（支持 filtered search）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 为 SIFT1M 数据集生成范围标签（数据标签）：
   python gen_random_labels.py range \\
       --output /data/lzg/sift-pipeann/sift1m_pq/data_labels.bin \\
       --num-vectors 1000000 \\
       --min-value 0 \\
       --max-value 1000

2. 为查询生成范围标签（查询标签）：
   python gen_random_labels.py query-range \\
       --output /data/lzg/sift-pipeann/sift1m_pq/query_labels.bin \\
       --num-queries 10000 \\
       --min-value 0 \\
       --max-value 1000 \\
       --range-size 100

3. 为 SIFT1M 数据集生成 spmat 标签（数据标签）：
   python gen_random_labels.py spmat \\
       --output /data/lzg/sift-pipeann/sift1m_pq/data_labels.spmat \\
       --num-vectors 1000000 \\
       --num-labels 100 \\
       --min-labels 1 \\
       --max-labels 5

4. 为查询生成 spmat 标签（查询标签）：
   python gen_random_labels.py query-spmat \\
       --output /data/lzg/sift-pipeann/sift1m_pq/query_labels.spmat \\
       --num-queries 10000 \\
       --num-labels 100 \\
       --min-labels 1 \\
       --max-labels 3
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='标签类型')

    # range 标签（数据）
    parser_range = subparsers.add_parser('range', help='生成范围标签（数据）')
    parser_range.add_argument('--output', required=True, help='输出文件路径')
    parser_range.add_argument('--num-vectors', type=int, required=True, help='向量数量')
    parser_range.add_argument('--min-value', type=int, required=True, help='最小值')
    parser_range.add_argument('--max-value', type=int, required=True, help='最大值')
    parser_range.add_argument('--seed', type=int, default=42, help='随机种子')

    # query-range 标签（查询）
    parser_qrange = subparsers.add_parser('query-range', help='生成查询范围标签')
    parser_qrange.add_argument('--output', required=True, help='输出文件路径')
    parser_qrange.add_argument('--num-queries', type=int, required=True, help='查询数量')
    parser_qrange.add_argument('--min-value', type=int, required=True, help='最小值')
    parser_qrange.add_argument('--max-value', type=int, required=True, help='最大值')
    parser_qrange.add_argument('--range-size', type=int, required=True, help='范围大小')
    parser_qrange.add_argument('--seed', type=int, default=42, help='随机种子')

    # spmat 标签（数据）
    parser_spmat = subparsers.add_parser('spmat', help='生成稀疏矩阵标签（数据）')
    parser_spmat.add_argument('--output', required=True, help='输出文件路径')
    parser_spmat.add_argument('--num-vectors', type=int, required=True, help='向量数量')
    parser_spmat.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_spmat.add_argument('--min-labels', type=int, required=True, help='每个向量最少标签数')
    parser_spmat.add_argument('--max-labels', type=int, required=True, help='每个向量最多标签数')
    parser_spmat.add_argument('--seed', type=int, default=42, help='随机种子')

    # query-spmat 标签（查询）
    parser_qspmat = subparsers.add_parser('query-spmat', help='生成查询标签集合')
    parser_qspmat.add_argument('--output', required=True, help='输出文件路径')
    parser_qspmat.add_argument('--num-queries', type=int, required=True, help='查询数量')
    parser_qspmat.add_argument('--num-labels', type=int, required=True, help='标签空间大小')
    parser_qspmat.add_argument('--min-labels', type=int, required=True, help='每个查询最少标签数')
    parser_qspmat.add_argument('--max-labels', type=int, required=True, help='每个查询最多标签数')
    parser_qspmat.add_argument('--seed', type=int, default=42, help='随机种子')
    parser_qspmat.add_argument('--data-labels', type=str, default=None,
                               help='数据标签文件路径（用于确保足够的匹配数）')
    parser_qspmat.add_argument('--min-matches', type=int, default=10,
                               help='每个查询至少需要的匹配数量（默认：10）')
    parser_qspmat.add_argument('--selector', type=str, default='subset',
                               choices=['subset', 'intersect'],
                               help='过滤选择器类型（默认：subset）')

    args = parser.parse_args()

    if args.command == 'range':
        generate_range_labels(args.output, args.num_vectors,
                            args.min_value, args.max_value, args.seed)
    elif args.command == 'query-range':
        generate_query_range_labels(args.output, args.num_queries,
                                   args.min_value, args.max_value,
                                   args.range_size, args.seed)
    elif args.command == 'spmat':
        generate_spmat_labels(args.output, args.num_vectors, args.num_labels,
                            args.min_labels, args.max_labels, args.seed)
    elif args.command == 'query-spmat':
        generate_query_spmat_labels(args.output, args.num_queries, args.num_labels,
                                   args.min_labels, args.max_labels, args.seed,
                                   data_labels_file=args.data_labels,
                                   min_matches=args.min_matches,
                                   selector_type=args.selector)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
