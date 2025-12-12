#!/usr/bin/env python3
"""
计算所有标签的 Ground Truth 文件

此脚本独立于搜索脚本运行，可以在搜索开始前或同时运行。
搜索脚本会等待GT文件生成完成后再使用。

用法:
    python3 compute_gt_all.py                    # 计算所有标签 (1-100)
    python3 compute_gt_all.py --start 50         # 从标签50开始
    python3 compute_gt_all.py --labels 1,5,10    # 只计算指定标签
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='计算所有标签的 Ground Truth 文件')
    parser.add_argument('--start', type=int, default=1, help='起始标签ID (默认: 1)')
    parser.add_argument('--end', type=int, default=100, help='结束标签ID (默认: 100)')
    parser.add_argument('--labels', type=str, default=None, help='指定标签列表，逗号分隔')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录 (默认: $SIFT1M_DATA_DIR)')
    parser.add_argument('--force', action='store_true', help='强制重新计算已存在的GT文件')
    parser.add_argument('--k', type=int, default=10, help='Top-K (默认: 10)')
    parser.add_argument('--metric', type=str, default='l2', help='距离度量 (默认: l2)')
    parser.add_argument('--threads', type=int, default=None, help='限制最大线程数 (通过OMP_NUM_THREADS)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    gen_labels_script = script_dir / "gen_random_labels.py"
    
    if not gen_labels_script.exists():
        print(f"错误: 找不到 {gen_labels_script}")
        sys.exit(1)
    
    cmd = ["python3", str(gen_labels_script), "compute-gt-all"]
    cmd.extend(["--start", str(args.start)])
    cmd.extend(["--end", str(args.end)])
    
    if args.labels:
        cmd.extend(["--labels", args.labels])
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    if args.force:
        cmd.append("--force")
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--metric", args.metric])
    if args.threads:
        cmd.extend(["--threads", str(args.threads)])
    
    env = os.environ.copy()
    if args.data_dir:
        env["SIFT1M_DATA_DIR"] = args.data_dir
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(130)


if __name__ == "__main__":
    main()