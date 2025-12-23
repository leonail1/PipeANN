#!/usr/bin/env python3
"""
Pre-filter Search 纯顺序流式处理实验脚本

用法:
    python3 run_prefilter_experiment.py --label 50
    python3 run_prefilter_experiment.py --start-label 1 --end-label 100 --num-runs 20
    python3 run_prefilter_experiment.py --threads 1 2 4 8
    python3 run_prefilter_experiment.py --measure-memory
    python3 run_prefilter_experiment.py --batch-size 100000
"""

import os
import sys
import subprocess
import re
import argparse
import time
from pathlib import Path


def ensure_bitmap_file(data_dir, build_bitmap_exe):
    bitmap_file = f"{data_dir}/data_labels.bitmap"
    label_file = f"{data_dir}/data_labels.spmat"
    
    if os.path.exists(bitmap_file):
        return bitmap_file
    
    print(f"位图文件不存在，正在生成: {bitmap_file}")
    if not os.path.exists(build_bitmap_exe):
        print(f"错误: build_label_bitmap 不存在: {build_bitmap_exe}")
        print("   请先编译: cd build && make build_label_bitmap")
        sys.exit(1)
    
    result = subprocess.run([str(build_bitmap_exe), label_file, bitmap_file],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"生成位图失败: {result.stderr}")
        sys.exit(1)
    
    print(f"位图文件已生成: {bitmap_file}")
    return bitmap_file


def wait_for_gt_file(gt_file: str, query_labels_file: str, label_id: int,
                     check_interval: float = 2.0, first_wait: bool = False) -> bool:
    if os.path.exists(gt_file) and os.path.exists(query_labels_file):
        return True
    
    if first_wait:
        print()
        print("=" * 60)
        print("  GT文件不存在，等待计算...")
        print("  请在另一个终端运行以下命令来计算GT:")
        print()
        print("    python3 /home/lzg/PipeANN/scripts/compute_gt_all.py")
        print()
        print("=" * 60)
        print()
    
    wait_count = 0
    while not (os.path.exists(gt_file) and os.path.exists(query_labels_file)):
        wait_count += 1
        if wait_count % 15 == 1:
            print(f"    等待 Label {label_id} 的GT文件... (已等待 {int(wait_count * check_interval)}s)", flush=True)
        time.sleep(check_interval)
    
    print(f"    ✓ Label {label_id} 的GT文件已就绪")
    return True


def run_single_label(prefilter_exe, data_dir, label_id, num_runs,
                     num_threads=1, batch_size=200000, measure_memory=False):
    data_file = f"{data_dir}/bigann_1m.bin"
    query_file = f"{data_dir}/bigann_query.bin"
    query_labels_file = f"{data_dir}/query_labels_{label_id}.spmat"
    gt_file = f"{data_dir}/groundtruth_1m_filtered_label_{label_id}.bin"
    bitmap_file = f"{data_dir}/data_labels.bitmap"
    
    base_cmd = [
        str(prefilter_exe), "uint8",
        data_file, query_file, "l2", "subset",
        bitmap_file, query_labels_file,
        gt_file, "10", str(num_runs), str(num_threads), str(batch_size)
    ]
    
    peak_memory_kb = None
    
    if measure_memory:
        cmd = ["/usr/bin/time", "-v"] + base_cmd
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        mem_match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', output)
        if mem_match:
            peak_memory_kb = int(mem_match.group(1))
    else:
        result = subprocess.run(base_cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
    
    table_match = re.search(
        r'^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',
        output, re.MULTILINE)
    
    if table_match:
        result_dict = {
            'filter_lat': float(table_match.group(1)),
            'filter_std': float(table_match.group(2)),
            'search_lat': float(table_match.group(3)),
            'search_std': float(table_match.group(4)),
            'total_lat': float(table_match.group(5)),
            'filtered_cnt': float(table_match.group(6)),
            'valid_runs': int(float(table_match.group(7))),
            'recall': float(table_match.group(8)),
            'peak_memory_mb': peak_memory_kb / 1024.0 if peak_memory_kb else None
        }
        return result_dict
    return None


def run_experiment_for_threads(prefilter_exe, data_dir, script_dir, num_threads, 
                                start_label, end_label, num_runs, batch_size, measure_memory=False):
    results_file = script_dir / f"prefilter_streaming_results_threads_{num_threads}.csv"
    
    completed_labels = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(',') if lines else []
            has_memory = 'PeakMemoryMB' in header
            
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    label_id = int(parts[0])
                    result = {
                        'filter_lat': float(parts[2]),
                        'filter_std': float(parts[3]),
                        'search_lat': float(parts[4]),
                        'search_std': float(parts[5]),
                        'total_lat': float(parts[6]),
                        'filtered_cnt': float(parts[7]),
                        'valid_runs': int(float(parts[8])),
                        'recall': float(parts[9]),
                        'peak_memory_mb': None
                    }
                    if has_memory and len(parts) > 10:
                        try:
                            result['peak_memory_mb'] = float(parts[10]) if parts[10] else None
                        except (ValueError, IndexError):
                            pass
                    completed_labels[label_id] = result
        print(f"  线程数 {num_threads}: 已完成的Label数: {len(completed_labels)}")
    else:
        with open(results_file, 'w') as f:
            f.write("Label,Selectivity,FilterLat_us,FilterStd,SearchLat_us,SearchStd,TotalLat_us,FilteredCnt,ValidRuns,Recall,PeakMemoryMB\n")
    
    results = dict(completed_labels)
    first_wait_done = False
    
    for label_id in range(start_label, end_label + 1):
        if label_id in completed_labels:
            if not measure_memory or completed_labels[label_id].get('peak_memory_mb') is not None:
                continue
        
        query_labels_file = f"{data_dir}/query_labels_{label_id}.spmat"
        gt_file = f"{data_dir}/groundtruth_1m_filtered_label_{label_id}.bin"
        
        mem_str = " (测量内存)" if measure_memory else ""
        print(f"  [{num_threads}T] [{label_id}/{end_label}] 选择性={label_id}%{mem_str} ... ", flush=True)
        
        if not os.path.exists(query_labels_file) or not os.path.exists(gt_file):
            wait_for_gt_file(gt_file, query_labels_file, label_id,
                           check_interval=2.0, first_wait=not first_wait_done)
            first_wait_done = True
        
        result = run_single_label(prefilter_exe, data_dir, label_id, num_runs,
                                  num_threads, batch_size, measure_memory)
        
        if result:
            results[label_id] = result
            mem_info = f", Mem={result['peak_memory_mb']:.1f}MB" if result.get('peak_memory_mb') else ""
            print(f"    FilterLat={result['filter_lat']:.2f}us, SearchLat={result['search_lat']:.2f}us, "
                  f"Total={result['total_lat']:.2f}us, Recall={result['recall']:.2f}{mem_info}")
            
            with open(results_file, 'a') as f:
                mem_val = f"{result['peak_memory_mb']:.2f}" if result.get('peak_memory_mb') else ""
                f.write(f"{label_id},{label_id},{result['filter_lat']},{result['filter_std']},"
                        f"{result['search_lat']},{result['search_std']},{result['total_lat']},"
                        f"{result['filtered_cnt']},{result['valid_runs']},{result['recall']},{mem_val}\n")
        else:
            print(f"    ⚠ 无法解析输出，跳过")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Pre-filter Search 纯顺序流式处理实验')
    parser.add_argument('--num-runs', type=int, default=20, help='每个label运行次数 (默认: 20)')
    parser.add_argument('--start-label', type=int, default=1, help='起始Label ID (默认: 1)')
    parser.add_argument('--end-label', type=int, default=100, help='结束Label ID (默认: 100)')
    parser.add_argument('--label', type=int, default=None, help='指定单个Label ID进行测试')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录')
    parser.add_argument('--threads', type=int, nargs='+', default=[1], help='OpenMP线程数列表 (默认: 1)')
    parser.add_argument('--measure-memory', action='store_true', help='使用/usr/bin/time测量内存峰值')
    parser.add_argument('--batch-size', type=int, default=200000, help='流式处理分片大小 (默认: 200000)')
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir or os.getenv("SIFT1M_DATA_DIR", "/mnt/ext4/lzg/sift1m_pq")
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    PREFILTER_EXE = PROJECT_ROOT / "build/tests/search_prefilter_streaming"
    BUILD_BITMAP_EXE = PROJECT_ROOT / "build/tests/build_label_bitmap"
    
    thread_list = args.threads
    batch_size = args.batch_size
    
    single_label_mode = args.label is not None
    
    ensure_bitmap_file(DATA_DIR, BUILD_BITMAP_EXE)
    
    if single_label_mode:
        num_threads = thread_list[0]
        print("========================================")
        print(f"SIFT1M Pre-filter Sequential Streaming 单Label测试 (Label={args.label})")
        print("========================================")
        print(f"数据目录: {DATA_DIR}")
        print(f"运行次数: {args.num_runs}")
        print(f"分片大小: {batch_size}")
        if args.measure_memory:
            print("内存测量: 启用")
        print()
        
        if not os.path.exists(PREFILTER_EXE):
            print(f"错误: 可执行文件不存在: {PREFILTER_EXE}")
            print("   请先编译项目: cd build && make search_prefilter_streaming")
            sys.exit(1)
        
        query_labels_file = f"{DATA_DIR}/query_labels_{args.label}.spmat"
        gt_file = f"{DATA_DIR}/groundtruth_1m_filtered_label_{args.label}.bin"
        
        if not os.path.exists(query_labels_file) or not os.path.exists(gt_file):
            wait_for_gt_file(gt_file, query_labels_file, args.label, check_interval=2.0, first_wait=True)
        
        result = run_single_label(PREFILTER_EXE, DATA_DIR, args.label, args.num_runs,
                                  num_threads, batch_size, args.measure_memory)
        
        if result:
            print(f"Label {args.label} 测试结果:")
            print(f"  Filter Latency:  {result['filter_lat']:.2f} us (std: {result['filter_std']:.2f})")
            print(f"  Search Latency:  {result['search_lat']:.2f} us (std: {result['search_std']:.2f})")
            print(f"  Total Latency:   {result['total_lat']:.2f} us")
            print(f"  Filtered Count:  {result['filtered_cnt']:.0f}")
            print(f"  Valid Runs:      {result['valid_runs']}")
            print(f"  Recall:          {result['recall']:.4f}")
            if result.get('peak_memory_mb'):
                print(f"  Peak Memory:     {result['peak_memory_mb']:.2f} MB")
        else:
            print("无法解析输出结果")
            sys.exit(1)
        return
    
    print("========================================")
    print("SIFT1M Pre-filter Sequential Streaming 实验")
    print("========================================")
    print(f"数据目录: {DATA_DIR}")
    print(f"运行次数: {args.num_runs}")
    print(f"Label范围: {args.start_label} - {args.end_label}")
    print(f"线程数列表: {thread_list}")
    print(f"分片大小: {batch_size}")
    if args.measure_memory:
        print("内存测量: 启用")
    print()
    
    if not os.path.exists(PREFILTER_EXE):
        print(f"错误: 可执行文件不存在: {PREFILTER_EXE}")
        print("   请先编译项目: cd build && make search_prefilter_streaming")
        sys.exit(1)
    
    for num_threads in thread_list:
        print()
        print(f"--- 运行线程数: {num_threads} ---")
        run_experiment_for_threads(
            PREFILTER_EXE, DATA_DIR, SCRIPT_DIR, num_threads,
            args.start_label, args.end_label, args.num_runs, batch_size, args.measure_memory
        )
    
    print()
    print("========================================")
    print("实验完成")
    print("========================================")
    print()
    print("结果文件:")
    for num_threads in thread_list:
        results_file = SCRIPT_DIR / f"prefilter_streaming_results_threads_{num_threads}.csv"
        print(f"  {results_file}")


if __name__ == "__main__":
    main()
