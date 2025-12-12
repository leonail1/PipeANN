#!/usr/bin/env python3
"""
SIFT1M Filtered Search 标签选择性实验

运行命令（输出同时显示在命令行和保存到 run.log）:
    python3 sift1m_filtered_search_demo.py 2>&1 | tee run.log
    python3 sift1m_filtered_search_demo.py --mode range --min-recall 98 --max-recall 99 2>&1 | tee run.log
    python3 sift1m_filtered_search_demo.py --mode converge 2>&1 | tee run.log

注意: GT文件需要预先计算，请在另一个终端运行:
    python3 /home/lzg/PipeANN/scripts/compute_gt_all.py
"""

import os
import sys
import subprocess
import re
import argparse
import time
from pathlib import Path

def wait_for_gt_file(gt_file: str, query_labels_file: str, label_id: int, 
                     check_interval: float = 2.0, first_wait: bool = False) -> bool:
    """
    等待GT文件出现
    
    Args:
        gt_file: GT文件路径
        query_labels_file: 查询标签文件路径
        label_id: 标签ID
        check_interval: 检查间隔（秒）
        first_wait: 是否是第一次等待（用于显示提示）
    
    Returns:
        True when files exist
    """
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

def main():
    parser = argparse.ArgumentParser(description='SIFT1M Filtered Search 标签选择性实验')
    parser.add_argument('--mode', type=str, default='range', choices=['range', 'converge'],
                        help='搜索模式: range=在recall区间内停止, converge=二分到收敛')
    parser.add_argument('--min-recall', type=float, default=98.0,
                        help='目标最小召回率 (默认: 98.0)')
    parser.add_argument('--max-recall', type=float, default=98.05,
                        help='目标最大召回率 (range模式使用, 默认: 99.0)')
    args = parser.parse_args()
    
    DATA_DIR = os.getenv("SIFT1M_DATA_DIR", "/mnt/ext4/lzg/sift1m_pq")
    DATA_FILE = f"{DATA_DIR}/bigann_1m.bin"
    QUERY_FILE = f"{DATA_DIR}/bigann_query.bin"
    INDEX_DIR = f"{DATA_DIR}/indices"
    INDEX_PREFIX = f"{INDEX_DIR}/sift1m_filtered"
    
    NUM_THREADS = 64
    R = 22
    L_BUILD = 96
    PQ_BYTES = 24
    MEMORY_GB = 32
    METRIC = "l2"
    NBR_TYPE = "pq"
    
    BEAM_WIDTH = 32
    K = 10
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    print("========================================")
    print("SIFT1M Filtered Search - 标签选择性实验")
    print("========================================")
    print(f"数据目录: {DATA_DIR}")
    print(f"索引目录: {INDEX_DIR}")
    print(f"搜索模式: {args.mode}")
    print(f"目标召回率: {args.min_recall}%" + (f" - {args.max_recall}%" if args.mode == 'range' else ' (收敛)'))
    print()
    
    if not os.path.isdir(DATA_DIR) or not os.path.exists(DATA_FILE) or not os.path.exists(QUERY_FILE):
        print("❌ 数据文件不存在，请检查路径配置")
        sys.exit(1)
    
    REQUIRED_EXECUTABLES = [
        "build/tests/build_disk_index",
        "build/tests/build_memory_index",
        "build/tests/search_disk_index_filtered",
        "build/tests/utils/compute_groundtruth",
        "build/tests/utils/gen_random_slice"
    ]
    
    for exe in REQUIRED_EXECUTABLES:
        if not os.path.exists(PROJECT_ROOT / exe):
            print(f"❌ 可执行文件不存在: {exe}")
            print("   请先编译项目")
            sys.exit(1)
    
    print("[1/4] 生成数据标签 (标签i有i%的向量包含, i=1~100)...")
    if not os.path.exists(f"{DATA_DIR}/data_labels.spmat"):
        subprocess.run([
            "python3", str(SCRIPT_DIR / "gen_random_labels.py"), "data-labels",
            "--output", f"{DATA_DIR}/data_labels.spmat",
            "--num-vectors", "1000000",
            "--num-labels", "100",
            "--seed", "42"
        ], stdout=subprocess.DEVNULL, check=True)
        print("  ✓ 完成")
    else:
        print("  ✓ 已存在")
    
    print()
    print(f"[2/4] 构建索引 (R={R}, L={L_BUILD}, PQ={PQ_BYTES}B)...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    if not os.path.exists(f"{INDEX_PREFIX}_disk.index"):
        subprocess.run([
            str(PROJECT_ROOT / "build/tests/build_disk_index"), "uint8",
            DATA_FILE, INDEX_PREFIX,
            str(R), str(L_BUILD), str(PQ_BYTES), str(MEMORY_GB), str(NUM_THREADS),
            METRIC, NBR_TYPE,
            "spmat", f"{DATA_DIR}/data_labels.spmat"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("  ✓ 完成")
    else:
        print("  ✓ 已存在")
    
    print()
    print("[3/4] 构建内存索引...")
    if not os.path.exists(f"{INDEX_PREFIX}_mem.index"):
        subprocess.run([
            str(PROJECT_ROOT / "build/tests/utils/gen_random_slice"), "uint8",
            DATA_FILE, f"{INDEX_PREFIX}_SAMPLE_RATE_0.01", "0.01"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        subprocess.run([
            str(PROJECT_ROOT / "build/tests/build_memory_index"), "uint8",
            f"{INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin",
            f"{INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin",
            f"{INDEX_PREFIX}_mem.index",
            "32", "64", "1.2", str(NUM_THREADS), METRIC
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("  ✓ 完成")
    else:
        print("  ✓ 已存在")
    
    print()
    print("[4/4] 开始标签选择性实验...")
    print(f"  目标: 对每个标签找到最优L值 (模式={args.mode}, Recall>={args.min_recall}%)")
    print()
    
    RESULTS_FILE = SCRIPT_DIR / "filtered_search_results.csv"
    
    completed_labels = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    label_id = int(line.split(',')[0])
                    completed_labels.add(label_id)
        print(f"  已完成的Label: {sorted(completed_labels)}")
    else:
        with open(RESULTS_FILE, 'w') as f:
            f.write("Label,Selectivity,L,Recall,AvgLat_us,QPS,P99Lat,MeanHops,MeanIOs\n")
    
    pending_labels = [i for i in range(1, 101) if i not in completed_labels]
    first_wait_done = False
    
    print()
    
    for LABEL_ID in range(1, 101):
        if LABEL_ID in completed_labels:
            print(f"\n  [{LABEL_ID}/100] 选择性={LABEL_ID}% 已完成，跳过")
            continue
        SELECTIVITY = LABEL_ID
        CURRENT_QUERY_LABELS = f"{DATA_DIR}/query_labels_{LABEL_ID}.spmat"
        CURRENT_FILTERED_GT = f"{DATA_DIR}/groundtruth_1m_filtered_label_{LABEL_ID}.bin"
        
        print(f"\n  [{LABEL_ID}/100] 选择性={SELECTIVITY}% ... ", flush=True)
        
        if not os.path.exists(CURRENT_QUERY_LABELS) or not os.path.exists(CURRENT_FILTERED_GT):
            wait_for_gt_file(CURRENT_FILTERED_GT, CURRENT_QUERY_LABELS, LABEL_ID,
                           check_interval=2.0, first_wait=not first_wait_done)
            first_wait_done = True
        else:
            print(f"    GT文件已存在，直接使用")
        
        L_MIN = 10
        L_MAX = 100
        TARGET_RECALL_MIN = args.min_recall
        TARGET_RECALL_MAX = args.max_recall
        MAX_LATENCY_US = 100000
        BEST_L = -1
        BEST_RECALL = 0
        BEST_LATENCY = 0
        BEST_QPS = 0
        BEST_P99LAT = 0
        BEST_MEAN_HOPS = 0
        BEST_MEAN_IOS = 0
        
        def test_L(L):
            result = subprocess.run([
                str(PROJECT_ROOT / "build/tests/search_disk_index_filtered"), "uint8",
                INDEX_PREFIX, "1", str(BEAM_WIDTH),
                QUERY_FILE, CURRENT_FILTERED_GT, str(K),
                METRIC, NBR_TYPE, "subset", CURRENT_QUERY_LABELS,
                "0", "10", str(L)
            ], capture_output=True, text=True)
            
            SEARCH_OUTPUT = result.stdout + result.stderr
            
            output_lines = SEARCH_OUTPUT.split('\n')
            table_lines = [line for line in output_lines if re.match(r'^\s*\d+\s+\d+\s+[0-9.]+', line) or 'Recall@10' in line or '===' in line]
            
            if table_lines:
                print("    === 搜索结果 ===")
                print('\n'.join(table_lines))
                print("    ===============\n")
            
            table_match = re.search(r'^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', SEARCH_OUTPUT, re.MULTILINE)
            
            if table_match:
                recall = float(table_match.group(8))
                avg_latency = float(table_match.group(4))
                qps = float(table_match.group(3))
                p99lat = float(table_match.group(5))
                mean_hops = float(table_match.group(6))
                mean_ios = float(table_match.group(7))
            else:
                recall = 0
                avg_latency = 0
                qps = 0
                p99lat = 0
                mean_hops = 0
                mean_ios = 0
            
            print(f"    提取到的指标: Recall={recall}, AvgLat={avg_latency}, QPS={qps}, P99Lat={p99lat}, Hops={mean_hops}, IOs={mean_ios}")
            return recall, avg_latency, qps, p99lat, mean_hops, mean_ios
        
        print(f"\n    开始搜索最优L值 (初始L_MAX={L_MAX})...")
        L_PREV = L_MIN
        
        while True:
            print(f"    先测试 L_MAX={L_MAX}...", flush=True)
            recall_max, lat_max, qps_max, p99lat_max, hops_max, ios_max = test_L(L_MAX)
            
            if lat_max >= MAX_LATENCY_US:
                print(f"    ✗ L_MAX={L_MAX} 延迟超限 ({lat_max}us >= {MAX_LATENCY_US}us)")
                if recall_max >= TARGET_RECALL_MIN:
                    print(f"    但已达到目标召回率 ({recall_max}% >= {TARGET_RECALL_MIN}%), 开始二分")
                    BEST_L = L_MAX
                    BEST_RECALL = recall_max
                    BEST_LATENCY = lat_max
                    BEST_QPS = qps_max
                    BEST_P99LAT = p99lat_max
                    BEST_MEAN_HOPS = hops_max
                    BEST_MEAN_IOS = ios_max
                    break
                else:
                    print(f"    且召回率不足 ({recall_max}% < {TARGET_RECALL_MIN}%), 停止搜索")
                    BEST_L = L_MAX
                    BEST_RECALL = recall_max
                    BEST_LATENCY = lat_max
                    BEST_QPS = qps_max
                    BEST_P99LAT = p99lat_max
                    BEST_MEAN_HOPS = hops_max
                    BEST_MEAN_IOS = ios_max
                    break
            
            if recall_max >= TARGET_RECALL_MIN:
                print(f"    ✓ L_MAX={L_MAX} 达到目标召回率 ({recall_max}% >= {TARGET_RECALL_MIN}%), 开始二分")
                BEST_L = L_MAX
                BEST_RECALL = recall_max
                BEST_LATENCY = lat_max
                BEST_QPS = qps_max
                BEST_P99LAT = p99lat_max
                BEST_MEAN_HOPS = hops_max
                BEST_MEAN_IOS = ios_max
                break
            else:
                recall_gap = TARGET_RECALL_MIN - recall_max
                if recall_gap > 50:
                    growth_factor = 2.0
                elif recall_gap > 20:
                    growth_factor = 1.5
                elif recall_gap > 10:
                    growth_factor = 1.3
                elif recall_gap > 5:
                    growth_factor = 1.2
                else:
                    growth_factor = 1.1
                
                L_PREV = L_MAX
                L_MIN = L_MAX + 1
                L_MAX_NEW = int(L_MAX * growth_factor)
                if L_MAX_NEW == L_MAX:
                    L_MAX_NEW = L_MAX + 100
                print(f"    → L_MAX={L_MAX} 召回率不足 ({recall_max}% < {TARGET_RECALL_MIN}%, 差距{recall_gap:.1f}%), 增长因子={growth_factor}, 新L_MAX={L_MAX_NEW}")
                L_MAX = L_MAX_NEW
        
        if BEST_RECALL >= TARGET_RECALL_MIN:
            print(f"\n    在 [{L_PREV}, {BEST_L}] 范围内二分搜索最小L...")
            L_MIN_BINARY = L_PREV
            L_MAX_BINARY = BEST_L
            
            while L_MIN_BINARY < L_MAX_BINARY:
                L_MID = (L_MIN_BINARY + L_MAX_BINARY) // 2
                
                print(f"    尝试 L={L_MID} (范围: {L_MIN_BINARY}-{L_MAX_BINARY})...", flush=True)
                recall, lat, qps, p99lat, hops, ios = test_L(L_MID)
                
                if args.mode == 'range':
                    if TARGET_RECALL_MIN <= recall <= TARGET_RECALL_MAX:
                        print(f"    ✓✓ 召回率在目标范围内 ({recall}% ∈ [{TARGET_RECALL_MIN}%, {TARGET_RECALL_MAX}%]), 找到最优L")
                        BEST_L = L_MID
                        BEST_RECALL = recall
                        BEST_LATENCY = lat
                        BEST_QPS = qps
                        BEST_P99LAT = p99lat
                        BEST_MEAN_HOPS = hops
                        BEST_MEAN_IOS = ios
                        break
                    elif recall > TARGET_RECALL_MAX:
                        print(f"    → 召回率过高 ({recall}% > {TARGET_RECALL_MAX}%), 减小L")
                        L_MAX_BINARY = L_MID - 1
                    elif recall >= TARGET_RECALL_MIN:
                        print(f"    ✓ 达到最低召回率 ({recall}% >= {TARGET_RECALL_MIN}%), 尝试减小L")
                        BEST_L = L_MID
                        BEST_RECALL = recall
                        BEST_LATENCY = lat
                        BEST_QPS = qps
                        BEST_P99LAT = p99lat
                        BEST_MEAN_HOPS = hops
                        BEST_MEAN_IOS = ios
                        L_MAX_BINARY = L_MID - 1
                    else:
                        print(f"    → 召回率不足 ({recall}% < {TARGET_RECALL_MIN}%), 增大L")
                        L_MIN_BINARY = L_MID + 1
                else:
                    if recall >= TARGET_RECALL_MIN:
                        print(f"    ✓ 达到最低召回率 ({recall}% >= {TARGET_RECALL_MIN}%), 尝试减小L")
                        BEST_L = L_MID
                        BEST_RECALL = recall
                        BEST_LATENCY = lat
                        BEST_QPS = qps
                        BEST_P99LAT = p99lat
                        BEST_MEAN_HOPS = hops
                        BEST_MEAN_IOS = ios
                        L_MAX_BINARY = L_MID - 1
                    else:
                        print(f"    → 召回率不足 ({recall}% < {TARGET_RECALL_MIN}%), 增大L")
                        L_MIN_BINARY = L_MID + 1
        
        print(f"\n  最终结果: L={BEST_L}, Recall={BEST_RECALL}%, Lat={BEST_LATENCY}us\n")
        with open(RESULTS_FILE, 'a') as f:
            f.write(f"{LABEL_ID},{SELECTIVITY},{BEST_L},{BEST_RECALL},{BEST_LATENCY},{BEST_QPS},{BEST_P99LAT},{BEST_MEAN_HOPS},{BEST_MEAN_IOS}\n")
        
        for tmp_file in [CURRENT_QUERY_LABELS, CURRENT_FILTERED_GT]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        
        print(f"  完成Label {LABEL_ID}，重启进程以避免内存泄漏...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    print()
    print("========================================")
    print("实验完成！")
    print("========================================")
    print(f"结果文件: {RESULTS_FILE}")
    print()
    print("CSV格式说明：")
    print("  Label        - 标签ID (1-100)")
    print("  Selectivity  - 选择性百分比 (%)")
    print("  L            - 最优搜索参数")
    print("  Recall       - 召回率 (%)")
    print("  AvgLat_us    - 平均延迟 (微秒)")
    print("  QPS          - 每秒查询数")
    print("  P99Lat       - P99延迟 (微秒)")
    print("  MeanHops     - 平均跳数")
    print("  MeanIOs      - 平均IO次数")
    print()

if __name__ == "__main__":
    main()
