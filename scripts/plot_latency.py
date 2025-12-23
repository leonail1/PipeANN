#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt

# 读取CSV文件
csv_path = '/home/lzg/PipeANN/scripts/filtered_search_results_r22_pq24_t1.csv'
selectivity = []
avg_latency = []
p99_latency = []
L_values = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        selectivity.append(float(row['Selectivity']))
        avg_latency.append(float(row['AvgLat_us']) / 1000)  # 微秒转毫秒
        p99_latency.append(float(row['P99Lat']) / 1000)      # 微秒转毫秒
        L_values.append(float(row['L']))

# 创建高分辨率图像
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 绘制平均延迟和P99延迟曲线
ax.plot(selectivity, avg_latency, 'b-o', markersize=3, linewidth=1.5, label='Avg Latency')
ax.plot(selectivity, p99_latency, 'r-s', markersize=3, linewidth=1.5, label='P99 Latency')

# 创建右侧Y轴绘制L值（显示在图的上半部分）
ax2 = ax.twinx()
ax2.plot(selectivity, L_values, 'g--^', markersize=4, linewidth=1.5, label='L', alpha=0.8)
ax2.set_ylabel('L', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-max(L_values), max(L_values) * 1.1)

# 在第一个点和每逢整十的点标注L值
for i, sel in enumerate(selectivity):
    if sel == 1 or sel % 10 == 0:
        ax2.annotate(f'{int(L_values[i])}', (sel, L_values[i]), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='darkgreen')

# 设置坐标轴标签
ax.set_xlabel('Selective Factor (%)', fontsize=12)
ax.set_ylabel('Latency (ms)', fontsize=12)

# 在10ms处画一条高对比度参考线
ax.axhline(y=10, color='lime', linestyle='--', linewidth=2, label='10ms Threshold')

# 设置标题
ax.set_title('Filtered Search Latency vs Selectivity (thread num:1)', fontsize=14)

# 添加峰值内存占用标注
ax.text(0.95, 0.95, 'Peak Memory: 30MB', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 添加图例（合并两个轴的图例）
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.95, 0.85), fontsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 保存高分辨率图像
output_path = '/home/lzg/PipeANN/scripts/latency_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图像已保存至: {output_path}')

# 显示图像（如果在交互环境中）
plt.show()