#!/usr/bin/env python3
import struct
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_disk_index(index_path, return_stats=False):
    file_size = os.path.getsize(index_path)
    print(f"=== Disk Index Analysis: {index_path} ===")
    print(f"Total file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    SECTOR_LEN = 4096
    
    with open(index_path, 'rb') as f:
        nr, nc = struct.unpack('II', f.read(8))
        print(f"\nMetadata header: nr={nr}, nc={nc}")
        
        npoints = struct.unpack('Q', f.read(8))[0]
        data_dim = struct.unpack('Q', f.read(8))[0]
        entry_point = struct.unpack('Q', f.read(8))[0]
        max_node_len = struct.unpack('Q', f.read(8))[0]
        nnodes_per_sector = struct.unpack('Q', f.read(8))[0]
        npts_cur_shard = struct.unpack('Q', f.read(8))[0]
        label_size = struct.unpack('Q', f.read(8))[0] if nr >= 7 else 0
        
        print(f"\n=== Index Metadata (as stored) ===")
        print(f"Number of points:     {npoints:,}")
        print(f"Data dimension:       {data_dim}")
        print(f"Entry point:          {entry_point}")
        print(f"Max node length:      {max_node_len} bytes")
        print(f"Nodes per sector:     {nnodes_per_sector}")
        print(f"Points in shard:      {npts_cur_shard:,}")
        print(f"Label size field:     {label_size} bytes")
        
        print(f"\n=== Probe Node Structure ===")
        f.seek(SECTOR_LEN)
        first_sector = f.read(SECTOR_LEN)
        
        possible_nnbrs_offsets = []
        for offset in range(0, max_node_len, 4):
            val = struct.unpack('I', first_sector[offset:offset+4])[0]
            if 1 <= val <= 128:
                next_vals = []
                valid = True
                for i in range(min(val, 10)):
                    next_val = struct.unpack('I', first_sector[offset+4+i*4:offset+8+i*4])[0]
                    if next_val >= npoints:
                        valid = False
                        break
                    next_vals.append(next_val)
                if valid and len(next_vals) > 0:
                    possible_nnbrs_offsets.append((offset, val, next_vals))
        
        print(f"Looking for nnbrs field (uint32 with value 1-128 followed by valid node IDs):")
        for offset, val, neighbors in possible_nnbrs_offsets[:5]:
            print(f"  Offset {offset}: nnbrs={val}, first neighbors={neighbors[:5]}")
        
        if possible_nnbrs_offsets:
            best_offset, nnbrs_val, _ = possible_nnbrs_offsets[0]
            vector_size = best_offset
            print(f"\n=== Inferred Structure ===")
            print(f"Vector size:          {vector_size} bytes")
            print(f"sizeof(T):            {vector_size / data_dim:.1f} bytes")
            
            sizeof_T = vector_size // data_dim if data_dim > 0 else 4
            if sizeof_T == 4:
                data_type = "float32"
            elif sizeof_T == 1:
                data_type = "uint8"
            elif sizeof_T == 2:
                data_type = "int16/float16"
            else:
                data_type = f"unknown ({sizeof_T} bytes)"
            print(f"Inferred data type:   {data_type}")
            
            neighbor_list_size = max_node_len - vector_size - label_size
            actual_label_size = label_size
            if neighbor_list_size < 0:
                print(f"Note: neighbor_list_size negative, adjusting label_size...")
                actual_label_size = max_node_len - vector_size - (nnbrs_val + 1) * 4
                if actual_label_size < 0:
                    actual_label_size = 0
                neighbor_list_size = max_node_len - vector_size - actual_label_size
            
            range_val = neighbor_list_size // 4 - 1
            print(f"Max degree (range):   {range_val}")
            print(f"Neighbor list space:  {neighbor_list_size} bytes")
            print(f"Actual label size:    {actual_label_size} bytes")
        else:
            sizeof_T = 4
            vector_size = data_dim * sizeof_T
            neighbor_list_size = max_node_len - vector_size
            range_val = neighbor_list_size // 4 - 1
            actual_label_size = label_size
        
        if nnodes_per_sector > 0:
            bytes_per_write = SECTOR_LEN
        else:
            bytes_per_write = ((max_node_len + SECTOR_LEN - 1) // SECTOR_LEN) * SECTOR_LEN
        
        print(f"\n=== Disk Layout ===")
        print(f"Sector size:          {SECTOR_LEN} bytes")
        print(f"Nodes per sector:     {nnodes_per_sector}")
        print(f"Used space/sector:    {nnodes_per_sector * max_node_len} bytes")
        print(f"Wasted padding:       {SECTOR_LEN - nnodes_per_sector * max_node_len} bytes/sector")
        
        metadata_size = SECTOR_LEN
        num_sectors = (npoints + nnodes_per_sector - 1) // nnodes_per_sector if nnodes_per_sector > 0 else npoints
        data_size = num_sectors * bytes_per_write
        
        print(f"\n=== Space Breakdown ===")
        print(f"Metadata sector:      {metadata_size:,} bytes ({metadata_size/1024:.2f} KB)")
        print(f"Number of sectors:    {num_sectors:,}")
        print(f"Data sectors:         {data_size:,} bytes ({data_size/1024/1024:.2f} MB)")
        print(f"Total expected:       {metadata_size + data_size:,} bytes")
        print(f"Actual file size:     {file_size:,} bytes")
        
        vector_total = npoints * vector_size
        neighbor_total = npoints * neighbor_list_size
        label_total = npoints * actual_label_size
        padding_per_sector = SECTOR_LEN - (nnodes_per_sector * max_node_len)
        total_padding = num_sectors * padding_per_sector
        
        print(f"\n=== Detailed Space Usage ===")
        print(f"{'Component':<25} {'Size':>15} {'MB':>10} {'%':>8}")
        print("-" * 60)
        print(f"{'Vector data':<25} {vector_total:>15,} {vector_total/1024/1024:>10.2f} {vector_total*100/file_size:>7.1f}%")
        print(f"{'Neighbor lists':<25} {neighbor_total:>15,} {neighbor_total/1024/1024:>10.2f} {neighbor_total*100/file_size:>7.1f}%")
        if actual_label_size > 0:
            print(f"{'Labels':<25} {label_total:>15,} {label_total/1024/1024:>10.2f} {label_total*100/file_size:>7.1f}%")
        print(f"{'Sector padding':<25} {total_padding:>15,} {total_padding/1024/1024:>10.2f} {total_padding*100/file_size:>7.1f}%")
        print(f"{'Metadata':<25} {metadata_size:>15,} {metadata_size/1024/1024:>10.4f} {metadata_size*100/file_size:>7.3f}%")
        print("-" * 60)
        accounted = vector_total + neighbor_total + label_total + total_padding + metadata_size
        print(f"{'Total accounted':<25} {accounted:>15,} {accounted/1024/1024:>10.2f} {accounted*100/file_size:>7.1f}%")
        
        raw_data_size = npoints * data_dim * sizeof_T
        print(f"\n=== Comparison with Raw Data ===")
        print(f"Raw vector data:      {raw_data_size:,} bytes ({raw_data_size/1024/1024:.2f} MB)")
        print(f"Index file size:      {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"Amplification:        {file_size / raw_data_size:.2f}x")
        
        print(f"\n=== Why is the index larger? ===")
        overhead = file_size - raw_data_size
        print(f"Total overhead:       {overhead:,} bytes ({overhead/1024/1024:.2f} MB)")
        print(f"\nBreakdown of overhead:")
        print(f"  1. Neighbor lists:  {neighbor_total:,} bytes ({neighbor_total/1024/1024:.2f} MB)")
        print(f"     - Stores graph connectivity (max {range_val} neighbors per node)")
        if actual_label_size > 0:
            print(f"  2. Labels:          {label_total:,} bytes ({label_total/1024/1024:.2f} MB)")
            print(f"     - Stores {actual_label_size} bytes of label data per node")
        print(f"  3. Sector padding:  {total_padding:,} bytes ({total_padding/1024/1024:.2f} MB)")
        print(f"     - Alignment waste for {SECTOR_LEN}-byte sector alignment")
        print(f"  4. Metadata:        {metadata_size:,} bytes")
        
        print(f"\n=== Sample Node Analysis ===")
        total_edges = 0
        sample_count = min(1000, npoints)
        degree_dist = {}
        
        for i in range(sample_count):
            if nnodes_per_sector > 0:
                sector_idx = i // nnodes_per_sector
                node_in_sector = i % nnodes_per_sector
                node_offset = SECTOR_LEN + sector_idx * SECTOR_LEN + node_in_sector * max_node_len
            else:
                node_offset = SECTOR_LEN + i * bytes_per_write
            
            f.seek(node_offset + vector_size)
            nnbrs = struct.unpack('I', f.read(4))[0]
            total_edges += nnbrs
            degree_dist[nnbrs] = degree_dist.get(nnbrs, 0) + 1
            
            if i < 5:
                print(f"  Node {i}: {nnbrs} neighbors")
        
        avg_degree = total_edges / sample_count
        print(f"  ...")
        print(f"  Average degree ({sample_count} samples): {avg_degree:.2f}")
        print(f"  Max degree in samples: {max(degree_dist.keys())}")
        print(f"  Min degree in samples: {min(degree_dist.keys())}")
        
        actual_neighbor_bytes = npoints * (avg_degree * 4 + 4)
        wasted_in_neighbors = neighbor_total - actual_neighbor_bytes
        print(f"\n=== Estimated Neighbor List Efficiency ===")
        print(f"Max neighbor space:   {neighbor_total:,} bytes")
        print(f"Est. actual used:     {int(actual_neighbor_bytes):,} bytes")
        print(f"Est. wasted:          {int(wasted_in_neighbors):,} bytes ({wasted_in_neighbors/1024/1024:.2f} MB)")
        
        if return_stats:
            return {
                'file_size': file_size,
                'npoints': npoints,
                'data_dim': data_dim,
                'max_node_len': max_node_len,
                'nnodes_per_sector': nnodes_per_sector,
                'range': range_val,
                'vector_total': vector_total,
                'neighbor_total': neighbor_total,
                'label_total': label_total,
                'total_padding': total_padding,
                'metadata_size': metadata_size,
                'raw_data_size': raw_data_size,
                'degree_dist': degree_dist,
                'avg_degree': avg_degree,
                'actual_neighbor_bytes': actual_neighbor_bytes,
                'wasted_in_neighbors': wasted_in_neighbors,
            }

def plot_disk_index_analysis(index_path, output_dir=None):
    stats = analyze_disk_index(index_path, return_stats=True)
    if stats is None:
        print("Failed to analyze index")
        return
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'disk_index_plots')
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    components = ['Vector Data', 'Neighbor Lists', 'Sector Padding', 'Metadata']
    sizes = [stats['vector_total'], stats['neighbor_total'], 
             stats['total_padding'], stats['metadata_size']]
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    wedges, texts, autotexts = ax1.pie(sizes, labels=components, autopct='%1.1f%%',
                                        colors=colors, explode=[0.02]*len(components))
    index_without_label = stats['file_size'] - stats['label_total']
    ax1.set_title(f'Disk Index Space Breakdown (Without Labels)\n({index_without_label/1024/1024:.1f} MB total)', fontsize=14)
    plt.setp(autotexts, size=10, weight='bold')
    fig1.savefig(os.path.join(output_dir, 'space_breakdown_pie.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    index_without_label_mb = index_without_label / 1024 / 1024
    label_mb = stats['label_total'] / 1024 / 1024
    x = ['Disk Index\n(Without Labels)', 'Labels']
    heights = [index_without_label_mb, label_mb]
    colors2 = ['#3498db', '#e74c3c']
    bars = ax2.bar(x, heights, color=colors2, edgecolor='black', width=0.5)
    for bar, h in zip(bars, heights):
        ax2.text(bar.get_x() + bar.get_width()/2, h + max(heights)*0.02,
                f'{h:.1f} MB', ha='center', va='bottom', fontsize=12, weight='bold')
    ax2.set_ylabel('Size (MB)', fontsize=12)
    ax2.set_title('Disk Index vs Labels Size', fontsize=14)
    ax2.set_ylim(0, max(heights) * 1.2)
    fig2.savefig(os.path.join(output_dir, 'index_vs_labels.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\n=== Generated Plots ===")
    plots = ['space_breakdown_pie.png', 'index_vs_labels.png']
    for p in plots:
        print(f"  {os.path.join(output_dir, p)}")
    
    return stats


if __name__ == "__main__":
    plot_mode = '--plot' in sys.argv
    if plot_mode:
        sys.argv.remove('--plot')
    
    if len(sys.argv) < 2:
        index_path = "/mnt/ext4/lzg/sift1m_pq/indices/sift1m_filtered_disk.index"
    else:
        index_path = sys.argv[1]
    
    if plot_mode:
        plot_disk_index_analysis(index_path)
    else:
        analyze_disk_index(index_path)