#!/usr/bin/env python3
"""
Check which columns have NaN values in the raw data
"""
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '.')

# Load a sample dataset
df = pd.read_csv('gym_atena/reactida/raw_datasets_with_pkt_nums/1.tsv', sep='\t', index_col=0)

print("NaN Analysis for Dataset 1")
print("=" * 60)
print(f"\nTotal NaN values: {df.isna().sum().sum()}")
print(f"\nColumns with NaN (sorted by count):")
print("-" * 60)

nan_counts = df.isna().sum().sort_values(ascending=False)
for col, count in nan_counts.items():
    if count > 0:
        percentage = (count / len(df)) * 100
        print(f"  {col:30s}: {count:6d} NaN ({percentage:5.1f}%)")

print("\n" + "=" * 60)
print("\n📋 Columns we're currently preprocessing:")
print("  - tcp_dstport, tcp_srcport, length, tcp_stream")
print("  - ip_src, ip_dst")  
print("  - highest_layer, info_line")
print("  - eth_src, eth_dst")

print("\nColumns we're NOT preprocessing (but have NaN):")
not_handled = []
handled_cols = {'tcp_dstport', 'tcp_srcport', 'length', 'tcp_stream', 'ip_src', 'ip_dst', 'highest_layer', 'info_line', 'eth_src', 'eth_dst'}
for col, count in nan_counts.items():
    if count > 0 and col not in handled_cols:
        not_handled.append(col)
        print(f"  - {col}")

print(f"\nTotal unhandled columns with NaN: {len(not_handled)}")

