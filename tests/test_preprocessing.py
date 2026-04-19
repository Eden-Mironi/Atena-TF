#!/usr/bin/env python3
"""
Simple test to verify preprocessing is working
"""
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, '.')

# Import the Repository class
from gym_atena.reactida.utils.utilities import Repository

print("Testing Preprocessing Function")
print("=" * 60)

# Create test dataframe with NaN values (like real data)
test_df = pd.DataFrame({
    'length': [100, np.nan, 200, np.nan, 150],
    'tcp_srcport': [80, np.nan, 443, np.nan, 22],
    'tcp_dstport': [443, 80, np.nan, np.nan, 22],
    'ip_src': ['192.168.1.1', np.nan, '10.0.0.1', np.nan, '172.16.0.1'],
    'highest_layer': ['TCP', np.nan, 'HTTP', np.nan, 'SSH']
})

print("\nBEFORE Preprocessing:")
print(f"  Total values: {test_df.size}")
print(f"  NaN count: {test_df.isna().sum().sum()}")
print(f"  Length column: {test_df['length'].tolist()}")
print(f"  tcp_srcport column: {test_df['tcp_srcport'].tolist()}")

# Apply preprocessing
processed_df = Repository._preprocess_dataframe(test_df)

print("\nAFTER Preprocessing:")
print(f"  Total values: {processed_df.size}")
print(f"  NaN count: {processed_df.isna().sum().sum()}")
print(f"  Length column: {processed_df['length'].tolist()}")
print(f"  tcp_srcport column: {processed_df['tcp_srcport'].tolist()}")
print(f"  ip_src column: {processed_df['ip_src'].tolist()}")
print(f"  highest_layer column: {processed_df['highest_layer'].tolist()}")

# Verify
if processed_df.isna().sum().sum() == 0:
    print("\nSUCCESS: All NaN values filled!")
else:
    print(f"\nFAILED: Still have {processed_df.isna().sum().sum()} NaN values!")
    print(f"   Remaining NaN by column:")
    for col in processed_df.columns:
        nan_count = processed_df[col].isna().sum()
        if nan_count > 0:
            print(f"     - {col}: {nan_count} NaN values")

print("\n" + "=" * 60)

