#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from gym_atena.reactida.utils.utilities import Repository
import pandas as pd
import numpy as np

# Test with sample data
test_df = pd.DataFrame({
    'length': [100, np.nan, 200],
    'tcp_srcport': [80, np.nan, 443],
    'packet_number': [1, 2, np.nan],
    'ip_src': ['192.168.1.1', np.nan, '10.0.0.1'],
    'eth_dst': ['aa:bb:cc:dd:ee:ff', np.nan, '11:22:33:44:55:66'],
    'highest_layer': ['TCP', np.nan, 'HTTP'],
    'info_line': ['Some info', np.nan, 'More info']
})

print("BEFORE:")
print(f"  NaN count: {test_df.isna().sum().sum()}")

processed = Repository._preprocess_dataframe(test_df)

print("AFTER:")
print(f"  NaN count: {processed.isna().sum().sum()}")
print(f"\nValues:")
for col in processed.columns:
    print(f"  {col}: {processed[col].tolist()}")

if processed.isna().sum().sum() == 0:
    print("\nSUCCESS: All NaN filled!")
else:
    print(f"\nFAILED: {processed.isna().sum().sum()} NaN remaining")

