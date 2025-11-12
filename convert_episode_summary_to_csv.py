#!/usr/bin/env python3
"""
Convert episode_summary.jsonl to TensorBoard-style CSV format
"""
import json
import pandas as pd
import os
from datetime import datetime

def convert_episode_summary_to_csv(jsonl_path, output_dir='reward_learning_curves'):
    """
    Convert episode_summary.jsonl to TensorBoard CSV format
    
    Args:
        jsonl_path: Path to episode_summary.jsonl
        output_dir: Output directory for CSV files
    """
    print(f"Converting {jsonl_path} to CSV format...")
    
    # Read JSONL file
    episodes = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Use first episode timestamp as base (or use current time)
    base_time = datetime.now().timestamp()
    
    # Detect what reward components are available
    sample_ep = episodes[0] if episodes else {}
    has_reward_components = 'reward_components' in sample_ep
    
    # List of metrics to export
    metrics_to_export = {
        'episode_reward': ('total_reward', 'episode_reward'),
    }
    
    # Add reward components if available
    if has_reward_components:
        reward_comp_keys = sample_ep.get('reward_components', {}).keys()
        for key in reward_comp_keys:
            metrics_to_export[key] = (f'reward_components.{key}', key)
    
    # Create CSV for each metric
    output_paths = {}
    
    for metric_name, (json_path, csv_name) in metrics_to_export.items():
        # Create output directory
        metric_dir = os.path.join(output_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        
        # Extract data for this metric
        data = {
            'Wall time': [],
            'Step': [],
            'Value': []
        }
        
        for ep in episodes:
            episode_num = ep['episode']
            
            # Navigate to the value using dot notation
            value = ep
            for part in json_path.split('.'):
                value = value.get(part, 0.0) if isinstance(value, dict) else 0.0
            
            data['Wall time'].append(base_time + episode_num)
            data['Step'].append(episode_num)
            data['Value'].append(value if value is not None else 0.0)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV (matching TensorBoard export format)
        output_path = os.path.join(metric_dir, f'run_.-tag-{csv_name}.csv')
        df.to_csv(output_path, index=False)
        output_paths[metric_name] = output_path
        
        print(f"Saved {metric_name} to: {output_path}")
        print(f"   Range: [{df['Value'].min():.2f}, {df['Value'].max():.2f}]")
        print(f"   Mean: {df['Value'].mean():.2f}")
    
    # Also create summary statistics
    print(f"\nTraining Summary:")
    print(f"   Total episodes: {len(episodes)}")
    
    # Calculate moving average for total reward
    if 'episode_reward' in output_paths:
        df = pd.read_csv(output_paths['episode_reward'])
        window_size = min(100, len(df) // 10)
        if window_size > 0:
            df['MA'] = df['Value'].rolling(window=window_size).mean()
            print(f"   Final {window_size}-episode average: {df['MA'].iloc[-1]:.2f}")
    
    return output_paths


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        # Default to latest training run
        jsonl_path = 'results/0511-10:50/episode_summary.jsonl'
    
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        print(f"Usage: python {sys.argv[0]} [path/to/episode_summary.jsonl]")
        return 1
    
    output_path = convert_episode_summary_to_csv(jsonl_path)
    
    print(f"\nNow you can run your vldb_demo_graphs notebook!")
    print(f"   The CSV file is ready at: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

