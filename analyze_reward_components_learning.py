#!/usr/bin/env python3
"""
Analyze learning curves for individual reward components
Suggested by professor to identify convergence issues in specific components
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# import seaborn as sns  # Not used
from collections import defaultdict
import argparse

def load_tf_training_data(results_path):
    """Load TensorFlow training data from results directory"""
    training_data = {
        'steps': [],
        'episodes': [],
        'total_rewards': [],
        'reward_components': defaultdict(list)
    }
    
    # Try to load from reward_analysis.jsonl first (has detailed step-by-step breakdown)
    reward_analysis_path = os.path.join(results_path, 'reward_analysis.jsonl')
    if os.path.exists(reward_analysis_path):
        print(f"Loading TF step-by-step data from: {reward_analysis_path}")
        with open(reward_analysis_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    step = data.get('step', line_num)
                    episode = data.get('episode', 0)
                    total_reward = data.get('reward', 0)
                    
                    training_data['steps'].append(step)
                    training_data['episodes'].append(episode)
                    training_data['total_rewards'].append(total_reward)
                    
                    # Extract individual reward components from reward_breakdown
                    reward_breakdown = data.get('reward_breakdown', {})
                    if isinstance(reward_breakdown, dict):
                        for component, value in reward_breakdown.items():
                            if isinstance(value, (int, float)):
                                training_data['reward_components'][component].append(value)
                            else:
                                training_data['reward_components'][component].append(0)
                                
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
    else:
        # Fallback to episode_summary.jsonl with aggregated data
        episode_summary_path = os.path.join(results_path, 'episode_summary.jsonl')
        if os.path.exists(episode_summary_path):
            print(f"Loading TF episode data from: {episode_summary_path}")
            with open(episode_summary_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        episode = data.get('episode', line_num)
                        total_reward = data.get('total_reward', 0)
                        
                        training_data['episodes'].append(episode)
                        training_data['total_rewards'].append(total_reward)
                        
                        # Extract from reward_summary (episode averages)
                        reward_summary = data.get('reward_summary', {})
                        if isinstance(reward_summary, dict):
                            for key, value in reward_summary.items():
                                if key.startswith('avg_') and isinstance(value, (int, float)):
                                    component_name = key.replace('avg_', '')
                                    training_data['reward_components'][component_name].append(value)
                                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {e}")
                        continue
    
    # Convert to numpy arrays for easier processing
    for component in training_data['reward_components']:
        training_data['reward_components'][component] = np.array(training_data['reward_components'][component])
    
    training_data['steps'] = np.array(training_data['steps']) if training_data['steps'] else np.array([])
    training_data['episodes'] = np.array(training_data['episodes'])
    training_data['total_rewards'] = np.array(training_data['total_rewards'])
    
    return training_data

def smooth_data(data, window_size=50):
    """Apply moving average smoothing"""
    if len(data) < window_size:
        return data
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def calculate_convergence_metrics(data, window_size=100):
    """Calculate convergence metrics for a data series"""
    if len(data) < window_size * 2:
        return {'converged': False, 'final_mean': np.mean(data), 'stability': np.std(data)}
    
    # Calculate stability in final portion
    final_portion = data[-window_size:]
    final_mean = np.mean(final_portion)
    final_std = np.std(final_portion)
    
    # Calculate trend in final portion
    x = np.arange(len(final_portion))
    trend = np.polyfit(x, final_portion, 1)[0]  # Linear slope
    
    # Define convergence criteria
    stability_threshold = 0.1  # Standard deviation threshold
    trend_threshold = 0.01     # Trend threshold
    
    converged = (abs(trend) < trend_threshold) and (final_std < stability_threshold)
    
    return {
        'converged': converged,
        'final_mean': final_mean,
        'final_std': final_std,
        'trend': trend,
        'stability': final_std
    }

def plot_component_learning_curves(training_data, output_dir, smooth_window=50):
    """Create detailed learning curves for each reward component"""
    
    # Get all reward components
    components = list(training_data['reward_components'].keys())
    components = [c for c in components if len(training_data['reward_components'][c]) > 0]
    
    if not components:
        print("Warning: No reward components found in training data")
        return {}  # Return empty dict instead of None
    
    # Choose x-axis: use steps if available, otherwise episodes
    if len(training_data['steps']) > 0:
        x_data = training_data['steps']
        x_label = 'Training Step'
        print(f"Using step-by-step data (x-axis: {len(x_data)} steps)")
    else:
        x_data = training_data['episodes'] 
        x_label = 'Episode'
        print(f"Using episode data (x-axis: {len(x_data)} episodes)")
    
    print(f"Found {len(components)} reward components: {components}")
    
    # Create subplot grid
    n_components = len(components) + 1  # +1 for total reward
    cols = 3
    rows = (n_components + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle('Learning Curves by Reward Component', fontsize=16, y=0.98)
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Plot total reward first
    total_rewards = training_data['total_rewards']
    
    if len(total_rewards) > 0:
        ax = axes[0]
        ax.plot(x_data, total_rewards, alpha=0.3, color='blue', linewidth=0.5)
        
        # Add smoothed line
        if len(total_rewards) > smooth_window:
            smoothed_rewards = smooth_data(total_rewards, smooth_window)
            smoothed_x = x_data[smooth_window-1:]
            ax.plot(smoothed_x, smoothed_rewards, color='blue', linewidth=2, label='Total Reward (smoothed)')
        
        ax.set_title('Total Reward')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate convergence
        conv_metrics = calculate_convergence_metrics(total_rewards)
        status = "Converged" if conv_metrics['converged'] else "Not Converged"
        ax.text(0.02, 0.95, f"{status}\nFinal: {conv_metrics['final_mean']:.2f}±{conv_metrics['final_std']:.2f}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot individual components
    convergence_summary = {}
    
    for i, component in enumerate(components):
        ax_idx = i + 1
        if ax_idx >= len(axes):
            break
            
        ax = axes[ax_idx]
        component_data = training_data['reward_components'][component]
        
        if len(component_data) == 0:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{component} (No Data)')
            continue
        
        # Ensure same length as x_data
        min_len = min(len(x_data), len(component_data))
        x_subset = x_data[:min_len]
        component_subset = component_data[:min_len]
        
        # Plot raw data
        ax.plot(x_subset, component_subset, alpha=0.3, linewidth=0.5)
        
        # Add smoothed line
        if len(component_subset) > smooth_window:
            smoothed_data = smooth_data(component_subset, smooth_window)
            smoothed_x = x_subset[smooth_window-1:]
            ax.plot(smoothed_x, smoothed_data, linewidth=2, label=f'{component} (smoothed)')
        
        ax.set_title(f'{component}')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Calculate convergence metrics
        conv_metrics = calculate_convergence_metrics(component_subset)
        convergence_summary[component] = conv_metrics
        
        status = "Converged" if conv_metrics['converged'] else "Not Converged"
        ax.text(0.02, 0.95, f"{status}\nFinal: {conv_metrics['final_mean']:.2f}±{conv_metrics['final_std']:.2f}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(components) + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'reward_components_learning_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved component learning curves: {plot_path}")
    
    plt.show()
    
    # Print convergence summary
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    converged_components = []
    not_converged_components = []
    
    for component, metrics in convergence_summary.items():
        status = "CONVERGED" if metrics['converged'] else "NOT CONVERGED"
        print(f"{component:20s}: {status:15s} | Final: {metrics['final_mean']:8.3f} ± {metrics['final_std']:6.3f} | Trend: {metrics['trend']:8.3f}")
        
        if metrics['converged']:
            converged_components.append(component)
        else:
            not_converged_components.append(component)
    
    print("\n" + "-"*60)
    print(f"CONVERGED COMPONENTS ({len(converged_components)}): {', '.join(converged_components)}")
    print(f"UNSTABLE COMPONENTS ({len(not_converged_components)}): {', '.join(not_converged_components)}")
    
    if not_converged_components:
        print(f"\nPROFESSOR'S INSIGHT CONFIRMED:")
        print(f"   Components causing instability: {', '.join(not_converged_components)}")
        print(f"   Focus optimization efforts on these components!")
    else:
        print(f"\nAll individual components appear to be converging!")
        print(f"   The total reward instability may be due to component interactions.")
    
    return convergence_summary

def main():
    parser = argparse.ArgumentParser(description='Analyze reward component learning curves')
    parser.add_argument('--tf-path', required=True, help='Path to TensorFlow training results')
    parser.add_argument('--output', default='conclusions', help='Output directory for plots')
    parser.add_argument('--smooth-window', type=int, default=50, help='Window size for smoothing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("REWARD COMPONENT LEARNING CURVE ANALYSIS")
    print("="*60)
    print(f"TensorFlow Results: {args.tf_path}")
    print(f"Output Directory: {args.output}")
    print(f"Smoothing Window: {args.smooth_window}")
    
    # Load training data
    print(f"\nLoading training data...")
    training_data = load_tf_training_data(args.tf_path)
    
    if len(training_data['episodes']) == 0:
        print("No training data found!")
        return
    
    print(f"Loaded {len(training_data['episodes'])} episodes")
    print(f"Found {len(training_data['reward_components'])} reward components")
    
    # Generate component learning curves
    print(f"\nGenerating component learning curves...")
    convergence_summary = plot_component_learning_curves(training_data, args.output, args.smooth_window)
    
    # Save convergence summary
    summary_path = os.path.join(args.output, 'convergence_analysis.txt')
    with open(summary_path, 'w') as f:
        f.write("REWARD COMPONENT CONVERGENCE ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        if convergence_summary:
            for component, metrics in convergence_summary.items():
                status = "CONVERGED" if metrics['converged'] else "NOT_CONVERGED"
                f.write(f"{component}: {status} | Final: {metrics['final_mean']:.3f}±{metrics['final_std']:.3f}\n")
        else:
            f.write("No reward components found in training data.\n")
    
    print(f"\nAnalysis complete! Check {args.output}/ for detailed results.")

if __name__ == "__main__":
    main()
