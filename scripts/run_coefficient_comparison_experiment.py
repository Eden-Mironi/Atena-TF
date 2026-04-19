#!/usr/bin/env python3
"""
Comprehensive coefficient comparison experiment:
- Train 4 models with different coefficient sets for 2000 training steps each
- Run comparison analysis on each model vs master
- Determine which configuration best matches the master
"""

import os
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import time

# 4 coefficient configurations to test
COEFFICIENT_CONFIGS = [
    {
        "name": "first_config",
        "description": "Poor performance config",
        "kl_coeff": 2.8,
        "compaction_coeff": 2.5,
        "diversity_coeff": 6.0,
        "humanity_coeff": 4.8,
        "expected": "Unknown performance"
    },
    {
        "name": "second_config",
        "description": "Excellent performance config 1",
        "kl_coeff": 2.2,
        "compaction_coeff": 2.0,
        "diversity_coeff": 8.0,
        "humanity_coeff": 4.5,
        "expected": "Excellent performance (avg reward: 5.411)"
    },
    {
        "name": "third_config",  
        "description": "Excellent performance config 2",
        "kl_coeff": 1.0,
        "compaction_coeff": 1.0,
        "diversity_coeff": 2.0,
        "humanity_coeff": 1.0,
        "expected": "Excellent performance (avg reward: 5.249)"
    },
    {
        "name": "baseline",
        "description": "Baseline config",
        "kl_coeff": 1.0,
        "compaction_coeff": 1.0,
        "diversity_coeff": 1.0,
        "humanity_coeff": 1.0,
        "expected": "Baseline configuration"
    }
]

def backup_original_config():
    """Backup the original config file"""
    config_path = Path("Configuration/config.py")
    backup_path = Path("Configuration/config_original_backup.py")
    
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to {backup_path}")
    else:
        print(f"Config backup already exists at {backup_path}")

def restore_original_config():
    """Restore the original config file"""
    config_path = Path("Configuration/config.py")
    backup_path = Path("Configuration/config_original_backup.py")
    
    if backup_path.exists():
        shutil.copy2(backup_path, config_path)
        print(f"Restored original config from {backup_path}")
    else:
        print(f"No backup found at {backup_path}")

def update_config_coefficients(config):
    """Update the config file with new coefficient values"""
    config_path = Path("Configuration/config.py")
    
    print(f"Updating coefficients for {config['name']}...")
    
    # Read the current config file
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Replace coefficient values while preserving indentation
    replacements = {
        'kl_coeff': config["kl_coeff"],
        'compaction_coeff': config["compaction_coeff"], 
        'diversity_coeff': config["diversity_coeff"],
        'humanity_coeff': config["humanity_coeff"]
    }
    
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        original_line = line
        
        # Check if this line contains one of our target coefficients
        for coeff_name, coeff_value in replacements.items():
            if (stripped.startswith(f'{coeff_name} =') and 
                not stripped.startswith('#')):
                
                # Preserve the original indentation
                indent = line[:len(line) - len(line.lstrip())]
                new_line = f"{indent}{coeff_name} = {coeff_value}\n"
                original_line = new_line
                print(f"  Updated: {coeff_name} = {coeff_value}")
                break
        
        updated_lines.append(original_line)
    
    # Write back to file
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)
    
    # Verify the config file is valid Python by trying to import it
    try:
        import importlib
        import sys
        
        # Remove from cache if already imported
        if 'Configuration.config' in sys.modules:
            del sys.modules['Configuration.config']
            
        # Try to import the updated config
        import Configuration.config as test_cfg
        
        # Verify the values were set correctly
        actual_values = {
            'kl_coeff': getattr(test_cfg, 'kl_coeff', None),
            'compaction_coeff': getattr(test_cfg, 'compaction_coeff', None),
            'diversity_coeff': getattr(test_cfg, 'diversity_coeff', None),
            'humanity_coeff': getattr(test_cfg, 'humanity_coeff', None)
        }
        
        # Check if values match expectations
        for coeff_name, expected_value in replacements.items():
            actual_value = actual_values.get(coeff_name)
            if actual_value != expected_value:
                print(f"Warning: {coeff_name} = {actual_value}, expected {expected_value}")
            else:
                print(f"Verified: {coeff_name} = {actual_value}")
                
        print(f"Config file successfully updated and verified for {config['name']}")
        
    except Exception as e:
        print(f"Config file validation failed: {e}")
        print(f"This will cause training to fail!")
        raise e

def run_training(config, steps=2000):
    """Run training with the specified configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"COEFF_EXPERIMENT_{config['name']}_{timestamp}"
    
    print(f"\nSTARTING TRAINING FOR: {config['name']}")
    print(f"Output directory: results/{output_dir}")
    print(f"🔢 Training steps: {steps}")
    print(f"⚙️  Coefficients: kl={config['kl_coeff']}, comp={config['compaction_coeff']}, div={config['diversity_coeff']}, hum={config['humanity_coeff']}")
    print(f"⏳ This experiment will run to completion before the next one starts...")
    
    # Run training
    try:
        
        cmd = [
            "python", "train_with_decay.py",
            "--steps", str(steps),
            "--outdir", f"results/{output_dir}"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"⏱️  Training started at: {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            print(f"🏁 Finished at: {datetime.now().strftime('%H:%M:%S')}")
            
            # Find the actual output directory
            # Since we're using --outdir, the directory should be exactly where we specified
            actual_dir = Path(f"results/{output_dir}")
            
            # If that doesn't exist, fall back to searching
            if not actual_dir.exists():
                results_dir = Path("results")
                for dir_path in results_dir.iterdir():
                    if dir_path.is_dir() and output_dir in dir_path.name:
                        actual_dir = dir_path
                        break
                    
            if actual_dir:
                config['output_path'] = str(actual_dir)
                print(f"Results saved to: {actual_dir}")
                print(f"{config['name']} experiment COMPLETE - ready for next experiment")
                return True
            else:
                print(f"Could not find output directory matching {output_dir}")
                return False
        else:
            print(f"TRAINING FAILED with return code {result.returncode}")
            if result.stdout:
                print(f"stdout: {result.stdout}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Training timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"Training failed with exception: {e}")
        return False

def run_comparison(config):
    """Run comparison analysis on the trained model"""
    if 'output_path' not in config:
        print(f"No output path for {config['name']}")
        return None
    
    output_path = config['output_path']
    comparison_output = f"proof_{config['name']}.png"
    
    print(f"\nRunning comparison analysis for {config['name']}")
    print(f"Input: {output_path}")
    print(f"🖼️  Output: {comparison_output}")
    
    try:
        cmd = [
            "python", "create_real_proof_comparison.py",
            "--tf-path", output_path,
            "--output", comparison_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"Comparison completed successfully")
            
            # Parse the output to extract key metrics
            stdout = result.stdout
            metrics = {}
            
            # Look for key information in the output
            for line in stdout.split('\n'):
                if 'TensorFlow:' in line and 'episodes' in line:
                    try:
                        episodes = int(line.split(':')[1].strip().split()[0])
                        metrics['episodes'] = episodes
                    except:
                        pass
                elif 'Final action distribution:' in line:
                    # Extract action distribution
                    if 'back:' in line:
                        try:
                            back_pct = float(line.split('back\': ')[1].split(',')[0])
                            metrics['back_pct'] = back_pct
                        except:
                            pass
                    if 'filter:' in line:
                        try:
                            filter_pct = float(line.split('filter\': ')[1].split(',')[0])  
                            metrics['filter_pct'] = filter_pct
                        except:
                            pass
                    if 'group:' in line:
                        try:
                            group_pct = float(line.split('group\': ')[1].split('}')[0])
                            metrics['group_pct'] = group_pct
                        except:
                            pass
            
            # Check if summary file was created
            summary_file = f"comparison_summary_{Path(output_path).name}.txt"
            if Path(summary_file).exists():
                # Parse summary file for detailed metrics
                with open(summary_file, 'r') as f:
                    summary_content = f.read()
                
                # Extract reward statistics
                for line in summary_content.split('\n'):
                    if 'TensorFlow Mean:' in line:
                        try:
                            tf_mean = float(line.split(':')[1].strip())
                            metrics['tf_mean_reward'] = tf_mean
                        except:
                            pass
                    elif 'Master Mean:' in line:
                        try:
                            master_mean = float(line.split(':')[1].strip())
                            metrics['master_mean_reward'] = master_mean
                        except:
                            pass
                    elif 'Difference:' in line:
                        try:
                            diff = float(line.split(':')[1].strip())
                            metrics['reward_difference'] = diff
                        except:
                            pass
            
            config['comparison_metrics'] = metrics
            config['comparison_output'] = comparison_output
            print(f"Extracted metrics: {metrics}")
            
            return metrics
            
        else:
            print(f"Comparison failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Comparison timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"Comparison failed with exception: {e}")
        return None

def analyze_results(configs):
    """Analyze all results and determine the best configuration"""
    print(f"\n🔬 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)
    
    valid_configs = [c for c in configs if 'comparison_metrics' in c]
    
    if not valid_configs:
        print("No valid results to analyze")
        return
    
    # Print detailed results table
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Config':<15} {'Episodes':<10} {'TF Reward':<12} {'Master Reward':<12} {'Difference':<12} {'Back%':<8} {'Filter%':<8} {'Group%':<8}")
    print("-" * 80)
    
    best_config = None
    best_score = float('inf')  # Lower reward difference is better
    
    for config in valid_configs:
        metrics = config['comparison_metrics']
        
        episodes = metrics.get('episodes', 'N/A')
        tf_reward = metrics.get('tf_mean_reward', 'N/A')
        master_reward = metrics.get('master_mean_reward', 'N/A')
        reward_diff = metrics.get('reward_difference', 'N/A')
        back_pct = metrics.get('back_pct', 'N/A')
        filter_pct = metrics.get('filter_pct', 'N/A')
        group_pct = metrics.get('group_pct', 'N/A')
        
        # Format values properly, handling 'N/A' cases
        episodes_str = f"{episodes}" if isinstance(episodes, (int, float)) else f"{episodes}"
        tf_reward_str = f"{tf_reward:.3f}" if isinstance(tf_reward, (int, float)) else f"{tf_reward}"
        master_reward_str = f"{master_reward:.3f}" if isinstance(master_reward, (int, float)) else f"{master_reward}"
        reward_diff_str = f"{reward_diff:.3f}" if isinstance(reward_diff, (int, float)) else f"{reward_diff}"
        back_pct_str = f"{back_pct:.1f}" if isinstance(back_pct, (int, float)) else f"{back_pct}"
        filter_pct_str = f"{filter_pct:.1f}" if isinstance(filter_pct, (int, float)) else f"{filter_pct}"
        group_pct_str = f"{group_pct:.1f}" if isinstance(group_pct, (int, float)) else f"{group_pct}"
        
        print(f"{config['name']:<15} {episodes_str:<10} {tf_reward_str:<12} {master_reward_str:<12} {reward_diff_str:<12} {back_pct_str:<8} {filter_pct_str:<8} {group_pct_str:<8}")
        
        # Determine best config (closest to master in terms of reward)
        if isinstance(reward_diff, (int, float)) and reward_diff < best_score:
            best_score = reward_diff
            best_config = config
    
    print("-" * 80)
    
    if best_config:
        print(f"\nBEST CONFIGURATION: {best_config['name']}")
        print(f"   Description: {best_config['description']}")
        print(f"   Coefficients: kl={best_config['kl_coeff']}, comp={best_config['compaction_coeff']}, div={best_config['diversity_coeff']}, hum={best_config['humanity_coeff']}")
        print(f"   Reward difference from master: {best_score:.3f}")
        print(f"   Comparison plot: {best_config.get('comparison_output', 'N/A')}")
        print(f"   Training results: {best_config.get('output_path', 'N/A')}")
        
        # Action distribution analysis
        metrics = best_config['comparison_metrics']
        if all(key in metrics for key in ['back_pct', 'filter_pct', 'group_pct']):
            print(f"\nACTION DISTRIBUTION ANALYSIS:")
            print(f"   Back actions:   {metrics['back_pct']:.1f}%")
            print(f"   Filter actions: {metrics['filter_pct']:.1f}%") 
            print(f"   Group actions:  {metrics['group_pct']:.1f}%")
            
            # Check for policy collapse
            if metrics['back_pct'] < 5:
                print(f"   Warning: Low back action percentage - possible policy collapse")
            elif 10 <= metrics['back_pct'] <= 50:
                print(f"   Healthy back action percentage")
            
            # Check for balance
            action_entropy = -sum(p/100 * np.log2(p/100 + 1e-10) for p in [metrics['back_pct'], metrics['filter_pct'], metrics['group_pct']] if p > 0)
            if action_entropy > 1.4:  # High entropy = good balance
                print(f"   Well-balanced action distribution (entropy: {action_entropy:.2f})")
            else:
                print(f"   Imbalanced action distribution (entropy: {action_entropy:.2f})")
    
    # Save comprehensive report
    report_file = f"coefficient_experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_data = {
        'experiment_date': datetime.now().isoformat(),
        'configurations': configs,
        'best_configuration': best_config['name'] if best_config else None,
        'best_score': best_score if best_config else None
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return best_config

def main():
    """Run the complete coefficient comparison experiment"""
    print("COEFFICIENT COMPARISON EXPERIMENT")
    print("=" * 60)
    print("Testing 4 different coefficient configurations (2000 steps each):")
    for i, config in enumerate(COEFFICIENT_CONFIGS, 1):
        print(f"  {i}. {config['name']}: {config['expected']}")
    print()
    
    # Backup original config
    backup_original_config()
    
    successful_configs = []
    
    try:
        # Run training for each configuration SEQUENTIALLY
        for i, config in enumerate(COEFFICIENT_CONFIGS, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(COEFFICIENT_CONFIGS)}: {config['name'].upper()}")
            print(f"{'='*80}")
            print(f"⏳ SEQUENTIAL EXECUTION: This experiment will complete before the next one starts")
            
            # Small delay to make sequential nature clear
            if i > 1:
                print(f"⏸️  Brief pause between experiments...")
                time.sleep(2)
            
            # Update config
            update_config_coefficients(config)
            
            # Run training (BLOCKS until complete)  
            if run_training(config, steps=2000):
                print(f"\nTraining complete for {config['name']} - now running analysis...")
                
                # Run comparison (BLOCKS until complete)
                if run_comparison(config):
                    successful_configs.append(config)
                    print(f"EXPERIMENT {i}/{len(COEFFICIENT_CONFIGS)} FULLY COMPLETED: {config['name']}")
                    print(f"Ready to proceed to next experiment...")
                else:
                    print(f"Experiment {i} - comparison analysis failed")
            else:
                print(f"Experiment {i} - training failed")
                print(f"⏭️  Proceeding to next experiment...")
        
        # Analyze results
        if successful_configs:
            best_config = analyze_results(successful_configs)
            
            print(f"\nEXPERIMENT COMPLETE!")
            print(f"   Successful experiments: {len(successful_configs)}/{len(COEFFICIENT_CONFIGS)}")
            if best_config:
                print(f"   Best configuration: {best_config['name']}")
                print(f"   Use this plot for your professor: {best_config.get('comparison_output')}")
        else:
            print(f"\nNO SUCCESSFUL EXPERIMENTS")
            print("   Check the error messages above for debugging")
            
    finally:
        # Always restore original config
        restore_original_config()
        print(f"\nOriginal configuration restored")

if __name__ == "__main__":
    main()
