#!/usr/bin/env python3
"""
COEFFICIENT COMPARISON TEST

This script runs two experiments with dramatically different coefficients
to show that they produce different learning curves.
"""

import subprocess
import os
import sys
import shutil

def run_experiment(name, coefficients, seed):
    """Run a single experiment with specific coefficients"""
    print(f"\nRUNNING EXPERIMENT: {name}")
    print(f"   Coefficients: {coefficients}")
    print(f"   Seed: {seed}")
    
    # Backup original config
    config_backup = "Configuration/config_backup.py"
    if not os.path.exists(config_backup):
        shutil.copy("Configuration/config.py", config_backup)
    
    # Modify config.py with new coefficients
    with open("Configuration/config.py", "r") as f:
        config_content = f.read()
    
    # Replace coefficient values
    for coeff_name, coeff_value in coefficients.items():
        # Find and replace the coefficient line
        import re
        pattern = f'{coeff_name}\\s*=\\s*[0-9.]+.*'
        replacement = f'{coeff_name} = {coeff_value}  # Test: {name}'
        config_content = re.sub(pattern, replacement, config_content)
    
    # Write modified config
    with open("Configuration/config.py", "w") as f:
        f.write(config_content)
    
    # Run training
    outdir = f"coefficient_test_{name.lower().replace(' ', '_')}"
    cmd = [
        sys.executable, "train_with_decay.py",
        "--steps", "200",  # Short test
        "--seed", str(seed),
        "--outdir", outdir,
        "--enable-decay"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"   SUCCESS: {name} completed")
        return outdir
    except subprocess.CalledProcessError as e:
        print(f"   FAILED: {name} failed with error: {e}")
        return None

def main():
    print("COEFFICIENT COMPARISON TEST")
    print("=" * 80)
    print("Testing two DRAMATICALLY different coefficient sets")
    print("If seeds work correctly, these should produce different curves!")
    print()
    
    # Experiment 1: Low coefficients (conservative rewards)
    exp1_coefficients = {
        'humanity_coeff': 0.1,
        'diversity_coeff': 0.1, 
        'kl_coeff': 0.1,
        'compaction_coeff': 0.1
    }
    
    # Experiment 2: High coefficients (aggressive rewards)  
    exp2_coefficients = {
        'humanity_coeff': 10.0,
        'diversity_coeff': 10.0,
        'kl_coeff': 10.0,
        'compaction_coeff': 10.0
    }
    
    # Run experiments
    outdir1 = run_experiment("LOW_COEFFICIENTS", exp1_coefficients, seed=42)
    outdir2 = run_experiment("HIGH_COEFFICIENTS", exp2_coefficients, seed=123)
    
    # Restore original config
    if os.path.exists("Configuration/config_backup.py"):
        shutil.copy("Configuration/config_backup.py", "Configuration/config.py")
        os.remove("Configuration/config_backup.py")
    
    print(f"\nCOMPARISON TEST COMPLETE!")
    print("=" * 80)
    
    if outdir1 and outdir2:
        print(f"Both experiments completed successfully!")
        print(f"Results saved to:")
        print(f"   - Low coefficients:  {outdir1}/")
        print(f"   - High coefficients: {outdir2}/")
        print()
        print("🔬 EXPECTED RESULTS:")
        print("   - Different learning curves (high coeffs should learn faster)")
        print("   - Different reward distributions")
        print("   - Different action type preferences")
        print()
        print("❓ IF THEY'RE STILL IDENTICAL:")
        print("   - Check if you're comparing the right directories")
        print("   - Verify the training actually used different coefficients")
        print("   - Look at the raw reward values in the logs")
    else:
        print("Some experiments failed - check error messages above")

if __name__ == "__main__":
    main()
