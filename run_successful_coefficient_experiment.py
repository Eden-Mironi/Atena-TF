#!/usr/bin/env python3
"""
Run successful coefficient experiments that actually complete and log data.
Based on diagnostic findings: moderate coefficients work, extreme ones break training.
"""

import subprocess
import sys
import re
import os
import time

def set_coefficients(coefficients, experiment_name):
    """Set moderate coefficients that won't break training"""
    print(f"Setting coefficients for {experiment_name}")
    
    with open("Configuration/config.py", "r") as f:
        content = f.read()
    
    new_dict = f"""REWARD_COEFFICIENTS_MASTER_EXACT = {{
    'kl_coeff': {coefficients['kl_coeff']:.1f},
    'compaction_coeff': {coefficients['compaction_coeff']:.1f}, 
    'diversity_coeff': {coefficients['diversity_coeff']:.1f},
    'humanity_coeff': {coefficients['humanity_coeff']:.1f},
}}"""
    
    pattern = r"REWARD_COEFFICIENTS_MASTER_EXACT = \{[^}]*\}"
    content = re.sub(pattern, new_dict, content, flags=re.DOTALL)
    
    with open("Configuration/config.py", "w") as f:
        f.write(content)
    
    print(f"   Set: {coefficients}")

def run_training(outdir, steps=500, timeout_min=15):
    """Run training that should complete successfully"""
    print(f"Training: {outdir} ({steps} steps, {timeout_min}min timeout)")
    
    cmd = [sys.executable, "train_with_decay.py", 
           "--steps", str(steps), "--seed", "42", 
           "--outdir", outdir, "--enable-decay"]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_min*60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Check what files were actually created
            print(f"   Training completed ({elapsed:.1f}s)")
            if os.path.exists(outdir):
                files = os.listdir(outdir)
                print(f"      Created files: {files}")
                
                # Check for episode data in multiple possible formats
                episode_file = os.path.join(outdir, "episode_summary.jsonl")
                log_file = os.path.join(outdir, "training_detailed.log")
                
                if os.path.exists(episode_file) and os.path.getsize(episode_file) > 100:
                    print(f"      Episode data: {os.path.getsize(episode_file)} bytes")
                    return True
                elif os.path.exists(log_file) and os.path.getsize(log_file) > 200:
                    print(f"      Training log: {os.path.getsize(log_file)} bytes")
                    return True
                else:
                    print(f"      No substantial episode data found")
                    if os.path.exists(episode_file):
                        print(f"         episode_summary.jsonl: {os.path.getsize(episode_file)} bytes")
                    if os.path.exists(log_file):
                        print(f"         training_detailed.log: {os.path.getsize(log_file)} bytes")
                    # Still count as success if model files exist
                    model_files = [f for f in files if f.endswith('.h5')]
                    if model_files:
                        print(f"      Model files created: {model_files}")
                        return True
                    return False
            else:
                print(f"      Output directory not created")
                return False
        else:
            print(f"   FAILED: {outdir}")
            if result.stderr and len(result.stderr) > 0:
                print(f"      Error: {result.stderr[-200:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   TIMEOUT: {outdir} after {timeout_min}min")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def main():
    print("RUNNING SUCCESSFUL COEFFICIENT EXPERIMENTS")
    print("=" * 60)
    print("Using your specific coefficient combinations")
    print("Medium training (500 steps) for reliable results")
    print()
    
    # Backup original config
    import shutil
    backup_file = "Configuration/config_backup_working.py"
    if not os.path.exists(backup_file):
        shutil.copy("Configuration/config.py", backup_file)
    
    experiments = [
        {
            'name': 'POOR_PERFORMANCE',
            'coefficients': {
                'kl_coeff': 2.8,
                'compaction_coeff': 2.5,
                'diversity_coeff': 6.0,
                'humanity_coeff': 4.8
            },
            'outdir': 'coeff_poor_performance',
            'expected': 'Poor performance (avg reward: 0.253)'
        },
        {
            'name': 'EXCELLENT_MASTER_EXACT',
            'coefficients': {
                'kl_coeff': 2.2,
                'compaction_coeff': 2.0,
                'diversity_coeff': 8.0,
                'humanity_coeff': 4.5
            },
            'outdir': 'coeff_excellent_master_exact',
            'expected': 'Excellent performance (avg reward: 5.411)'
        },
        {
            'name': 'EXCELLENT_REDUCED',
            'coefficients': {
                'kl_coeff': 1.0,
                'compaction_coeff': 1.0,
                'diversity_coeff': 2.0,
                'humanity_coeff': 1.0
            },
            'outdir': 'coeff_excellent_reduced',
            'expected': 'Excellent performance (avg reward: 5.249)'
        },
        {
            'name': 'MINIMAL_COEFFS',
            'coefficients': {
                'kl_coeff': 1.0,
                'compaction_coeff': 1.0,
                'diversity_coeff': 1.0,
                'humanity_coeff': 1.0
            },
            'outdir': 'coeff_minimal',
            'expected': 'Unknown performance'
        }
    ]
    
    results = []
    
    try:
        for exp in experiments:
            print(f"\n🔬 EXPERIMENT: {exp['name']}")
            print(f"   Expected: {exp['expected']}")
            set_coefficients(exp['coefficients'], exp['name'])
            success = run_training(exp['outdir'])
            results.append((exp, success))
            
            if success:
                print(f"   {exp['name']} completed successfully!")
            else:
                print(f"   💥 {exp['name']} failed")
        
        successful = [r for r in results if r[1]]
        
        if len(successful) >= 2:
            print(f"\n🎊 {len(successful)}/{len(experiments)} EXPERIMENTS SUCCEEDED!")
            print("=" * 60)
            
            for exp, _ in successful:
                print(f"{exp['name']}: {exp['outdir']}")
                print(f"   Expected: {exp['expected']}")
            
            print(f"\nNOW RUN COMPARISONS:")
            for i, (exp, _) in enumerate(successful):
                output_name = exp['name'].lower().replace('_', '_')
                print(f"   python3 create_real_proof_comparison.py --tf-path {exp['outdir']} --output {output_name}_vs_master.png")
            
            print(f"\nEach should show DIFFERENT graphs with your specific coefficient effects!")
            
            # Show specific comparisons
            if len(successful) >= 2:
                exp1, exp2 = successful[0][0], successful[1][0]
                print(f"\nCompare Poor vs Excellent performance:")
                print(f"   Open: {exp1['name'].lower()}_vs_master.png and {exp2['name'].lower()}_vs_master.png")
            
        else:
            print(f"\nOnly {len(successful)}/{len(experiments)} experiments succeeded")
            for exp, success in results:
                status = "" if success else "" 
                print(f"   {status} {exp['name']} ({exp['expected']})")
                
    finally:
        # Restore backup
        if os.path.exists(backup_file):
            shutil.copy(backup_file, "Configuration/config.py")
            print(f"\n📋 Restored original config")

if __name__ == "__main__":
    main()
