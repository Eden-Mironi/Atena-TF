#!/usr/bin/env python3
"""
Quick script to profile which imports are slow
"""
import time

def time_import(name, import_func):
    start = time.time()
    result = import_func()
    elapsed = time.time() - start
    print(f"  {name:.<50} {elapsed:>6.2f}s")
    return result

print("⏱️  Import Timing Breakdown:")
print("=" * 60)

time_import("Configuration.config", lambda: __import__('Configuration.config'))
time_import("gym", lambda: __import__('gym'))
time_import("numpy", lambda: __import__('numpy'))
time_import("pandas", lambda: __import__('pandas'))
time_import("matplotlib.pyplot", lambda: __import__('matplotlib.pyplot'))
time_import("seaborn", lambda: __import__('seaborn'))

# These are the potentially slow ones
time_import("tensorflow", lambda: __import__('tensorflow'))
time_import("gym_atena.envs.atena_env_cont", lambda: __import__('gym_atena.envs.atena_env_cont'))
time_import("Evaluation.notebook_utils", lambda: __import__('Evaluation.notebook_utils'))
time_import("live_recommender_agent", lambda: __import__('live_recommender_agent'))

print("=" * 60)
print("All imports complete!")

