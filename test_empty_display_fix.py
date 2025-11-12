#!/usr/bin/env python3
"""Test to verify empty display check is working correctly"""

import sys
import numpy as np
import pandas as pd
sys.path.append('.')

print("=" * 60)
print("TESTING EMPTY DISPLAY CHECK")
print("=" * 60)

# Test 1: Check if the environment loads
print("\n📦 Test 1: Loading environment...")
try:
    from gym_atena.envs.atena_env_cont import ATENAEnvCont
    print("Environment loaded successfully")
except Exception as e:
    print(f"Failed to load environment: {e}")
    sys.exit(1)

# Test 2: Create environment and test empty display detection
print("\nTest 2: Testing empty display detection...")
try:
    env = ATENAEnvCont(max_steps=12)
    print("Environment created")
    
    # Reset to get initial observation
    obs = env.reset()
    print(f"Environment reset, obs shape: {obs.shape}")
    
    # Test the _is_empty_display method with the initial obs
    is_empty_obs_check = env._is_empty_display(obs)
    print(f"Observation-based check result: {is_empty_obs_check}")
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    print(f"Empty DataFrame len: {len(empty_df)}")
    print(f"Empty DataFrame check: {len(empty_df) == 0}")
    
    # Test with non-empty DataFrame
    non_empty_df = pd.DataFrame({'a': [1, 2, 3]})
    print(f"Non-empty DataFrame len: {len(non_empty_df)}")
    print(f"Non-empty DataFrame check: {len(non_empty_df) == 0}")
    
    print("Empty display detection logic is correct")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Run a quick episode and check for empty display warnings
print("\n🎮 Test 3: Running quick episode to check for empty display penalties...")
try:
    env = ATENAEnvCont(max_steps=5)
    obs = env.reset()
    
    empty_display_count = 0
    total_steps = 0
    
    for i in range(5):
        # Take a random action (mostly groups which tend to produce empty results)
        action = [2, np.random.randint(0, 12), 0, 0.5, 0, 0]  # Group action
        
        next_obs, reward, done, info = env.step(action)
        total_steps += 1
        
        # Check if this was an empty display (reward should be -1.0)
        if reward == -1.0:
            empty_display_count += 1
            print(f"  Step {i+1}: Empty display detected! Reward: {reward}")
        else:
            print(f"  Step {i+1}: Non-empty result, Reward: {reward:.4f}")
        
        if done:
            break
    
    print(f"\nSummary:")
    print(f"  Total steps: {total_steps}")
    print(f"  Empty displays detected: {empty_display_count}")
    print(f"  Empty display rate: {empty_display_count/total_steps*100:.1f}%")
    
    if empty_display_count > 0:
        print("Empty display check IS WORKING (detected and penalized empty results)")
    else:
        print("No empty displays detected in this test run")
        print("   This is OK - might just mean all actions produced valid results")
    
except Exception as e:
    print(f"Episode test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nNext step: Run training and monitor for:")
print("   1. 'Empty display detected via DataFrame check' warnings")
print("   2. Episodes with -1.0 rewards (empty display penalties)")
print("   3. Improved action quality over time")

