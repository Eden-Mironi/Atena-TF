#!/usr/bin/env python3
"""
Simple Model Evaluation Script
Skips expert loading to avoid recursion bug
"""

import os
import sys
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ppo.agent import PPOAgent
from gym_atena.envs.atena_env_cont import ATENAEnvCont

def main():
    model_path = "results/MASTER_EXACT_DIVERSE_MEANS_20250903_105450/interrupted_model"
    
    print("SIMPLE MODEL EVALUATION")
    print("=" * 40)
    
    # Check files exist
    policy_path = f"{model_path}_policy_weights.weights.h5"
    value_path = f"{model_path}_value_weights.weights.h5"
    
    if not os.path.exists(policy_path):
        print(f"Policy weights not found: {policy_path}")
        return
    if not os.path.exists(value_path):
        print(f"Value weights not found: {value_path}")
        return
        
    print(f"Found policy weights: {policy_path}")
    print(f"Found value weights: {value_path}")
    
    # Create environment
    print("\nCreating environment...")
    env = ATENAEnvCont('netowrking')
    dummy_obs = env.reset()
    print(f"Environment created, obs shape: {dummy_obs.shape}")
    
    # Create agent
    print("\nCreating agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    
    # Initialize model with dummy forward pass
    print("\nInitializing model...")
    dummy_obs_batch = np.expand_dims(dummy_obs, axis=0)
    result = agent.act(dummy_obs_batch)
    print(f"Model initialized, result type: {type(result)}")
    
    # Load weights
    print("\nLoading weights...")
    agent.policy.load_weights(policy_path)
    agent.value_net.load_weights(value_path)
    print("Weights loaded successfully!")
    
    # Test diverse actions
    print("\nTESTING ACTION DIVERSITY")
    print("-" * 30)
    
    action_samples = []
    for i in range(10):
        action = agent.act_most_probable(dummy_obs_batch)
        action_samples.append(action)
        print(f"Action {i+1:2d}: {action}")
    
    # Check diversity
    action_array = np.array(action_samples)
    std_devs = np.std(action_array, axis=0)
    print(f"\nDIVERSITY ANALYSIS:")
    print(f"   Standard deviations per dimension: {std_devs}")
    print(f"   Mean std: {np.mean(std_devs):.4f}")
    print(f"   Total range per dim: {np.max(action_array, axis=0) - np.min(action_array, axis=0)}")
    
    # Test single episode simulation
    print(f"\n🎮 SINGLE EPISODE SIMULATION")
    print("-" * 30)
    
    obs = env.reset()
    episode_actions = []
    
    for step in range(10):
        action = agent.act_most_probable(np.expand_dims(obs, axis=0))
        obs, reward, done, info = env.step(action, compressed=True)
        
        # Extract discrete action
        raw_action = info.get('raw_action', 'Unknown')
        episode_actions.append(raw_action)
        
        print(f"Step {step+1:2d}: {raw_action}")
        
        if done:
            break
    
    print(f"\nSUCCESS!")
    print(f"   Generated {len(episode_actions)} diverse actions!")
    print(f"   Unique actions: {len(set(map(str, episode_actions)))}")
    
if __name__ == "__main__":
    main()
