#!/usr/bin/env python3
"""
Test ChainerRL-style Temporal Coupling

This script tests our new act_and_train method that matches ChainerRL's temporal learning:
- Agent sees previous reward when making decisions
- Proper episode ending with stop_episode_and_train
- Sequential experience processing like master

This should fix the learning curve differences by matching master's temporal patterns.
"""

import numpy as np
import sys
sys.path.append('.')

from models.ppo.agent import PPOAgent
from envs.env import create_env
import Configuration.config as cfg

def test_temporal_coupling():
    """Test ChainerRL-style temporal coupling"""
    
    print("TESTING CHAINERRL-STYLE TEMPORAL COUPLING")
    print("=" * 80)
    
    # Create environment and agent
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    # Create agent with parametric softmax (matching master)
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_parametric_softmax=True,  # Master's FFParamSoftmax
        n_hidden_channels=600,        # Master's 600 channels
        update_interval=10            # Small for testing
    )
    
    print(f"Agent created:")
    print(f"   - Observation dim: {obs_dim}")
    print(f"   - Action dim: {action_dim}")
    print(f"   - Policy: {'Parametric Softmax' if agent.use_parametric_softmax else 'Gaussian'}")
    print(f"   - Hidden channels: {agent.n_hidden_channels}")
    print()
    
    # Test ChainerRL-style training loop
    print("STARTING CHAINERRL-STYLE TRAINING SEQUENCE:")
    print("=" * 80)
    
    total_steps = 50
    episode_count = 0
    step_count = 0
    
    # Initial state (matching master exactly)
    obs = env.reset()
    prev_reward = 0.0  # Master starts with reward 0
    
    print(f"🏁 Initial state: obs_shape={obs.shape}, prev_reward={prev_reward}")
    
    while step_count < total_steps:
        print(f"\nStep {step_count + 1}:")
        
        # ChainerRL-style action selection with temporal coupling!
        action, log_prob, value = agent.act_and_train(obs, prev_reward)
        
        print(f"   🧠 act_and_train(obs, prev_reward={prev_reward:.3f})")
        print(f"   📤 Action: {action.numpy() if hasattr(action, 'numpy') else action}")
        print(f"   Value: {value.numpy():.4f}, LogProb: {log_prob.numpy():.4f}")
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        step_count += 1
        
        print(f"   🌍 Environment response: reward={reward:.3f}, done={done}")
        
        # Update for next iteration
        obs = next_obs
        prev_reward = reward  # Pass this reward to next act_and_train!
        
        # Handle episode termination (ChainerRL style)
        if done or step_count >= total_steps:
            print(f"   Episode {episode_count + 1} ending with final reward: {reward:.3f}")
            
            # ChainerRL-style episode ending!
            agent.stop_episode_and_train(obs, reward, done=True)
            
            episode_count += 1
            
            # Reset for new episode (if not finished)
            if step_count < total_steps:
                obs = env.reset()
                prev_reward = 0.0  # Reset reward for new episode
                print(f"   New episode started: obs_shape={obs.shape}, prev_reward={prev_reward}")
    
    print(f"\nTEMPORAL COUPLING TEST COMPLETE!")
    print("=" * 80)
    print(f"Results:")
    print(f"   - Total steps: {step_count}")
    print(f"   - Episodes completed: {episode_count}")
    print(f"   - Agent has experience buffer: {hasattr(agent, 'experience_buffer')}")
    
    if hasattr(agent, 'experience_buffer'):
        print(f"   - Experience buffer size: {len(agent.experience_buffer)}")
        print(f"   - Sample transition: {agent.experience_buffer[0] if agent.experience_buffer else 'None'}")
    
    print("\nKEY DIFFERENCES FROM OLD APPROACH:")
    print("   Agent sees previous reward in act_and_train(obs, prev_reward)")
    print("   Proper episode ending with stop_episode_and_train()")
    print("   Sequential experience processing (not batch)")
    print("   Temporal coupling enables cause-effect learning")
    
    print(f"\nThis should fix learning curve differences by matching ChainerRL's temporal patterns!")

if __name__ == "__main__":
    test_temporal_coupling()
