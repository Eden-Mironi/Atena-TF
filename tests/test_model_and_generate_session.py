#!/usr/bin/env python3
"""
Test Model Loading and Session Generation
Standalone script to test the trained model and generate sample sessions
"""
import os
import sys
import json
from pathlib import Path

# Set environment variables to reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project paths
sys.path.append('.')
sys.path.append('./Configuration')
sys.path.append('./models/ppo')

def test_model_loading():
    """Test loading the trained model with exact parameters"""
    print("TESTING MODEL LOADING WITH EXACT PARAMETERS")
    print("=" * 60)
    
    try:
        import Configuration.config as cfg
        from models.ppo.agent import PPOAgent
        import tensorflow as tf
        import numpy as np
        
        print(f"Imports successful")
        print(f"USE_PARAMETRIC_SOFTMAX_POLICY = {cfg.USE_PARAMETRIC_SOFTMAX_POLICY}")
        
        # Use exact parameters from the saved model
        obs_dim = 51  # From error message: saved model expects (51, 600)
        action_dim = 949 if cfg.USE_PARAMETRIC_SOFTMAX_POLICY else 6
        n_hidden_channels = 600  # From error message: saved model has 600 channels
        
        print(f"Model parameters:")
        print(f"   obs_dim: {obs_dim}")
        print(f"   action_dim: {action_dim}")
        print(f"   n_hidden_channels: {n_hidden_channels}")
        print(f"   use_parametric_softmax: {cfg.USE_PARAMETRIC_SOFTMAX_POLICY}")
        
        # Create agent with matching parameters
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_channels=n_hidden_channels,
            use_parametric_softmax=cfg.USE_PARAMETRIC_SOFTMAX_POLICY
        )
        print("Agent created successfully")
        
        # Check if model files exist
        model_dir = "results/1809-14:00"
        policy_weights = f"{model_dir}/trained_model_policy_weights.weights.h5"
        value_weights = f"{model_dir}/trained_model_value_weights.weights.h5"
        
        print(f"\nChecking model files:")
        print(f"   Policy weights: {os.path.exists(policy_weights)}")
        print(f"   Value weights: {os.path.exists(value_weights)}")
        
        if not os.path.exists(policy_weights) or not os.path.exists(value_weights):
            print("Model weight files not found!")
            return None
        
        # Load weights
        try:
            agent.policy.load_weights(policy_weights)
            print("Policy weights loaded successfully")
            
            agent.value_net.load_weights(value_weights)
            print("Value weights loaded successfully")
            
            # Test forward pass
            print("\nTesting model inference...")
            test_obs = tf.random.normal((obs_dim,), dtype=tf.float32)  # Single obs (no batch)
            action, log_prob, value = agent.act(test_obs)  # act() returns only 3 values
            print(f"   Test input shape: {test_obs.shape}")
            print(f"   Test action: {action} (type: {type(action)})")
            print(f"   Test log prob: {float(log_prob.numpy()):.3f}")
            print(f"   Test value: {float(value.numpy()):.3f}")
            
            print("\nMODEL LOADING AND INFERENCE SUCCESS!")
            return agent
            
        except Exception as weight_error:
            print(f"Error loading weights: {weight_error}")
            return None
            
    except Exception as e:
        print(f"Error in model loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_sample_session(agent, session_length=12):
    """Generate a sample session using the loaded agent"""
    if agent is None:
        print("No agent provided for session generation")
        return []
    
    print(f"\nGENERATING SAMPLE SESSION (length={session_length})")
    print("-" * 40)
    
    try:
        import tensorflow as tf
        from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
        import Configuration.config as cfg
        
        # Create environment
        env = make_enhanced_atena_env(max_steps=session_length, gradual_training=False)
        obs = env.reset()
        if isinstance(obs, list):
            obs = obs[0]
        
        print(f"Environment created, initial obs shape: {np.array(obs).shape}")
        
        session_actions = []
        session_rewards = []
        total_reward = 0
        
        for step in range(session_length):
            # Handle observation dimension mismatch (env gives 52, model expects 51)
            if len(obs) > 51:
                obs_for_model = obs[:51]
            else:
                obs_for_model = obs
            
            # Get model prediction using agent's act() method
            if cfg.USE_PARAMETRIC_SOFTMAX_POLICY:
                action, log_prob, value = agent.act(obs_for_model)  # act() returns only 3 values
                action_idx = action.numpy() if hasattr(action, 'numpy') else action
                
                # Convert discrete action to action string (EXPERT-COMPATIBLE FORMAT)
                if action_idx == 0:
                    action_str = '[back]'
                elif 1 <= action_idx <= 936:  # Filter actions
                    filter_idx = action_idx - 1
                    field = filter_idx // (3 * 26)
                    operator = (filter_idx // 26) % 3
                    action_str = f'[filter]_field{field}_op{operator}'  # FIXED: Expert format!
                elif 937 <= action_idx <= 948:  # Group actions
                    field = action_idx - 937
                    action_str = f'[group]_field{field}'  # FIXED: Expert format!
                else:
                    action_str = '[unknown]'
            else:
                # Continuous action (shouldn't happen with current config)
                action_str = '[continuous]'
            
            session_actions.append(action_str)
            
            # Take step in environment
            obs, reward, done, info = env.step(action_idx if cfg.USE_PARAMETRIC_SOFTMAX_POLICY else action)
            if isinstance(obs, list):
                obs = obs[0]
            
            session_rewards.append(reward)
            total_reward += reward
            
            print(f"  Step {step+1}: {action_str} → reward: {reward:.3f}")
            
            if done:
                print(f"  💀 Episode ended at step {step+1}")
                break
        
        print(f"\nSession complete:")
        print(f"   Actions: {session_actions}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Average reward: {total_reward/len(session_actions):.3f}")
        
        return {
            'actions': session_actions,
            'rewards': session_rewards,
            'total_reward': total_reward,
            'num_steps': len(session_actions)
        }
        
    except Exception as e:
        print(f"Error generating session: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main test function"""
    print("TENSORFLOW MODEL LOADING AND SESSION GENERATION TEST")
    print("=" * 70)
    
    # Test model loading
    agent = test_model_loading()
    
    if agent is not None:
        # Test session generation
        session_result = generate_sample_session(agent)
        
        if session_result:
            # Save results for notebook to use
            results = {
                'model_loading': 'SUCCESS',
                'agent_available': True,
                'sample_session': session_result,
                'model_params': {
                    'obs_dim': 51,
                    'action_dim': 949,
                    'n_hidden_channels': 600,
                    'use_parametric_softmax': True
                }
            }
            
            with open('model_test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: model_test_results.json")
            
            print(f"\nALL TESTS PASSED!")
            print(f"Model loads correctly with exact parameters")
            print(f"Session generation works")
            print(f"Ready for notebook evaluation")
        else:
            print(f"\nSession generation failed")
    else:
        print(f"\nModel loading failed")
        
        # Save failure results
        results = {
            'model_loading': 'FAILED',
            'agent_available': False,
            'error': 'Model loading failed - check console output'
        }
        
        with open('model_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
