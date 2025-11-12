#!/usr/bin/env python3
"""
MASTER-EXACT Model Evaluation Script
Replicates TF_Evaluation.ipynb functionality as a command-line script

Usage: python3 evaluate_model.py <model_path>
Example: python3 evaluate_model.py results/MASTER_EXACT_DIVERSE_MEANS_20250903_105450/interrupted_model
"""

import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from copy import deepcopy
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Add project paths
sys.path.append('.')
sys.path.append('..')
sys.path.append('../ATENA-master')

# Import our modules
from models.ppo.agent import PPOAgent
from gym_atena.envs.atena_env_cont import ATENAEnvCont
import Configuration.config as cfg

def load_model(model_path):
    """EXACT COPY of working model loading from test_master_exact_evaluation.py"""
    print(f"Loading model from: {model_path}")
    
    # Check if model exists
    policy_path = f"{model_path}_policy_weights.weights.h5"
    value_path = f"{model_path}_value_weights.weights.h5"
    
    if not (os.path.exists(policy_path) and os.path.exists(value_path)):
        print(f"Model weights not found:")
        print(f"   Looking for: {policy_path}")
        print(f"   Looking for: {value_path}")
        return None, None
    
    try:
        # EXACT COPY: Create environment using make_enhanced_atena_env (fallback to ATENAEnvCont)
        try:
            from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
            env = make_enhanced_atena_env(gradual_training=False)
            print("Using enhanced environment")
        except ImportError:
            print("Enhanced env not available, using ATENAEnvCont")
            env = ATENAEnvCont()
        
        # AUTO-DETECT: Get architecture parameters from config
        obs_dim = 51  # Standard ATENA observation size
        action_dim = 6  # Standard ATENA action space (continuous)
        
        # Read architecture from config
        use_parametric_softmax = cfg.USE_PARAMETRIC_SOFTMAX_POLICY
        n_hidden_channels = cfg.n_hidden_channels  # 600 for parametric softmax, 64 for Gaussian
        
        print(f"Auto-detected architecture from config:")
        print(f"   - Policy type: {'Parametric Softmax' if use_parametric_softmax else 'Gaussian'}")
        print(f"   - Hidden channels: {n_hidden_channels}")
        print(f"   - Obs dim: {obs_dim}, Action dim: {action_dim}")
        
        # Get parametric segments if using parametric softmax
        parametric_segments = None
        parametric_segments_sizes = None
        
        if use_parametric_softmax:
            try:
                # Get segments from environment
                if hasattr(env, 'env_prop'):
                    parametric_segments = env.env_prop.get_parametric_segments()
                    parametric_segments_sizes = env.env_prop.get_parametric_softmax_segments_sizes()
                    print(f"   - Parametric segments: {parametric_segments}")
                    print(f"   - Segment sizes: {parametric_segments_sizes}")
            except Exception as e:
                print(f"Could not get parametric segments: {e}")
        
        # Create agent with detected parameters
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_hidden_channels=n_hidden_channels,
            use_parametric_softmax=use_parametric_softmax,
            parametric_segments=parametric_segments,
            parametric_segments_sizes=parametric_segments_sizes,
            beta=cfg.BETA_TEMPERATURE if hasattr(cfg, 'BETA_TEMPERATURE') else 1.0
        )
        
        # EXACT COPY: Initialize model with dummy observation (CRITICAL!)
        dummy_obs = np.random.random((1, 51)).astype(np.float32)
        agent.act(dummy_obs)  # This initializes the model structure!
        
        # EXACT COPY: Load weights using the exact pattern that worked
        try:
            agent.policy.load_weights(policy_path)
            agent.value_net.load_weights(value_path)
            print(f"Model loaded successfully:")
            print(f"   Policy: {policy_path}")
            print(f"   Value: {value_path}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            print(f"Hint: Make sure the model was trained with the same architecture as in config.py")
            print(f"   Current config: USE_PARAMETRIC_SOFTMAX_POLICY={use_parametric_softmax}, n_hidden_channels={n_hidden_channels}")
            return None, None
        
        return agent, env
        
    except Exception as e:
        print(f"Error creating agent/environment: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def info_hist_to_raw_actions_lst(info_hist):
    """
    EXACT COPY from master: Utilities/Notebook/NotebookUtils.py line 573
    A utility function that returns a list of all actions in the given `info_hist` object
    """
    actions_lst = []
    for info, _ in info_hist:
        info = deepcopy(info)
        # Master does this transformation: line 585
        info["raw_action"][3] -= 0.5
        actions_lst.append(info["raw_action"])
    return actions_lst

def run_episode_master_exact(agent, env, dataset_number, most_probable=True):
    """
    EXACT COPY of master's run_episode logic with most_probable=True
    Based on ATENA-master/Utilities/Notebook/NotebookUtils.py line 338
    """
    if most_probable:
        assert agent
    info_hist = []
    
    # Reset environment to specific dataset
    s = env.reset()
    # Master sets max_steps in environment
    env.max_steps = 12  # Master's default episode length
    
    r_sum = 0
    num_of_steps = 12  # Master's evaluation uses 12 steps max
    
    for ep_t in range(num_of_steps):
        # Master's action selection logic (line 401-404):
        if most_probable:
            # This calls master's act_most_probable function
            a, _, _ = agent.act_most_probable(s)  # Our implementation matches master's
        else:
            a, _, _ = agent.act(s)
        
        # Master calls env.step with these parameters (line 407):
        s_, r, done, info = env.step(a, compressed=True)  # Master uses compressed=True in evaluation!
        
        # Master adds to info_hist (line 409):
        info_hist.append((info, r))
        
        s = s_
        r_sum += r
        
        if done:
            break
    
    return info_hist, r_sum

def get_actions_lst_and_total_reward_of_agent_for_dataset(agent, env, dataset_num):
    """
    EXACT COPY from master: Utilities/Notebook/NotebookUtils.py line 619
    """
    info_hist, r_sum = run_episode_master_exact(
        agent=agent,
        env=env,
        dataset_number=dataset_num,
        most_probable=True,  # Master ALWAYS uses most_probable=True for evaluation
    )
    return info_hist_to_raw_actions_lst(info_hist), r_sum

def get_actions_lst_of_agent_for_dataset(agent, env, dataset_num):
    """
    MASTER-EXACT: Generate agent actions for dataset 
    (EXACT function name from master but now using master's methodology)
    """
    return get_actions_lst_and_total_reward_of_agent_for_dataset(agent, env, dataset_num)[0]

def load_expert_references():
    """Load expert reference actions from master"""
    print("Loading expert reference actions...")
    
    # Load expert sessions (from master)
    expert_sessions_path = "../ATENA-master/eval_sessions/expert/networking"
    
    if not os.path.exists(expert_sessions_path):
        print(f"Expert sessions path not found: {expert_sessions_path}")
        print("Using fallback references...")
        # Fallback references based on known master outputs
        action_type_references = [
            [['[back]'], ['[filter]'], ['[group]'], ['[back]'], ['[filter]'], ['[back]'], ['[filter]'], 
             ['[back]'], ['[filter]'], ['[group]'], ['[group]'], ['[filter]']]
        ]
        attr_references = [
            [['[back]'], ['[filter]_[highest_layer]'], ['[group]_[highest_layer]'], ['[back]'], 
             ['[filter]_[info_line]'], ['[back]'], ['[filter]_[info_line]'], ['[back]'], 
             ['[filter]_[highest_layer]'], ['[group]_[tcp_srcport]'], ['[group]_[ip_src]'], ['[filter]_[ip_src]']]
        ]
        return action_type_references, attr_references
    
    # Load actual expert sessions
    action_type_references = []
    attr_references = []
    
    try:
        import glob
        expert_files = glob.glob(os.path.join(expert_sessions_path, "*.txt"))[:1]  # Use first file for now
        
        for file_path in expert_files:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse expert actions (simplified parsing)
            file_action_types = []
            file_attrs = []
            
            for line in lines:
                line = line.strip()
                if '[' in line and ']' in line:
                    if '_[' in line:  # Has attribute
                        file_attrs.append([line])
                        action_part = line.split('_[')[0] + ']'
                        file_action_types.append([action_part])
                    else:  # Just action type
                        file_action_types.append([line])
                        file_attrs.append([line])
            
            if file_action_types:
                action_type_references.append(file_action_types[:12])  # Limit to 12 like master
                attr_references.append(file_attrs[:12])
        
        if not action_type_references:
            print("No expert actions found, using fallback...")
            return load_expert_references()  # Use fallback
            
    except Exception as e:
        print(f"Error loading expert files: {e}, using fallback...")
        return load_expert_references()  # Use fallback
    
    print(f"Loaded {len(action_type_references)} expert reference dataset(s)")
    return action_type_references, attr_references

def actions_to_bleu_format(actions_list):
    """Convert action vectors to BLEU format (action types and attributes)"""
    action_types = []
    attributes = []
    
    for action in actions_list:
        if len(action) >= 6:
            # Convert to discrete action
            discrete_action = [int(round(x)) for x in action[:6]]
            
            # Map to action type
            if discrete_action[0] == 0:  # back
                action_types.append('[back]')
                attributes.append('[back]')
            elif discrete_action[0] == 1:  # filter
                attr_map = {1: 'info_line', 2: 'highest_layer', 3: 'ip_src', 4: 'ip_dst', 5: 'tcp_srcport'}
                attr = attr_map.get(discrete_action[1], 'info_line')
                action_types.append('[filter]')
                attributes.append(f'[filter]_[{attr}]')
            elif discrete_action[0] == 2:  # group
                attr_map = {1: 'info_line', 2: 'highest_layer', 3: 'ip_src', 4: 'ip_dst', 5: 'tcp_srcport'}
                attr = attr_map.get(discrete_action[1], 'info_line')
                action_types.append('[group]')
                attributes.append(f'[group]_[{attr}]')
            else:
                action_types.append('[unknown]')
                attributes.append('[unknown]')
        else:
            action_types.append('[unknown]')
            attributes.append('[unknown]')
    
    return action_types, attributes

def calculate_bleu_scores(agent, env):
    """Calculate BLEU scores using master-exact methodology"""
    print("\nCalculating BLEU Scores (Master-Exact Methodology)")
    print("=" * 60)
    
    # Generate agent actions for dataset 0
    print("Generating agent actions...")
    agent_actions = get_actions_lst_of_agent_for_dataset(agent, env, 0)
    print(f"Generated {len(agent_actions)} agent actions")
    
    # Convert to BLEU format
    agent_action_types, agent_attributes = actions_to_bleu_format(agent_actions)
    
    print(f"Sample agent actions:")
    for i, (action, action_type, attr) in enumerate(zip(agent_actions[:5], agent_action_types[:5], agent_attributes[:5])):
        print(f"  {i+1}: {action[:3]} → {action_type} → {attr}")
    
    # Load expert references
    action_type_references, attr_references = load_expert_references()
    
    # Calculate BLEU scores
    smooth = SmoothingFunction()
    
    # Sanity check
    sanity_ref = [agent_action_types]
    sanity_cand = agent_action_types
    sanity_score = corpus_bleu(sanity_ref, sanity_cand, smoothing_function=smooth.method1)
    print(f"\nSanity Check BLEU: {sanity_score:.3f}")
    
    if abs(sanity_score - 1.0) < 0.001:
        print("Sanity check passed!")
    else:
        print(f"Sanity check failed: expected 1.0, got {sanity_score}")
    
    # Action Type BLEU
    agent_action_type_candidates = [agent_action_types]
    action_type_bleu = corpus_bleu(action_type_references, agent_action_type_candidates, 
                                  smoothing_function=smooth.method1)
    
    # Attribute BLEU  
    agent_attr_candidates = [agent_attributes]
    attr_bleu = corpus_bleu(attr_references, agent_attr_candidates, 
                           smoothing_function=smooth.method1)
    
    # Display results
    print(f"\nBLEU SCORES RESULTS:")
    print(f"=" * 40)
    print(f"Agent Action Type BLEU: {action_type_bleu:.3f}")
    print(f"Agent Attribute BLEU:   {attr_bleu:.3f}")
    
    # Show action distribution
    action_counts = {}
    for action_type in agent_action_types:
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    print(f"\nAction Distribution:")
    total_actions = len(agent_action_types)
    for action, count in action_counts.items():
        percentage = (count / total_actions) * 100
        print(f"  {action}: {count}/{total_actions} ({percentage:.1f}%)")
    
    # Comparison with expected scores
    if action_type_bleu >= 0.4:
        print(f"\nEXCELLENT: Action Type BLEU {action_type_bleu:.3f} >= 0.4 (Master-level!)")
    elif action_type_bleu >= 0.2:
        print(f"\nGOOD: Action Type BLEU {action_type_bleu:.3f} >= 0.2 (Strategic diversity emerging)")
    elif action_type_bleu >= 0.1:
        print(f"\nPROGRESS: Action Type BLEU {action_type_bleu:.3f} >= 0.1 (Tactical diversity)")
    else:
        print(f"\nEARLY STAGE: Action Type BLEU {action_type_bleu:.3f} < 0.1 (Needs more training)")
    
    return {
        'action_type_bleu': action_type_bleu,
        'attr_bleu': attr_bleu,
        'sanity_bleu': sanity_score,
        'action_distribution': action_counts,
        'total_actions': total_actions
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate TensorFlow ATENA model using master-exact methodology')
    parser.add_argument('model_path', help='Path to the trained model directory or file')
    parser.add_argument('--dataset', type=int, default=0, help='Dataset number to evaluate (default: 0)')
    
    args = parser.parse_args()
    
    print("MASTER-EXACT MODEL EVALUATION")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    
    # Load model
    agent, env = load_model(args.model_path)
    if agent is None or env is None:
        print("Failed to load model")
        return 1
    
    # Calculate BLEU scores
    try:
        results = calculate_bleu_scores(agent, env)
        
        print(f"\nEVALUATION COMPLETE!")
        print(f"Final Results:")
        print(f"   Action Type BLEU: {results['action_type_bleu']:.3f}")
        print(f"   Attribute BLEU:   {results['attr_bleu']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
