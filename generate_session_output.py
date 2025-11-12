#!/usr/bin/env python3
"""
Generate ATENA-TF Session Output
Creates session files in master's exact format showing step-by-step exploration
"""

import numpy as np
import tensorflow as tf
import sys
import os
from io import StringIO
import argparse

# Add paths
sys.path.append('.')
sys.path.append('./gym_atena')

from models.ppo.agent import PPOAgent
from vectorized_envs import make_evaluation_batch_env
import Configuration.config as cfg

def format_action_description(raw_action, info_dict):
    """Convert raw action and info to human-readable description like master"""
    try:
        if 'action' in info_dict:
            # Use the environment's action description
            return info_dict['action']
        else:
            # Fallback: manual formatting based on raw action
            action_type = int(raw_action[0])
            if action_type == 0:
                return "Back to previous state"
            elif action_type == 1:
                col_idx = int(raw_action[1]) 
                condition_idx = int(raw_action[2])
                filter_value = raw_action[3]
                return f"Filter on Column {col_idx} with condition {condition_idx} = {filter_value:.6f}"
            elif action_type == 2:
                col_idx = int(raw_action[1])
                return f"Group on Column {col_idx} and aggregate with '<built-in function len>' on the column 'packet_number'"
            else:
                return f"Unknown action type {action_type}"
    except Exception as e:
        return f"Action description error: {e}"

def format_reward_breakdown(reward_info):
    """Format reward breakdown like master's dict_items format"""
    try:
        if hasattr(reward_info, '__dict__'):
            # Convert reward_info object to dict
            reward_dict = {}
            for attr in ['empty_display', 'empty_groupings', 'same_display_seen_already', 
                        'back', 'diversity', 'interestingness', 'kl_distance', 
                        'compaction_gain', 'humanity']:
                if hasattr(reward_info, attr):
                    reward_dict[attr] = getattr(reward_info, attr)
        elif isinstance(reward_info, dict):
            reward_dict = reward_info
        else:
            reward_dict = {'total_reward': float(reward_info)}
        
        # Format as master's dict_items
        items = [(k, v) for k, v in reward_dict.items()]
        return f"dict_items({items})"
    except Exception as e:
        return f"dict_items([('reward_error', '{e}')])"

def format_human_rules(reward_info):
    """Format human rules like master's NetHumanRule format"""
    try:
        # Check if reward_info has rules_reward_info populated (like master)
        if hasattr(reward_info, 'rules_reward_info') and reward_info.rules_reward_info:
            # Master's format: dict_items([(NetHumanRule.enum_name, value), ...])
            items = list(reward_info.rules_reward_info.items())
            return f"dict_items({items})"
        # Check for alternative rule storage
        elif hasattr(reward_info, 'humanity_rules') and reward_info.humanity_rules:
            items = [(rule, score) for rule, score in reward_info.humanity_rules.items()]
            return f"dict_items({items})"
        else:
            # Empty rules like we're currently seeing
            return "dict_items([])"
    except Exception as e:
        return f"dict_items([('rule_error', '{e}')])"

def format_action_vector(action):
    """Format action vector like master's format: [2. 2. 0. 0. 0. 0.]"""
    try:
        if isinstance(action, (list, tuple, np.ndarray)):
            # Convert to floats and format with single decimal precision
            formatted = [f"{float(x):.0f}." if float(x) == int(float(x)) else f"{float(x):.1f}" for x in action]
            return f"[{' '.join(formatted)}]"
        else:
            return str(action)
    except Exception as e:
        return f"[action_format_error: {e}]"

def format_data_display(info_dict, max_rows=20, head_rows=10, tail_rows=10):
    """Format the data display like master's pandas output with truncation for long dataframes"""
    try:
        if 'raw_display' not in info_dict or info_dict['raw_display'] is None:
            return "No display data available"
        
        raw_display = info_dict['raw_display']
        
        # Handle tuple case: (input_df, processed_df) - we want the processed result
        if isinstance(raw_display, tuple) and len(raw_display) >= 2:
            # Master shows the processed/grouped result (second element)
            df = raw_display[1] if raw_display[1] is not None else raw_display[0]
            if df is None:
                return "No display data available"
        # Handle single DataFrame case
        elif hasattr(raw_display, 'to_string'):
            df = raw_display
        else:
            return str(raw_display)
        
        # Check if it's a dataframe and if truncation is needed
        if hasattr(df, 'to_string') and hasattr(df, '__len__'):
            if len(df) > max_rows:
                # Truncate: show first head_rows, "...", then last tail_rows
                return _format_truncated_dataframe(df, head_rows, tail_rows)
            else:
                return df.to_string()
        else:
            return str(df)
    except Exception as e:
        return f"Display error: {e}"

def _format_truncated_dataframe(df, head_rows=10, tail_rows=10):
    """Format a dataframe with truncation in the middle"""
    try:
        # Get head and tail sections
        head_df = df.head(head_rows)
        tail_df = df.tail(tail_rows)
        
        # Convert to string
        head_str = head_df.to_string()
        tail_str = tail_df.to_string()
        
        # Split into lines
        head_lines = head_str.split('\n')
        tail_lines = tail_str.split('\n')
        
        # Create the truncation indicator
        # Match the format from session1.txt - use "..." for each column position
        if len(head_lines) > 1:
            # Parse the header line to determine column positions
            header_line = head_lines[0]
            
            # Create a "..." pattern that respects column alignment
            # We'll put "..." at the beginning and for each major column
            truncation_parts = ["..."]
            
            # Add "..." for each column (approximating column width)
            num_cols = len(df.columns)
            for i in range(min(num_cols, 5)):  # Show ... for up to 5 columns
                truncation_parts.append("...")
            
            truncation_line = "             ".join(truncation_parts)
        else:
            truncation_line = "..."
        
        # Combine: header + head data + "..." + tail data (without tail header)
        result_lines = head_lines + [truncation_line]
        
        # Add tail lines but skip the header line
        if len(tail_lines) > 1:
            result_lines.extend(tail_lines[1:])
        
        return '\n'.join(result_lines)
    except Exception as e:
        # Fallback to regular string conversion
        return df.to_string()

def generate_session_output(model_path: str, dataset_id: int = 0, max_steps: int = 12, output_file: str = None):
    """Generate a session output file in master's exact format"""
    
    print(f"GENERATING TF SESSION OUTPUT")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_id}")
    print(f"🔢 Max steps: {max_steps}")
    print("="*60)
    
    # Create environment with DataFrame returns enabled for session display
    # Use single environment (not vectorized) to get proper DataFrame displays
    from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
    
    # Create single environment with ret_df=True to get actual data displays
    eval_env = make_enhanced_atena_env(max_steps=max_steps, gradual_training=False, ret_df=True)
    print(f"Created single environment with DataFrame returns enabled")
    
    obs_space = eval_env.observation_space
    action_space = eval_env.action_space
    
    # Create and load agent (matching evaluation script)
    parametric_segments = ((), (12, 3, 26), (12,))
    parametric_segments_sizes = [1, 12*3*26, 12]
    
    agent = PPOAgent(
        obs_dim=obs_space.shape[0],
        action_dim=action_space.shape[0],
        learning_rate=cfg.adam_lr,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_ratio=0.2,
        gamma=cfg.ppo_gamma,
        lambda_=cfg.ppo_lambda,
        use_parametric_softmax=True,
        parametric_segments=parametric_segments,
        parametric_segments_sizes=parametric_segments_sizes,
        n_hidden_channels=600,
        beta=1.0,
    )
    
    # Build networks
    sample_obs = eval_env.observation_space.sample()
    sample_obs = tf.expand_dims(tf.convert_to_tensor(sample_obs, dtype=tf.float32), 0)
    _ = agent.policy(sample_obs)
    _ = agent.value_net(sample_obs)
    
    # Load weights
    try:
        if os.path.isfile(model_path):
            agent.load_weights(model_path)
        elif os.path.isdir(model_path):
            policy_weights = os.path.join(model_path, "trained_model_policy_weights.weights.h5")
            value_weights = os.path.join(model_path, "trained_model_value_weights.weights.h5")
            if os.path.exists(policy_weights) and os.path.exists(value_weights):
                agent.policy.load_weights(policy_weights)
                agent.value_net.load_weights(value_weights)
            else:
                agent.load_weights(os.path.join(model_path, "trained_model"))
        print(f"Successfully loaded trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Generate session output
    session_output = []
    
    try:
        # Reset environment to specified dataset
        obs = eval_env.reset(dataset_number=dataset_id)
        step_count = 0
        
        print(f"Starting session generation...")
        
        while step_count < max_steps:
            step_count += 1
            
            # Get action from trained agent (stochastic for diversity)
            action_result = agent.act(obs)  # Use stochastic action selection
            
            if isinstance(action_result, tuple):
                action_idx = action_result[0]
                if isinstance(action_idx, np.ndarray):
                    action_idx = action_idx.item()
            else:
                action_idx = action_result
            
            # Convert discrete action to continuous for single environment
            if hasattr(eval_env, '_discrete_to_continuous_action'):
                continuous_action = eval_env._discrete_to_continuous_action(action_idx)
            else:
                continuous_action = np.zeros(6, dtype=np.float32)
            
            # Execute action (single environment returns scalars, not arrays)
            next_obs, reward, done, info_dict = eval_env.step(continuous_action)
            
            # Get raw_action for display
            if 'raw_action' in info_dict:
                raw_action = info_dict['raw_action']
            else:
                raw_action = continuous_action
            
            # Format output in master's exact style
            session_output.append(format_action_vector(raw_action))
            session_output.append(format_action_description(raw_action, info_dict))
            session_output.append(f"reward:{reward}")
            
            # Add reward breakdown
            if 'reward_info' in info_dict:
                session_output.append(format_reward_breakdown(info_dict['reward_info']))
                session_output.append("")  # Empty line
                session_output.append(format_human_rules(info_dict['reward_info']))
            else:
                session_output.append(f"dict_items([('total_reward', {reward})])")
                session_output.append("")
                session_output.append("dict_items([])")
            
            session_output.append("")  # Empty line
            
            # Add data display
            display_output = format_data_display(info_dict)
            session_output.append(display_output)
            session_output.append("-" * 51)  # Master uses exactly 51 dashes
            
            # Update observation
            obs = next_obs
            
            print(f"  Step {step_count}: {format_action_description(raw_action, info_dict)[:60]}...")
            
            if done:
                print(f"  🏁 Episode completed at step {step_count}")
                break
                
    except Exception as e:
        print(f"Error during session generation: {e}")
        return None
    finally:
        eval_env.close()
    
    # Save output
    if output_file is None:
        output_file = f"tf_session_dataset{dataset_id}.txt"
    
    try:
        with open(output_file, 'w') as f:
            for line in session_output:
                f.write(line + '\n')
        
        print(f"Session saved to: {output_file}")
        print(f"📏 Generated {len(session_output)} lines of output")
        print(f"Completed {step_count} exploration steps")
        
        return output_file
        
    except Exception as e:
        print(f"Error saving session: {e}")
        return None

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Generate ATENA-TF session output in master format')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (e.g., results/0409-20:13/trained_model)')
    parser.add_argument('--dataset', type=int, default=0,
                       help='Dataset ID to explore (default: 0)')
    parser.add_argument('--steps', type=int, default=12,
                       help='Maximum steps in session (default: 12)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (default: tf_session_datasetX.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path) and not os.path.exists(args.model_path.replace('/trained_model', '')):
        print(f"Model not found at: {args.model_path}")
        return 1
    
    result = generate_session_output(
        model_path=args.model_path,
        dataset_id=args.dataset,
        max_steps=args.steps,
        output_file=args.output
    )
    
    if result:
        print(f"\nSUCCESS! Session output generated.")
        print(f"File: {result}")
        print(f"This shows your trained TF model exploring data step-by-step")
        print(f"Perfect for demonstrating the system working intelligently!")
        return 0
    else:
        print(f"\nFailed to generate session output.")
        return 1

if __name__ == "__main__":
    exit(main())
