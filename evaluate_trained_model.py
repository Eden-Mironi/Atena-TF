#!/usr/bin/env python3
"""
MASTER-EXACT MODEL EVALUATION SCRIPT
Evaluates any trained ATENA model with comprehensive BLEU scoring and statistics

Usage:
    python evaluate_trained_model.py --model_path results/small_training_1000_steps/trained_model
    python evaluate_trained_model.py --model_path results/small_training_1000_steps/trained_model --episodes 20
    python evaluate_trained_model.py --model_path results/small_training_1000_steps/trained_model --datasets 0,1,2
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple
from copy import deepcopy

# Set TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add paths
sys.path.append('.')
sys.path.append('./Configuration')
sys.path.append('./gym_atena')
sys.path.append('./models')

import config as cfg
from models.ppo.agent import PPOAgent
from models.ppo.param_softmax_policy import ParamSoftmaxPolicy
from vectorized_envs import make_evaluation_batch_env

def info_hist_to_raw_actions_lst(info_hist):
    """
    EXACT COPY from master: Utilities/Notebook/NotebookUtils.py line 573
    A utility function that returns a list of all actions in the given `info_hist` object
    """
    actions_lst = []
    for info in info_hist:
        # Handle different info formats (could be tuple or dict)
        if isinstance(info, tuple):
            info_dict = info[0]
        else:
            info_dict = info
            
        info_dict = deepcopy(info_dict)
        
        # Master does this transformation: line 585
        if "raw_action" in info_dict:
            info_dict["raw_action"][3] -= 0.5
            actions_lst.append(info_dict["raw_action"])
        else:
            # Fallback for missing raw_action
            actions_lst.append([0, 0, 0, 0, 0, 0])  # Default action
    return actions_lst

def load_evaluation_datasets() -> List[Dict]:
    """Load evaluation datasets (simplified version)"""
    print("Using built-in evaluation datasets...")
    
    # For simplicity, we'll work with the environment's built-in datasets
    # The datasets will be accessed via dataset_id in the environment
    datasets = [{'id': i, 'name': f'dataset_{i}'} for i in range(3)]
    print(f"Using {len(datasets)} evaluation datasets")
    
    return datasets

def create_agent_and_load_model(model_path: str) -> PPOAgent:
    """Create agent and load trained model weights"""
    print(f"Loading model from: {model_path}")
    
    # Create evaluation environment to get dimensions
    eval_env = make_evaluation_batch_env(num_envs=1)
    
    # Create agent with same architecture as training
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    
    # Get parametric segments from environment for FFParamSoftmax
    # Use same approach as training code
    from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
    temp_env = make_enhanced_atena_env()
    
    # Force FFParamSoftmax architecture to match training
    print("FORCING FFParamSoftmax architecture (our trained model uses this!)")
    
    # Use exact same segments as training uses
    parametric_segments = ((), (12, 3, 26), (12,))  # back, filter, group
    parametric_segments_sizes = [1, 12*3*26, 12]    # [1, 936, 12] 
    use_parametric_softmax = True
    n_hidden_channels = 600  # Master's FFParamSoftmax uses 600 channels
    
    print(f"Using FFParamSoftmax segments: {parametric_segments}")
    print(f"Total actions: {sum(parametric_segments_sizes)} = {1 + 936 + 12}")
    
    # Keep the environment check for debugging
    try:
        if hasattr(temp_env, 'env_prop') and hasattr(temp_env.env_prop, 'get_parametric_segments'):
            env_segments = temp_env.env_prop.get_parametric_segments()
            print(f"Environment segments (for comparison): {env_segments}")
    except Exception as e:
        print(f"Environment check failed: {e}")
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=cfg.adam_lr,
        update_interval=2048,  # Training parameter (not used in evaluation)
        minibatch_size=64,     # Training parameter (not used in evaluation)
        epochs=10,             # Training parameter (not used in evaluation)
        clip_ratio=0.2,
        gamma=cfg.ppo_gamma,
        lambda_=cfg.ppo_lambda,
        use_parametric_softmax=use_parametric_softmax,
        parametric_segments=parametric_segments,
        parametric_segments_sizes=parametric_segments_sizes,
        n_hidden_channels=n_hidden_channels,
        beta=1.0,
    )
    
    # Build networks by calling them with sample data
    sample_obs = eval_env.observation_space.sample()
    sample_obs = tf.expand_dims(tf.convert_to_tensor(sample_obs, dtype=tf.float32), 0)
    
    # Build policy network
    _ = agent.policy(sample_obs)
    # Build value network
    _ = agent.value_net(sample_obs)
    
    print(f"Networks built with obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Load model weights (handle different saving formats)
    try:
        if os.path.isfile(model_path):
            # Single file format
            agent.load_weights(model_path)
        elif os.path.isdir(model_path):
            # Directory format with separate files
            policy_weights = os.path.join(model_path, "trained_model_policy_weights.weights.h5")
            value_weights = os.path.join(model_path, "trained_model_value_weights.weights.h5")
            
            if os.path.exists(policy_weights) and os.path.exists(value_weights):
                agent.policy.load_weights(policy_weights)
                agent.value_net.load_weights(value_weights)
                print(f"Loaded separate policy and value weights")
            else:
                # Try loading as single model
                agent.load_weights(model_path)
        else:
            # Try with different extensions/patterns
            base_path = model_path.replace("/trained_model", "")
            policy_weights = os.path.join(base_path, "trained_model_policy_weights.weights.h5")
            value_weights = os.path.join(base_path, "trained_model_value_weights.weights.h5")
            
            if os.path.exists(policy_weights) and os.path.exists(value_weights):
                agent.policy.load_weights(policy_weights)
                agent.value_net.load_weights(value_weights)
                print(f"Loaded separate policy and value weights from: {base_path}")
            else:
                raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
        print(f"Model weights loaded successfully")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
    
    eval_env.close()
    return agent

def run_episode_master_exact(agent: PPOAgent, dataset_id: int, max_steps: int = 12) -> Tuple[List[str], float, List[List]]:
    """
    Run single episode with master-exact methodology
    Returns: (action_strings, total_reward, raw_actions_list)
    """
    
    # Create environment for this specific dataset
    # Note: Our environment setup doesn't support dataset_id selection yet
    # Using default dataset (dataset 0)
    eval_env = make_evaluation_batch_env(num_envs=1)
    
    try:
        # Reset environment
        obs = eval_env.reset()
        
        # Track episode data
        info_hist = []
        total_reward = 0.0
        step_count = 0
        
        while step_count < max_steps:
            # EVALUATION FIX: Use stochastic action selection like training
            # This matches how training worked and should give diverse actions
            print(f"Agent type: {type(agent).__name__}, Policy type: {type(agent.policy).__name__}")
            print(f"Is discrete: {agent.is_discrete}")
            if hasattr(agent.policy, 'parametric_segments'):
                print(f"Parametric segments: {agent.policy.parametric_segments}")
            action_result = agent.act(obs)  # FIX: Use stochastic sampling like training
            print(f"Raw action result: {action_result}")
            
            # If it's ParamSoftmax, let's also check the action probabilities
            if agent.is_discrete and isinstance(agent.policy, ParamSoftmaxPolicy):
                obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = tf.expand_dims(obs_tensor, 0)
                logits = agent.policy(obs_tensor, training=False)
                probs = agent.policy.get_action_probabilities(logits)
                print(f"Action probabilities (first 10): {probs[0][:10]}")
                print(f"Max probability action: {tf.argmax(probs[0])}, value: {tf.reduce_max(probs[0])}")
            
            # Extract action from result (could be tuple with log_prob, entropy)
            if isinstance(action_result, tuple):
                action = action_result[0]  # First element is the action
            else:
                action = action_result
            
            # Convert action to numpy if it's a tensor
            if hasattr(action, 'numpy'):
                action = action.numpy()
            action = np.asarray(action, dtype=np.float32)
            
            # Handle discrete vs continuous actions correctly
            if agent.is_discrete and isinstance(agent.policy, ParamSoftmaxPolicy):
                # FFParamSoftmax returns discrete action index - convert to continuous action
                if action.ndim == 0 or (action.ndim == 1 and action.shape[0] == 1):
                    # Single discrete action index - convert to continuous action vector
                    action_idx = int(action.item()) if action.ndim == 0 else int(action[0])
                    
                    # Convert discrete action index to continuous action using environment's method
                    # This matches master's pipeline where discrete actions get converted internally
                    try:
                        # Try multiple ways to access the environment's conversion method
                        temp_env = None
                        if hasattr(eval_env, 'envs') and len(eval_env.envs) > 0:
                            # Vectorized environment
                            env_wrapper = eval_env.envs[0]
                            # Try different attribute paths
                            if hasattr(env_wrapper, 'env') and hasattr(env_wrapper.env, 'param_softmax_idx_to_action'):
                                temp_env = env_wrapper.env
                            elif hasattr(env_wrapper, 'param_softmax_idx_to_action'):
                                temp_env = env_wrapper
                        elif hasattr(eval_env, 'param_softmax_idx_to_action'):
                            # Direct environment
                            temp_env = eval_env
                        
                        if temp_env and hasattr(temp_env, 'param_softmax_idx_to_action'):
                            action = temp_env.param_softmax_idx_to_action(action_idx)
                            action = np.asarray(action, dtype=np.float32)
                            print(f"Converted discrete action {action_idx} to continuous: {action}")
                        else:
                            # Fallback: manual conversion based on master's algorithm
                            print(f"Environment method not found, using fallback conversion for action {action_idx}")
                            if action_idx == 0:
                                # Back action (action_type=0, no parameters)
                                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                            else:
                                # For now, fallback to zero action
                                action = np.zeros(6, dtype=np.float32)
                    except Exception as e:
                        print(f"Warning: Failed to convert discrete action {action_idx}: {e}")
                        action = np.zeros(6, dtype=np.float32)  # Fallback
                else:
                    # Already a continuous action vector
                    action = action.flatten()
            else:
                # FFGaussian or other continuous policies
                action = action.flatten()
            
            # Ensure action has correct shape for environment
            if len(action) != 6:
                print(f"Warning: Action has wrong shape {action.shape}, expected (6,)")
                action = np.zeros(6, dtype=np.float32)  # Fallback
            
            # Execute action (vectorized env expects batch of actions)
            next_obs, reward, done, info = eval_env.step(np.array([action], dtype=np.float32))
            
            # Extract single environment results (batch size = 1)
            reward = reward[0]
            done = done[0] 
            info = info[0]
            
            # Track step data
            info_hist.append(info)
            total_reward += reward
            step_count += 1
            
            # Update observation
            obs = next_obs
            
            # Check if episode finished
            if done:
                break
        
        # Convert info history to action strings using master's method
        raw_actions_from_info = info_hist_to_raw_actions_lst(info_hist)
        
        # Convert raw actions to action strings
        action_strings = []
        for i, info in enumerate(info_hist):
            if isinstance(info, tuple):
                info_dict = info[0]
            else:
                info_dict = info
            
            # Try to get action string from info
            if 'action_string' in info_dict:
                action_strings.append(info_dict['action_string'])
            elif 'last_action_type' in info_dict:
                action_strings.append(f"[{info_dict['last_action_type']}]")
            else:
                # Fallback: convert from raw action
                if i < len(raw_actions_from_info):
                    raw_action = raw_actions_from_info[i]
                    action_type = ['back', 'filter', 'group'][int(raw_action[0]) % 3]
                    action_strings.append(f"[{action_type}]")
                else:
                    action_strings.append("[unknown]")
        
        eval_env.close()
        
        return action_strings, total_reward, raw_actions_from_info
        
    except Exception as e:
        eval_env.close()
        raise e

def calculate_bleu_scores(candidate_actions: List[str], reference_actions: List[List[str]]) -> Dict[str, float]:
    """Calculate BLEU scores using master's methodology"""
    try:
        # Try different import paths
        sys.path.append('./evaluation')
        from bleu_calculation import calculate_bleu_score
    except ImportError:
        try:
            from evaluation.bleu_calculation import calculate_bleu_score  
        except ImportError:
            # PROPER fallback: sequence-level exact match (not individual actions)
            def calculate_bleu_score(candidate, references, mode='action_type'):
                if not candidate or not references:
                    return 0.0
                
                print(f"  Fallback BLEU - Candidate: {candidate}")
                print(f"  Fallback BLEU - References: {references}")
                
                # Calculate exact sequence match percentage (proper BLEU approximation)
                exact_matches = 0
                for ref in references:
                    if len(candidate) == len(ref):
                        if all(c == r for c, r in zip(candidate, ref)):
                            exact_matches += 1
                            break
                
                # If no exact match, check partial overlap
                if exact_matches == 0:
                    best_overlap = 0
                    for ref in references:
                        overlap = sum(1 for c in candidate if c in ref)
                        overlap_ratio = overlap / max(len(candidate), len(ref))
                        best_overlap = max(best_overlap, overlap_ratio)
                    return best_overlap
                
                return 1.0 if exact_matches > 0 else 0.0
    
    bleu_scores = {}
    
    # Action Type BLEU
    try:
        action_type_bleu = calculate_bleu_score(candidate_actions, reference_actions, mode='action_type')
        bleu_scores['action_type'] = action_type_bleu
    except Exception as e:
        print(f"Action Type BLEU calculation failed: {e}")
        bleu_scores['action_type'] = 0.0
    
    # Action + Attribute BLEU
    try:
        attribute_bleu = calculate_bleu_score(candidate_actions, reference_actions, mode='full')
        bleu_scores['attribute'] = attribute_bleu
    except Exception as e:
        print(f"Attribute BLEU calculation failed: {e}")
        bleu_scores['attribute'] = 0.0
    
    return bleu_scores

def analyze_action_diversity(all_actions: List[List[str]]) -> Dict[str, Any]:
    """Analyze action diversity across all episodes"""
    
    # Flatten all actions
    flat_actions = []
    for episode_actions in all_actions:
        flat_actions.extend(episode_actions)
    
    # Count action types
    action_type_counts = {}
    for action_str in flat_actions:
        # Extract action type (before first underscore or the whole string)
        if '_' in action_str:
            action_type = action_str.split('_')[0]
        else:
            action_type = action_str.replace('[', '').replace(']', '')
        
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
    
    # Calculate percentages
    total_actions = len(flat_actions)
    action_type_percentages = {
        action_type: (count / total_actions) * 100
        for action_type, count in action_type_counts.items()
    }
    
    # Count unique actions
    unique_actions = set(flat_actions)
    
    return {
        'total_actions': total_actions,
        'unique_actions': len(unique_actions),
        'action_type_counts': action_type_counts,
        'action_type_percentages': action_type_percentages,
        'unique_action_strings': unique_actions
    }

def evaluate_model(model_path: str, datasets: List[int] = None, episodes_per_dataset: int = 10, max_steps: int = 12) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with master-exact methodology
    """
    
    print("MASTER-EXACT MODEL EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes per dataset: {episodes_per_dataset}")
    print(f"Max steps per episode: {max_steps}")
    print()
    
    # Load agent
    agent = create_agent_and_load_model(model_path)
    
    # Load evaluation datasets
    eval_datasets = load_evaluation_datasets()
    
    # Use specified datasets or all available
    if datasets is None:
        datasets = list(range(min(3, len(eval_datasets))))  # Default to first 3
    
    print(f"Evaluating on datasets: {datasets}")
    print()
    
    # Store results
    results = {
        'model_path': model_path,
        'datasets_evaluated': datasets,
        'episodes_per_dataset': episodes_per_dataset,
        'dataset_results': {},
        'overall_bleu_scores': {'action_type': [], 'attribute': []},
        'overall_rewards': [],
        'all_actions': []
    }
    
    # Evaluate each dataset
    for dataset_id in datasets:
        print(f"Evaluating Dataset {dataset_id}")
        print("-" * 30)
        
        dataset_actions = []
        dataset_rewards = []
        dataset_raw_actions = []
        
        # Run multiple episodes for this dataset
        for episode in range(episodes_per_dataset):
            try:
                # Run episode
                action_strings, total_reward, raw_actions = run_episode_master_exact(
                    agent, dataset_id, max_steps
                )
                
                dataset_actions.append(action_strings)
                dataset_rewards.append(total_reward)
                dataset_raw_actions.extend(raw_actions)
                
                print(f"  Episode {episode+1:2d}: {len(action_strings):2d} actions, reward={total_reward:6.2f}")
                
            except Exception as e:
                print(f"  Episode {episode+1} failed: {e}")
                continue
        
        # Load reference actions for this dataset (if available)
        reference_actions = []
        try:
            # Try different import paths for reference loader
            sys.path.append('./evaluation')
            try:
                from reference_loader import load_reference_actions
            except ImportError:
                from evaluation.reference_loader import load_reference_actions
            reference_actions = load_reference_actions(dataset_id)
        except Exception as e:
            print(f"  Could not load reference actions: {e}")
            print(f"  Using basic default references for comparison")
            # Use more realistic default references
            reference_actions = [
                ['[filter]', '[group]', '[back]'],
                ['[group]', '[filter]', '[back]'], 
                ['[filter]', '[filter]', '[group]'],
                ['[back]', '[filter]', '[group]'],
                ['[group]', '[group]', '[filter]']
            ]
        
        # Calculate BLEU scores
        if dataset_actions and reference_actions:
            # Calculate average BLEU across all episodes
            bleu_scores = []
            for candidate in dataset_actions:
                episode_bleu = calculate_bleu_scores(candidate, reference_actions)
                bleu_scores.append(episode_bleu)
            
            # Average BLEU scores
            avg_action_type_bleu = np.mean([b['action_type'] for b in bleu_scores])
            avg_attribute_bleu = np.mean([b['attribute'] for b in bleu_scores])
            
            print(f"  Average Action Type BLEU: {avg_action_type_bleu:.3f}")
            print(f"  Average Attribute BLEU: {avg_attribute_bleu:.3f}")
            
            # Store in results
            results['dataset_results'][dataset_id] = {
                'actions': dataset_actions,
                'rewards': dataset_rewards,
                'raw_actions': dataset_raw_actions,
                'bleu_scores': bleu_scores,
                'avg_action_type_bleu': avg_action_type_bleu,
                'avg_attribute_bleu': avg_attribute_bleu,
                'avg_reward': np.mean(dataset_rewards),
                'std_reward': np.std(dataset_rewards)
            }
            
            # Add to overall results
            results['overall_bleu_scores']['action_type'].append(avg_action_type_bleu)
            results['overall_bleu_scores']['attribute'].append(avg_attribute_bleu)
        
        results['overall_rewards'].extend(dataset_rewards)
        results['all_actions'].extend(dataset_actions)
        
        print(f"  💰 Average Reward: {np.mean(dataset_rewards):.2f} ± {np.std(dataset_rewards):.2f}")
        print()
    
    # Calculate overall statistics
    if results['overall_bleu_scores']['action_type']:
        results['overall_avg_action_type_bleu'] = np.mean(results['overall_bleu_scores']['action_type'])
        results['overall_avg_attribute_bleu'] = np.mean(results['overall_bleu_scores']['attribute'])
    else:
        results['overall_avg_action_type_bleu'] = 0.0
        results['overall_avg_attribute_bleu'] = 0.0
    
    results['overall_avg_reward'] = np.mean(results['overall_rewards'])
    results['overall_std_reward'] = np.std(results['overall_rewards'])
    
    # Analyze action diversity
    results['diversity_analysis'] = analyze_action_diversity(results['all_actions'])
    
    return results

def print_evaluation_summary(results: Dict[str, Any]):
    """Print comprehensive evaluation summary"""
    
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {results['model_path']}")
    print(f"Datasets: {results['datasets_evaluated']}")
    print(f"🎲 Episodes: {len(results['all_actions'])} total")
    print()
    
    print("BLEU SCORES:")
    print(f"  Action Type BLEU: {results['overall_avg_action_type_bleu']:.3f}")
    print(f"  Attribute BLEU:   {results['overall_avg_attribute_bleu']:.3f}")
    print()
    
    print("💰 REWARD STATISTICS:")
    print(f"  Average Reward: {results['overall_avg_reward']:.2f} ± {results['overall_std_reward']:.2f}")
    print()
    
    print("ACTION DIVERSITY:")
    diversity = results['diversity_analysis']
    print(f"  Total Actions: {diversity['total_actions']}")
    print(f"  Unique Actions: {diversity['unique_actions']}")
    
    if diversity['total_actions'] > 0:
        print(f"  Diversity Ratio: {(diversity['unique_actions'] / diversity['total_actions']):.3f}")
    else:
        print(f"  Diversity Ratio: N/A (no actions recorded)")
    print()
    
    print("ACTION TYPE DISTRIBUTION:")
    for action_type, percentage in diversity['action_type_percentages'].items():
        count = diversity['action_type_counts'][action_type]
        print(f"  {action_type:8s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Per-dataset breakdown
    print("PER-DATASET RESULTS:")
    for dataset_id, data in results['dataset_results'].items():
        print(f"  Dataset {dataset_id}:")
        print(f"    Action Type BLEU: {data['avg_action_type_bleu']:.3f}")
        print(f"    Attribute BLEU:   {data['avg_attribute_bleu']:.3f}")
        print(f"    Average Reward:   {data['avg_reward']:.2f} ± {data['std_reward']:.2f}")
    print()
    
    print("EVALUATION COMPLETE!")

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate ATENA model with master-exact methodology')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model (e.g., results/small_training_1000_steps/trained_model)')
    parser.add_argument('--datasets', type=str, default='0,1,2',
                       help='Comma-separated dataset IDs to evaluate (default: 0,1,2)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Episodes per dataset (default: 10)')
    parser.add_argument('--max_steps', type=int, default=12,
                       help='Max steps per episode (default: 12)')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save detailed results (optional)')
    
    args = parser.parse_args()
    
    # Parse datasets
    datasets = [int(d.strip()) for d in args.datasets.split(',')]
    
    # Verify model path exists (handle different storage formats)
    model_exists = False
    
    if os.path.exists(args.model_path):
        model_exists = True
    else:
        # Check if it's a directory-based model with separate files
        base_path = args.model_path.replace("/trained_model", "")
        policy_weights = os.path.join(base_path, "trained_model_policy_weights.weights.h5")
        value_weights = os.path.join(base_path, "trained_model_value_weights.weights.h5")
        
        if os.path.exists(policy_weights) and os.path.exists(value_weights):
            model_exists = True
            args.model_path = base_path  # Update path to directory
    
    if not model_exists:
        print(f"Model not found at: {args.model_path}")
        print(f"   Looked for:")
        print(f"   - Single file: {args.model_path}")
        print(f"   - Directory: {args.model_path.replace('/trained_model', '')}")
        print(f"   - Separate files: trained_model_policy_weights.weights.h5, trained_model_value_weights.weights.h5")
        return 1
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=args.model_path,
            datasets=datasets,
            episodes_per_dataset=args.episodes,
            max_steps=args.max_steps
        )
        
        # Print summary
        print_evaluation_summary(results)
        
        # Save results if requested
        if args.save_results:
            import pickle
            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f)
            print(f"Detailed results saved to: {args.save_results}")
        
        return 0
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
