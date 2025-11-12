"""
TensorFlow Live Recommender Agent - equivalent to ATENA-master RecommenderAgent
"""
import os
import json
import numpy as np
import tensorflow as tf
from models.ppo.agent import PPOAgent
from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
import gym_atena.global_env_prop as gep
import Configuration.config as cfg

class TFRecommenderAgent:
    """TensorFlow implementation of Live Recommender Agent"""
    
    def __init__(self, model_path, dataset_number=0, schema='NETWORKING'):
        """
        Initialize the TF Recommender Agent
        
        Args:
            model_path (str): Path to trained model
            dataset_number (int): Dataset to use (0, 1, 2, 3)
            schema (str): Data schema ('NETWORKING' or 'FLIGHTS')
        """
        self.model_path = model_path
        self.dataset_number = dataset_number
        self.schema = schema
        
        # Set configuration
        cfg.schema = schema
        
        # Set architecture to match training!
        # Models were trained with FF_PARAM_SOFTMAX, not FF_GAUSSIAN
        cfg.arch = 'FF_PARAM_SOFTMAX'
        
        # Create environment
        self.env = make_enhanced_atena_env()
        
        # Initialize state  
        self.state = self.env.reset(dataset_number=dataset_number)
        self.env.max_steps = 1000
        
        # Load trained agent
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        
        # Create agent with SAME architecture as training!
        # Models were trained with discrete parametric softmax policy
        parametric_segments = ((), (12, 3, 26), (12,))
        parametric_segments_sizes = [1, 12*3*26, 12]
        
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=cfg.adam_lr,
            update_interval=2048,
            minibatch_size=64,
            epochs=10,
            clip_ratio=0.2,
            gamma=cfg.ppo_gamma,
            lambda_=cfg.ppo_lambda,
            use_parametric_softmax=True,  # DISCRETE policy like training
            parametric_segments=parametric_segments,
            parametric_segments_sizes=parametric_segments_sizes,
            n_hidden_channels=600,
            beta=1.0,
        )
        
        # Try to load model if it exists (Keras 3 compatible)
        model_loaded = False
        
        # Check for Keras 3 compatible formats first
        if os.path.exists(f"{model_path}_policy_weights.weights.h5") or \
           os.path.exists(f"{model_path}_policy.keras") or \
           os.path.exists(f"{model_path}_policy_weights.index"):
            try:
                success = self.agent.load_model(model_path)
                if success:
                    model_loaded = True
                    print(f"Loaded trained model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        
        if not model_loaded:
            print(f"No compatible trained model found at {model_path}")
            print("   Using untrained model for demonstration")
            print("   Consider retraining the model for Keras 3 compatibility")
        
        # Store last recommendation for reuse
        self.last_recommendation = None
        self.last_action_vector = None
        
        print(f"TF Recommender Agent initialized:")
        print(f"   Schema: {schema}")
        print(f"   Dataset: {dataset_number}")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dim: {action_dim}")
    
    @property
    def original_dataset(self):
        """Get the original dataset"""
        return self.env.data
    
    def act(self, observation=None):
        """Get next action from trained agent (for evaluation compatibility)"""
        if observation is not None:
            self.state = observation
        
        # Get action from trained policy - USE DETERMINISTIC for consistent recommendations
        action, log_prob, value = self.agent.act_most_probable(self.state)
        
        # Convert to action vector
        action_vector = self.env.cont2dis(action)
        
        # Store for potential reuse
        self.last_action_vector = action_vector
        
        return action_vector
    
    def get_action(self, observation=None, deterministic=True):
        """
        Get action from agent (compatible with run_episode interface)
        
        SIMPLIFIED: Use exact same approach as generate_session_output.py
        
        Args:
            observation: Current observation (optional)
            deterministic: If True, use deterministic policy (default: True for compatibility)
        
        Returns:
            action: Action from agent (format depends on policy type)
        """
        if observation is not None:
            self.state = observation
        
        # Use agent.act() just like generate_session_output.py does!
        # Don't use act_most_probable - just use act() for both deterministic and stochastic
        # The environment will handle whatever format the agent returns
        action_result = self.agent.act(self.state)
        
        # Extract action from tuple (action, log_prob, value)
        if isinstance(action_result, tuple):
            action = action_result[0]
        else:
            action = action_result
        
        # Convert TensorFlow tensor to numpy if needed
        if hasattr(action, 'numpy'):
            action = action.numpy()
        
        # Ensure it's a 1D array if it came as 2D from batch
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            action = action.squeeze()
        
        # Store for potential reuse
        self.last_action_vector = action
        
        return action
    
    def get_agent_action(self):
        """Get next action from trained agent"""
        return self.act()
    
    def get_agent_action_str(self):
        """Get human-readable action string from agent"""
        # Get next action
        action_vector = self.get_agent_action()
        
        # Translate to human-readable format
        action_str = self.env.translate_action(action_vector, filter_by_field=True, filter_term=None)
        
        # Store as last recommendation
        self.last_recommendation = action_str
        
        return action_str
    
    def step_with_action(self, action_vector=None):
        """Execute one step with agent's action and return results"""
        # Use last action if none provided
        if action_vector is None:
            if self.last_action_vector is not None:
                action_vector = self.last_action_vector
            else:
                # Get new action if none stored
                action_vector = self.get_agent_action()
        
        # Execute action in environment
        new_state, reward, done, info = self.env.step(
            action_vector, 
            compressed=False, 
            filter_by_field=True,
            continuous_filter_term=False, 
            filter_term=None
        )
        
        # Update state
        self.state = new_state
        
        # Add action description to info
        if 'action_desc' not in info:
            info['action_desc'] = self.env.translate_action(action_vector, filter_by_field=True, filter_term=None)
        
        return new_state, reward, done, info
    
    def apply_custom_action(self, action_vector, filter_term=None):
        """Apply a custom action chosen by user"""
        # Execute action in environment
        new_state, reward, done, info = self.env.step(
            action_vector, 
            compressed=False, 
            filter_by_field=True,
            continuous_filter_term=False, 
            filter_term=filter_term
        )
        
        # Update state
        self.state = new_state
        
        return TFRecommenderStepResult(info, reward)
    
    def apply_agent_action(self, use_last_recommendation=False):
        """Apply the agent's recommended action"""
        if use_last_recommendation and self.last_action_vector is not None:
            action_vector = self.last_action_vector
        else:
            action_vector = self.get_agent_action()
        
        # Execute action in environment  
        new_state, reward, done, info = self.env.step(
            action_vector,
            compressed=False,
            filter_by_field=True,
            continuous_filter_term=True,
            filter_term=None
        )
        
        # Update state
        self.state = new_state
        
        return TFRecommenderStepResult(info, reward)
    
    def reset_environment(self, dataset_number=None):
        """Reset environment to initial state"""
        if dataset_number is not None:
            self.dataset_number = dataset_number
        
        self.state = self.env.reset(dataset_number=self.dataset_number)
        self.last_recommendation = None
        self.last_action_vector = None
        
        print(f"Environment reset to dataset {self.dataset_number}")

class TFRecommenderStepResult:
    """Step result containing display data and reward info"""
    
    def __init__(self, info, reward, verbose=True):
        self.info = info
        self.reward = reward
        self.verbose = verbose
        
        # Extract display data
        if "raw_display" in info:
            self.df_to_display = info["raw_display"][1]  # Aggregated dataframe
            if self.df_to_display is None:
                self.df_to_display = info["raw_display"][0]  # Filtered dataframe
        else:
            self.df_to_display = None
        
        # Extract reward info
        self.reward_info = info.get("reward_info", {})
        
        if verbose:
            self._print_step_info()
    
    def _print_step_info(self):
        """Print step information like original"""
        print(f"Action: {self.info.get('action', 'Unknown')}")
        print(f"Reward: {self.reward:.4f}")
        
        if self.reward_info and hasattr(self.reward_info, 'get'):
            print("Reward components:")
            for key, value in self.reward_info.items():
                print(f"  {key}: {value:.4f}")

def find_latest_trained_model(results_dir="results"):
    """
    Find the most recent trained model in the results directory
    
    Args:
        results_dir: Directory containing training run subdirectories (default: "results")
    
    Returns:
        Path to trained model (e.g., "results/0511-10:50/trained_model")
        or None if no model found
    """
    import glob
    import os
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found")
        return None
    
    # Find all subdirectories with trained models
    model_dirs = []
    
    # Check each subdirectory in results/
    for subdir in os.listdir(results_dir):
        full_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(full_path):
            continue
        
        # Check for trained_model files (Keras 3 format)
        model_base = os.path.join(full_path, "trained_model")
        keras3_path = f"{model_base}_policy_weights.weights.h5"
        
        # Check for best_agent files (alternative location)
        best_agent_base = os.path.join(full_path, "best_agent")
        best_keras3_path = f"{best_agent_base}_policy_weights.weights.h5"
        
        if os.path.exists(keras3_path):
            # Get modification time for sorting
            mtime = os.path.getmtime(keras3_path)
            model_dirs.append((mtime, model_base, "trained_model"))
        elif os.path.exists(best_keras3_path):
            mtime = os.path.getmtime(best_keras3_path)
            model_dirs.append((mtime, best_agent_base, "best_agent"))
    
    if not model_dirs:
        print(f"No trained models found in '{results_dir}/' directory")
        print(f"    Looking for files matching: trained_model_policy_weights.weights.h5")
        return None
    
    # Sort by modification time (newest first)
    model_dirs.sort(reverse=True)
    
    # Return the most recent model
    latest_mtime, latest_path, model_type = model_dirs[0]
    print(f"Found latest model: {latest_path} ({model_type})")
    
    return latest_path

# Example usage and testing
if __name__ == "__main__":
    print("Testing TF Recommender Agent...")
    
    # Find latest model
    model_path = find_latest_trained_model()
    if model_path:
        print(f"Found trained model: {model_path}")
        
        # Create recommender agent
        agent = TFRecommenderAgent(model_path, dataset_number=0)
        
        # Test getting recommendations
        print("\nGetting agent recommendation:")
        recommendation = agent.get_agent_action_str()
        print(f"Recommendation: {recommendation}")
        
        # Test custom action
        print("\nTesting custom action (Back):")
        back_action = [0, 0, 0, 0, 0, 0]  # Back action
        result = agent.apply_custom_action(back_action)
        print("Custom action completed!")
        
    else:
        print("No trained model found. Please run training first.")
