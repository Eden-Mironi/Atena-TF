"""
Real Vectorized Environments
Based on ATENA-master/envs.py - MultiprocessVectorEnv implementation

Master uses chainerrl.envs.MultiprocessVectorEnv to run multiple environments in parallel.
This is FUNDAMENTAL to how master's training works - not single environment with fake batches!
"""

import gym
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Any, Optional
import sys
import os

# Add paths for ATENA environment
sys.path.append('.')
sys.path.append('./gym_atena')

from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
import Configuration.config as cfg


class ScaleRewardWrapper(gym.Wrapper):
    """
    Reward scaling wrapper matching ChainerRL's ScaleReward
    Master uses: chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
    This scales ALL rewards by a constant factor (default 0.01)
    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale
        
    def step(self, action, **kwargs):
        """Pass through all kwargs to support ATENA's extended step() signature"""
        obs, reward, done, info = self.env.step(action, **kwargs)
        # Scale the reward
        scaled_reward = reward * self.scale
        return obs, scaled_reward, done, info


class ATENAVectorizedEnv:
    """
    Vectorized ATENA environment matching master's MultiprocessVectorEnv
    Runs multiple ATENA environments in parallel for batch training
    
    Based on:
    - ATENA-master/envs.py lines 80-89 (make_batch_env)
    - chainerrl.envs.MultiprocessVectorEnv functionality
    """
    
    def __init__(self, num_envs: int = 1, max_steps: int = 10, gradual_training: bool = True, 
                 seed: int = 0, is_test: bool = False):
        """
        Initialize vectorized ATENA environments
        
        Args:
            num_envs: Number of parallel environments (master's args.num_envs)
            max_steps: Maximum steps per episode  
            gradual_training: Enable gradual training (curriculum learning)
            seed: Random seed base
            is_test: Whether this is for evaluation (affects seeding)
        """
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.gradual_training = gradual_training
        self.is_test = is_test
        
        # Set different random seeds for different environments (matching master)
        # Master logic from envs.py lines 84-85:
        self.process_seeds = np.arange(num_envs) + seed * num_envs
        assert self.process_seeds.max() < 2 ** 32
        
        print(f"Creating {num_envs} vectorized ATENA environments")
        print(f"   Process seeds: {self.process_seeds}")
        print(f"   Max steps: {max_steps}")
        print(f"   Gradual training: {gradual_training}")
        print(f"   Is test: {is_test}")
        
        # Create individual environments
        self.envs = []
        for idx in range(num_envs):
            env = self._make_single_env(idx)
            self.envs.append(env)
            
        # Get spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.spec = self.envs[0].spec
        
        # Track episode states
        self.dones = [False] * num_envs
        
        print(f"Vectorized environments created successfully")
        print(f"   Observation space: {self.observation_space}")
        print(f"   Action space: {self.action_space}")
    
    def _make_single_env(self, process_idx: int):
        """
        Create a single ATENA environment with proper seeding
        Based on master's make_env_for_batch function
        """
        # Use different random seeds for train and test envs (master's logic)
        process_seed = int(self.process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if self.is_test else process_seed
        
        # Create ATENA environment
        env = make_enhanced_atena_env(
            max_steps=self.max_steps,
            gradual_training=self.gradual_training
        )
        
        # Set seed (matching master's seeding)
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        
        # Scale rewards like Master (only for training, not test)
        # Master's logic from envs.py line 71-74
        if not self.is_test:
            env = ScaleRewardWrapper(env, scale=cfg.reward_scale_factor)
            print(f"   Environment {process_idx}: Reward scaling enabled (factor={cfg.reward_scale_factor})")
        
        return env
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments and return batch of initial observations
        Returns: Array of shape (num_envs, obs_dim)
        """
        observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            observations.append(obs)
            self.dones[i] = False
            
        # Convert to numpy array (batch format)
        batch_obs = np.array(observations, dtype=np.float32)
        
        return batch_obs
    
    def step(self, actions: np.ndarray, compressed: bool = False, filter_by_field: bool = True, continuous_filter_term: bool = True, filter_term=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        MASTER-EXACT: Step all environments with batch of actions
        Signature matches master's exact parameters: compressed, filter_by_field, continuous_filter_term, filter_term
        
        Args:
            actions: Batch of actions, shape (num_envs, action_dim)
            compressed: Whether to apply compressed action scaling (master default: True)
            filter_by_field: Whether to filter by field (master default: True)
            continuous_filter_term: Whether to use continuous filter terms (master default: True)
            filter_term: Filter term parameter (master default: None)
            **kwargs: Additional parameters for compatibility
            
        Returns:
            observations: Batch observations (num_envs, obs_dim)
            rewards: Batch rewards (num_envs,)
            dones: Batch done flags (num_envs,)
            infos: List of info dicts
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                # If environment was done, reset it
                obs = env.reset()
                reward = 0.0
                done = False
                info = {}
                self.dones[i] = False
            else:
                # Step the environment with all master-exact parameters
                obs, reward, done, info = env.step(
                    action, 
                    compressed=compressed,
                    filter_by_field=filter_by_field,
                    continuous_filter_term=continuous_filter_term,
                    filter_term=filter_term,
                    **kwargs
                )
                self.dones[i] = done
                
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        # Convert to numpy arrays
        batch_obs = np.array(observations, dtype=np.float32)
        batch_rewards = np.array(rewards, dtype=np.float32)
        batch_dones = np.array(dones, dtype=bool)
        
        return batch_obs, batch_rewards, batch_dones, infos
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
    
    def seed(self, seeds: Optional[List[int]] = None):
        """Set seeds for all environments"""
        if seeds is None:
            seeds = [None] * self.num_envs
        elif len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}")
            
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)


def make_vectorized_atena_env(num_envs: int = 1, max_steps: int = 10, 
                             gradual_training: bool = True, seed: int = 0, 
                             is_test: bool = False) -> ATENAVectorizedEnv:
    """
    Create vectorized ATENA environment matching master's make_batch_env
    
    This is the TensorFlow equivalent of master's:
    chainerrl.envs.MultiprocessVectorEnv([(lambda: make_env_for_batch(...))])
    """
    return ATENAVectorizedEnv(
        num_envs=num_envs,
        max_steps=max_steps,
        gradual_training=gradual_training,
        seed=seed,
        is_test=is_test
    )


# Master's default configuration (based on arguments.py)
def make_training_batch_env(num_envs: int = 1) -> ATENAVectorizedEnv:
    """Create vectorized training environment matching master's configuration"""
    return make_vectorized_atena_env(
        num_envs=num_envs,
        max_steps=cfg.MAX_NUM_OF_STEPS,
        gradual_training=True,  # Master uses gradual training
        seed=0,  # Master's default seed
        is_test=False
    )


def make_evaluation_batch_env(num_envs: int = 1) -> ATENAVectorizedEnv:
    """Create vectorized evaluation environment matching master's configuration"""
    return make_vectorized_atena_env(
        num_envs=num_envs,
        max_steps=cfg.MAX_NUM_OF_STEPS,
        gradual_training=False,  # No gradual training in evaluation
        seed=0,
        is_test=True  # Different seeding for evaluation
    )


if __name__ == "__main__":
    # Test vectorized environments
    print("Testing vectorized ATENA environments...")
    
    # Test with 3 environments
    vec_env = make_training_batch_env(num_envs=3)
    
    # Test reset
    batch_obs = vec_env.reset()
    print(f"Batch observations shape: {batch_obs.shape}")
    
    # Test step
    batch_actions = np.random.randn(3, vec_env.action_space.shape[0])
    batch_obs, batch_rewards, batch_dones, batch_infos = vec_env.step(batch_actions)
    
    print(f"Batch rewards: {batch_rewards}")
    print(f"Batch dones: {batch_dones}")
    print(f"Number of infos: {len(batch_infos)}")
    
    vec_env.close()
    print("Vectorized environments test successful!")
