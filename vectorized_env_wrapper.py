"""
CRITICAL FIX: Vectorized Environment Wrapper
Replicates ChainerRL's MultiprocessVectorEnv behavior for true batch processing
"""

import numpy as np
import tensorflow as tf
from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
from typing import List, Tuple, Any

class VectorizedATENAEnv:
    """
    CRITICAL: Vectorized environment wrapper that replicates ChainerRL's 
    MultiprocessVectorEnv behavior for proper PPO batch processing.
    
    Master uses CONCURRENT experiences from parallel environments.
    This wrapper creates the SAME batch processing structure.
    """
    
    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs
        self.envs = [make_enhanced_atena_env() for _ in range(num_envs)]
        
        # Get spaces from first environment (all are identical)
        self.observation_space = self.envs[0].observation_space  
        self.action_space = self.envs[0].action_space
        
        print(f"VectorizedATENAEnv initialized:")
        print(f"   - Number of environments: {num_envs}")
        print(f"   - Observation space: {self.observation_space}")
        print(f"   - Action space: {self.action_space}")
    
    def reset(self, env_mask=None):
        """Reset environments and return batch observations.
        
        Args:
            env_mask: Boolean mask indicating which environments to reset.
                     If None, reset all environments.
        """
        if env_mask is None:
            # Reset all environments
            batch_obs = []
            for env in self.envs:
                obs = env.reset()
                batch_obs.append(obs)
            return np.array(batch_obs)
        else:
            # Reset only masked environments (ChainerRL behavior)
            batch_obs = []
            for i, (env, should_reset) in enumerate(zip(self.envs, env_mask)):
                if should_reset:
                    obs = env.reset()
                else:
                    # Use current observation (don't reset)
                    obs = self._last_obs[i] if hasattr(self, '_last_obs') else env.reset()
                batch_obs.append(obs)
            return np.array(batch_obs)
    
    def step(self, batch_actions, **kwargs):
        """Step all environments with batch actions.
        
        Args:
            batch_actions: Array/list of actions for each environment
            **kwargs: Additional arguments passed to each environment step
            
        Returns:
            batch_obs: Batch of next observations
            batch_rewards: Batch of rewards  
            batch_dones: Batch of done flags
            batch_infos: Batch of info dictionaries
        """
        batch_obs = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []
        
        for env, action in zip(self.envs, batch_actions):
            obs, reward, done, info = env.step(action, **kwargs)
            batch_obs.append(obs)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_infos.append(info)
        
        # Store for potential reset masking
        self._last_obs = batch_obs
        
        return (np.array(batch_obs), 
                np.array(batch_rewards, dtype=np.float32),
                np.array(batch_dones, dtype=bool), 
                batch_infos)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def seed(self, seeds):
        """Set seeds for all environments."""
        if not isinstance(seeds, list):
            seeds = [seeds + i for i in range(self.num_envs)]
        
        for env, seed in zip(self.envs, seeds):
            if hasattr(env, 'seed'):
                env.seed(seed)
    
    def render(self, mode='human'):
        """Render first environment."""
        return self.envs[0].render(mode)
    
    @property
    def env_prop(self):
        """Access environment properties from first environment."""
        return self.envs[0].env_prop


def create_vectorized_env(num_envs: int = 1):
    """Factory function to create vectorized ATENA environment.
    
    CRITICAL: This creates the SAME batch processing structure as 
    ChainerRL's MultiprocessVectorEnv, enabling proper PPO learning.
    """
    return VectorizedATENAEnv(num_envs)
