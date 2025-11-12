"""
Observation Normalizer - Matches ChainerRL's EmpiricalNormalization

Master uses: chainerrl.links.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)
This normalizes observations to zero mean, unit variance, and clips to [-5, 5]
"""

import numpy as np
import tensorflow as tf


class ObservationNormalizer:
    """
    Empirical observation normalization matching ChainerRL's implementation.
    
    Maintains running mean and variance of observations and normalizes them to:
    - Zero mean
    - Unit variance  
    - Clipped to [-clip_threshold, clip_threshold]
    """
    
    def __init__(self, obs_dim, clip_threshold=5.0, epsilon=1e-8):
        """
        Args:
            obs_dim: Dimension of observation space
            clip_threshold: Clip normalized observations to [-clip_threshold, clip_threshold]
            epsilon: Small constant for numerical stability
        """
        self.obs_dim = obs_dim
        self.clip_threshold = clip_threshold
        self.epsilon = epsilon
        
        # Running statistics
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
        self.std = np.ones(obs_dim, dtype=np.float32)
        
    def update(self, obs):
        """
        Update running statistics with new observation(s).
        
        Args:
            obs: Observation(s) to update statistics with
                 Can be single obs (shape: [obs_dim]) or batch (shape: [batch_size, obs_dim])
        """
        # Convert to numpy if needed
        if isinstance(obs, tf.Tensor):
            obs = obs.numpy()
            
        # Ensure 2D shape [batch_size, obs_dim]
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            
        batch_size = obs.shape[0]
        
        # Welford's online algorithm for running mean and variance
        for i in range(batch_size):
            self.count += 1
            delta = obs[i] - self.mean
            self.mean += delta / self.count
            delta2 = obs[i] - self.mean
            self.var += (delta * delta2 - self.var) / self.count
            
        # Update standard deviation
        self.std = np.sqrt(self.var + self.epsilon)
        
    def normalize(self, obs, update=True):
        """
        Normalize observation(s) using running statistics.
        
        Args:
            obs: Observation(s) to normalize
                 Can be single obs (shape: [obs_dim]) or batch (shape: [batch_size, obs_dim])
            update: Whether to update running statistics with this observation
            
        Returns:
            Normalized observation(s) with same shape as input
        """
        # Convert to numpy if needed
        if isinstance(obs, tf.Tensor):
            obs = obs.numpy()
            
        # Remember original shape
        original_shape = obs.shape
        single_obs = obs.ndim == 1
        
        # Ensure 2D shape
        if single_obs:
            obs = obs.reshape(1, -1)
            
        # Update statistics if requested
        if update:
            self.update(obs)
            
        # Normalize: (obs - mean) / std
        normalized = (obs - self.mean) / self.std
        
        # Clip to [-clip_threshold, clip_threshold]
        normalized = np.clip(normalized, -self.clip_threshold, self.clip_threshold)
        
        # Restore original shape
        if single_obs:
            normalized = normalized.reshape(original_shape)
            
        return normalized.astype(np.float32)
    
    def __call__(self, obs, update=True):
        """Convenience method to normalize observations."""
        return self.normalize(obs, update=update)
    
    def get_state(self):
        """Get normalizer state for saving."""
        return {
            'count': self.count,
            'mean': self.mean,
            'var': self.var,
            'std': self.std,
            'obs_dim': self.obs_dim,
            'clip_threshold': self.clip_threshold,
            'epsilon': self.epsilon
        }
    
    def set_state(self, state):
        """Restore normalizer state from saved state."""
        self.count = state['count']
        self.mean = state['mean']
        self.var = state['var']
        self.std = state['std']
        self.obs_dim = state['obs_dim']
        self.clip_threshold = state['clip_threshold']
        self.epsilon = state['epsilon']

