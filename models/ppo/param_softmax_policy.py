"""
FFParamSoftmax Policy Implementation for TensorFlow
==================================================

This implements the Parameterized Softmax policy that master used for successful evaluations.
Based on master's FFParamSoftmax architecture with 600 hidden channels.

Key Differences from FFGaussian:
- Uses discrete parametric softmax distributions
- Has separate segments for different action types
- Master achieved BLEU 0.33-0.56 with this architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Optional


class ParamSoftmaxPolicy(keras.Model):
    """
    Parameterized Softmax policy that uses multiple softmax distributions 
    for different action types and their parameters.
    
    Based on master's successful FFParamSoftmax architecture.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 n_hidden_layers: int = 2,
                 n_hidden_channels: int = 600,  # Master uses 600 for FFParamSoftmax
                 parametric_segments: Tuple = None,
                 pre_output_size: int = None,
                 beta: float = 1.0,
                 name="param_softmax_policy"):
        """
        Args:
            obs_dim: Observation space dimension
            n_hidden_layers: Number of hidden layers  
            n_hidden_channels: Hidden layer width (master uses 600)
            parametric_segments: Segments for each action type
            pre_output_size: Size of pre-output layer
            beta: Temperature parameter for softmax
        """
        super().__init__(name=name)
        
        self.obs_dim = obs_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.beta = beta
        
        # Use EXACT master segments - will be set dynamically from environment
        if parametric_segments is None:
            # Default to master's NETWORKING CUSTOM_WIDTH: back=(), filter=(12, 3, 26), group=(12,)
            # But this will be overridden by the environment's actual values
            self.parametric_segments = (
                tuple(),  # back - no parameters
                (12, 3, 26),  # filter - (12 cols, 3 ops, 26 CUSTOM_WIDTH bins)
                (12,),  # group - (12 cols)
            )
            print("Using default parametric segments - should be overridden by environment!")
        else:
            self.parametric_segments = parametric_segments
            print(f"Using environment parametric segments: {parametric_segments}")
            
        # Calculate pre-output size if not provided
        if pre_output_size is None:
            self.pre_output_size = self._calculate_pre_output_size()
        else:
            self.pre_output_size = pre_output_size
            
        # Build the network
        self._build_network()
        
        print(f"FFParamSoftmax Policy initialized:")
        print(f"  - Hidden channels: {n_hidden_channels} (master's 600)")
        print(f"  - Parametric segments: {self.parametric_segments}")
        print(f"  - Pre-output size: {self.pre_output_size}")
        print(f"  - Beta (temperature): {beta}")
    
    def _calculate_pre_output_size(self) -> int:
        """Calculate the size of pre-output layer based on segments"""
        # From master: 3 action types + sum of all segment parameters
        total_size = len(self.parametric_segments)  # Action types
        
        # Add parameter sizes for each action type
        for segment in self.parametric_segments:
            if segment:  # If action type has parameters
                for param_size in segment:
                    total_size += param_size
                    
        return total_size
    
    def _build_network(self):
        """Build the feedforward network with master's architecture"""
        # Hidden layers with master's specifications
        self.hidden_layers = []
        
        for i in range(self.n_hidden_layers):
            hidden = layers.Dense(
                self.n_hidden_channels,
                activation='relu',  # FIXED: Master uses ReLU activation (F.relu)
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name=f'hidden_{i}'
            )
            self.hidden_layers.append(hidden)
            
        # Pre-output layer
        self.pre_output_layer = layers.Dense(
            self.pre_output_size,
            activation=None,  # No activation - logits
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros',
            name='pre_output'
        )
        
        print(f"FFParamSoftmax network built with {self.n_hidden_layers} layers x {self.n_hidden_channels} units")
    
    def call(self, obs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the network
        
        Args:
            obs: Observations tensor [batch_size, obs_dim]
            training: Whether in training mode
            
        Returns:
            logits: Pre-output logits [batch_size, pre_output_size]
        """
        x = obs
        
        # Pass through hidden layers
        for hidden in self.hidden_layers:
            x = hidden(x, training=training)
            
        # Pre-output logits
        logits = self.pre_output_layer(x, training=training)
        
        return logits
    
    def get_action_probabilities(self, logits: tf.Tensor) -> tf.Tensor:
        """
        MASTER-EXACT: Convert logits to action probabilities using parametric softmax
        Matches master's exact algorithm from parameterized_softmax.py
        
        Args:
            logits: Raw logits [batch_size, pre_output_size]
            
        Returns:
            probs: Action probabilities [batch_size, total_actions]
        """
        num_action_types = len(self.parametric_segments)
        
        # Action type probabilities (first N logits for action types)
        action_type_logits = logits[:, :num_action_types]
        action_types_probs = tf.nn.softmax(self.beta * action_type_logits, axis=-1)
        
        # Process each action type EXACTLY like master
        result = []
        logits_offset = num_action_types
        
        for i, segment in enumerate(self.parametric_segments):
            action_type_prob = action_types_probs[:, i:i+1]  # [batch, 1]
            
            if not segment:  # No parameters for this action type (like 'back')
                result.append(action_type_prob)
            else:
                # MASTER'S EXACT ALGORITHM: Complex tiling for parameter combinations
                segments_factor = 1
                
                # Calculate total segment factor
                for sub_seg_size in segment:
                    segments_factor *= sub_seg_size
                
                # Start with action type probability
                current_prob = action_type_prob
                
                # Apply each parameter segment
                for sub_seg_size in segment:
                    # Get parameter probabilities
                    param_logits = logits[:, logits_offset:logits_offset + sub_seg_size]
                    sub_seg_probs = tf.nn.softmax(self.beta * param_logits, axis=-1)
                    
                    # MASTER'S EXACT TILING LOGIC:
                    # Repeat action_type_prob for each parameter option
                    repeated_prob = tf.repeat(current_prob, sub_seg_size, axis=1)
                    
                    # Tile parameter probs to match repeated structure  
                    current_size = tf.shape(current_prob)[1]
                    tiled_param_probs = tf.tile(sub_seg_probs, [1, current_size])
                    
                    # Multiply probabilities (joint probability)
                    current_prob = repeated_prob * tiled_param_probs
                    
                    logits_offset += sub_seg_size
                    
                result.append(current_prob)
        
        # Concatenate all action probabilities
        return tf.concat(result, axis=1)
    
    def sample_action(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        FIXED: Sample from FULL 949-action distribution (not hierarchical)
        
        The hierarchical structure is used for efficient computation but sampling
        must respect the true discrete action space: 1 back + 936 filter + 12 group.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            
        Returns:
            actions: Sampled action indices [batch_size] 
            log_probs: Log probabilities of sampled actions [batch_size]
        """
        logits = self(obs, training=False)
        probs = self.get_action_probabilities(logits)
        
        # Sample from full 949-dimensional distribution
        # This ensures proper weighting: ~0.11% back, ~98.6% filter, ~1.3% group
        dist = tf.random.categorical(tf.math.log(probs + 1e-8), 1)
        actions = tf.squeeze(dist, axis=1)
        
        # Calculate log probabilities from the full distribution
        log_probs = tf.math.log(tf.gather(probs, actions, batch_dims=1) + 1e-8)
        
        return actions, log_probs
    
    def most_probable_action(self, obs: tf.Tensor) -> tf.Tensor:
        """
        Get the most probable action (deterministic)
        
        Args:
            obs: Observations [batch_size, obs_dim]
            
        Returns:
            actions: Most probable action indices [batch_size]
        """
        logits = self(obs, training=False)
        probs = self.get_action_probabilities(logits)
        
        # Return argmax
        return tf.argmax(probs, axis=1)
    
    def action_to_discrete(self, action_idx: int) -> Tuple[int, Tuple]:
        """
        MASTER-EXACT: Convert action index to discrete action type and parameters
        Matches master's parametric_softmax_idx_to_discrete_action exactly
        
        Args:
            action_idx: Action index
            
        Returns:
            action_type: Action type number (0=back, 1=filter, 2=group)
            parameters: Parameters for the action
        """
        from collections import deque
        
        current_idx = 0
        action_type_num = 0
        result = deque()
        
        for action_num, segment in enumerate(self.parametric_segments):
            # Calculate segment size (matching master's get_seg_size)
            if not segment:
                segment_size = 1
            else:
                segment_size = 1
                for param_size in segment:
                    segment_size *= param_size
            
            if current_idx + segment_size > action_idx:  # Found the right segment
                in_segment_idx = action_idx - current_idx
                action_type_num = action_num
                
                # MASTER'S EXACT PARAMETER DECODING:
                # Decode from right to left (reverse order)
                for i in range(len(segment) - 1, -1, -1):
                    result.appendleft(in_segment_idx % segment[i])
                    in_segment_idx //= segment[i]
                break
            else:
                current_idx += segment_size
        else:  # If we didn't find a suitable segment (no break occurred)
            raise ValueError(f"Invalid action index {action_idx} for segments {self.parametric_segments}")
        
        return action_type_num, tuple(result)


def create_ffparamsoftmax_policy(obs_dim: int, **kwargs) -> ParamSoftmaxPolicy:
    """
    Factory function to create FFParamSoftmax policy
    
    Args:
        obs_dim: Observation dimension
        **kwargs: Additional arguments
        
    Returns:
        ParamSoftmaxPolicy instance
    """
    return ParamSoftmaxPolicy(obs_dim=obs_dim, **kwargs)


if __name__ == "__main__":
    # Test the implementation
    print("Testing FFParamSoftmax Policy...")
    
    policy = ParamSoftmaxPolicy(obs_dim=51)
    
    # Test forward pass
    test_obs = tf.random.normal([1, 51])
    logits = policy(test_obs)
    print(f"Logits shape: {logits.shape}")
    
    # Test action sampling
    actions, log_probs = policy.sample_action(test_obs)
    print(f"Sampled action: {actions.numpy()}")
    print(f"Log prob: {log_probs.numpy()}")
    
    # Test most probable action
    most_prob = policy.most_probable_action(test_obs)
    print(f"Most probable action: {most_prob.numpy()}")
    
    # Test action conversion
    action_type, params = policy.action_to_discrete(int(actions[0]))
    print(f"Action {int(actions[0])} -> Type: {action_type}, Params: {params}")
    
    print("FFParamSoftmax Policy test complete!")
