#!/usr/bin/env python3
"""
Reward Stabilization System for ATENA TensorFlow Implementation

Addresses the convergence issues identified by professor's analysis:
- Diversity component instability (±4.463 variance)
- Humanity component volatility (±2.674 variance) 
- Interestingness component oscillations

Implements: 
1. Exponential Moving Averages (EMA) for smoothing
2. Reward clipping to prevent extreme jumps
3. Adaptive penalty scaling
4. Numerical stability improvements
"""

import numpy as np
from collections import deque
import Configuration.config as cfg


class RewardStabilizer:
    """
    Stabilizes volatile reward components to achieve master-like convergence
    """
    
    def __init__(self, alpha=0.9, clip_factor=0.7, adaptive_penalties=True):
        """
        Initialize reward stabilizer
        
        Args:
            alpha: EMA smoothing factor (0.9 = smooth, 0.1 = responsive)
            clip_factor: Maximum reward change as fraction of current coefficient
            adaptive_penalties: Whether to use adaptive penalty scaling
        """
        self.alpha = alpha
        self.clip_factor = clip_factor  
        self.adaptive_penalties = adaptive_penalties
        
        # Component-specific EMAs
        self.diversity_ema = None
        self.humanity_ema = None
        self.interestingness_ema = None
        self.kl_distance_ema = None
        self.compaction_gain_ema = None
        
        # Penalty tracking for adaptive scaling
        self.recent_penalties = deque(maxlen=100)  # Track recent penalty frequency
        self.penalty_scale_factor = 1.0
        
        # Stability metrics
        self.reward_history = {
            'diversity': deque(maxlen=50),
            'humanity': deque(maxlen=50),
            'interestingness': deque(maxlen=50)
        }
    
    def update_ema(self, current_ema, new_value):
        """Update exponential moving average"""
        if current_ema is None:
            return float(new_value)
        return self.alpha * current_ema + (1 - self.alpha) * new_value
    
    def clip_reward_change(self, new_reward, current_ema, max_coeff):
        """
        Clip reward changes to prevent extreme jumps
        Allows gradual convergence instead of oscillations
        """
        if current_ema is None:
            return new_reward
        
        max_change = max_coeff * self.clip_factor
        change = new_reward - current_ema
        
        # Clip extreme changes
        if abs(change) > max_change:
            clipped_change = np.sign(change) * max_change
            return current_ema + clipped_change
        
        return new_reward
    
    def adaptive_penalty_scaling(self, penalty_value, penalty_type="same_display"):
        """
        Scale penalties based on recent frequency to prevent over-penalization
        """
        if not self.adaptive_penalties:
            return penalty_value
        
        # Track this penalty
        self.recent_penalties.append(penalty_type)
        
        # Calculate recent penalty frequency
        if len(self.recent_penalties) >= 10:
            recent_freq = len([p for p in self.recent_penalties if p == penalty_type]) / len(self.recent_penalties)
            
            # Reduce penalty severity if too frequent (prevents oscillation)
            if recent_freq > 0.3:  # More than 30% of recent steps
                self.penalty_scale_factor = max(0.3, self.penalty_scale_factor * 0.95)
            else:
                self.penalty_scale_factor = min(1.0, self.penalty_scale_factor * 1.02)
        
        return penalty_value * self.penalty_scale_factor
    
    def stabilize_diversity_reward(self, raw_diversity, is_penalty=False):
        """
        Stabilize diversity rewards with EMA smoothing and clipping
        
        Addresses the ±4.463 variance causing learning instability
        """
        # Apply adaptive penalty scaling for same display penalties
        if is_penalty:
            stabilized = self.adaptive_penalty_scaling(raw_diversity, "same_display")
        else:
            # Clip extreme diversity values to prevent sudden jumps
            stabilized = self.clip_reward_change(raw_diversity, self.diversity_ema, cfg.diversity_coeff)
        
        # Update EMA
        self.diversity_ema = self.update_ema(self.diversity_ema, stabilized)
        
        # Track for stability analysis
        self.reward_history['diversity'].append(stabilized)
        
        return stabilized
    
    def stabilize_humanity_reward(self, raw_humanity):
        """
        Stabilize humanity rewards to reduce ±2.674 variance
        """
        # Clip extreme humanity changes
        stabilized = self.clip_reward_change(raw_humanity, self.humanity_ema, cfg.humanity_coeff)
        
        # Update EMA  
        self.humanity_ema = self.update_ema(self.humanity_ema, stabilized)
        
        # Track for stability analysis
        self.reward_history['humanity'].append(stabilized)
        
        return stabilized
    
    def stabilize_interestingness_reward(self, raw_interestingness):
        """
        Stabilize interestingness rewards and components
        """
        # Clip extreme changes
        stabilized = self.clip_reward_change(raw_interestingness, self.interestingness_ema, 
                                           max(cfg.kl_coeff, cfg.compaction_coeff))
        
        # Update EMA
        self.interestingness_ema = self.update_ema(self.interestingness_ema, stabilized)
        
        # Track for stability analysis  
        self.reward_history['interestingness'].append(stabilized)
        
        return stabilized
    
    def stabilize_kl_distance(self, raw_kl):
        """Stabilize KL-distance component specifically"""
        self.kl_distance_ema = self.update_ema(self.kl_distance_ema, raw_kl)
        return self.kl_distance_ema
    
    def stabilize_compaction_gain(self, raw_compaction):
        """Stabilize compaction gain component specifically"""  
        self.compaction_gain_ema = self.update_ema(self.compaction_gain_ema, raw_compaction)
        return self.compaction_gain_ema
    
    def get_stability_metrics(self):
        """Get current stability metrics for monitoring"""
        metrics = {}
        
        for component, history in self.reward_history.items():
            if len(history) > 1:
                values = np.array(history)
                metrics[f"{component}_std"] = np.std(values)
                metrics[f"{component}_trend"] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 2 else 0
        
        metrics["penalty_scale_factor"] = self.penalty_scale_factor
        metrics["recent_penalty_freq"] = len(self.recent_penalties) / max(1, len(self.recent_penalties)) if self.recent_penalties else 0
        
        return metrics
    
    def is_converging(self, component, std_threshold=0.5):
        """Check if a component is converging (low recent standard deviation)"""
        if component not in self.reward_history:
            return False
        
        history = list(self.reward_history[component])
        if len(history) < 10:
            return False
        
        recent_std = np.std(history[-10:])  # Last 10 values
        return recent_std < std_threshold


class NumericalStabilityHelper:
    """
    Helper class for numerically stable reward calculations
    """
    
    @staticmethod
    def safe_log(x, epsilon=1e-8):
        """Numerically stable logarithm"""
        return np.log(np.maximum(x, epsilon))
    
    @staticmethod  
    def safe_divide(numerator, denominator, epsilon=1e-8):
        """Numerically stable division"""
        return numerator / (denominator + epsilon)
    
    @staticmethod
    def safe_sqrt(x, epsilon=1e-8):
        """Numerically stable square root"""
        return np.sqrt(np.maximum(x, epsilon))
    
    @staticmethod
    def stable_sigmoid(x, clip_value=50):
        """Numerically stable sigmoid function"""
        x_clipped = np.clip(x, -clip_value, clip_value)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def stable_distance_calculation(vec1, vec2):
        """
        Numerically stable distance calculation between displays/vectors
        
        Prevents NaN/Inf values that could cause reward instability
        """
        if vec1 is None or vec2 is None:
            return 1.0  # Default distance for missing data
        
        try:
            # Convert to arrays for stable computation
            v1 = np.asarray(vec1).flatten()
            v2 = np.asarray(vec2).flatten()
            
            # Ensure same length
            min_len = min(len(v1), len(v2))
            if min_len == 0:
                return 1.0
            
            v1 = v1[:min_len]
            v2 = v2[:min_len]
            
            # Stable Euclidean distance with normalization
            diff = v1 - v2
            distance = np.sqrt(np.sum(diff * diff))
            
            # Normalize by vector magnitudes for stability
            magnitude = NumericalStabilityHelper.safe_sqrt(np.sum(v1 * v1)) + NumericalStabilityHelper.safe_sqrt(np.sum(v2 * v2))
            normalized_distance = NumericalStabilityHelper.safe_divide(distance, magnitude)
            
            # Ensure valid range [0, 1]
            return np.clip(normalized_distance, 0.0, 1.0)
            
        except (ValueError, TypeError, FloatingPointError):
            # Return default distance on any numerical error
            return 1.0


# Global stabilizer instance
_global_stabilizer = None

def get_reward_stabilizer(reset=False):
    """Get global reward stabilizer instance"""
    global _global_stabilizer
    if _global_stabilizer is None or reset:
        _global_stabilizer = RewardStabilizer(
            alpha=0.85,  # Smooth but responsive
            clip_factor=0.6,  # Allow 60% of coefficient as max change  
            adaptive_penalties=True
        )
    return _global_stabilizer
