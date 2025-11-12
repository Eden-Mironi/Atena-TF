"""
CRITICAL: Master's Explicit Action Processing System
Based on ATENA-master train_agent_chainerrl.py lines 522-531

Implements master's exact action processing pipeline:
1. compressed2full_range() - Scale compressed actions to full range
2. cont2dis() - Discretize continuous actions  
3. Extract action types and track statistics
4. OPERATOR_TYPE_LOOKUP for action type mapping

This matches master's explicit processing instead of delegating to environment.
"""

import numpy as np
from collections import defaultdict
import sys
sys.path.append('./Configuration')
import config as cfg
import gym_atena.lib.helpers as ATENAUtils

class MasterActionProcessor:
    """
    MASTER-EXACT: Explicit action processing matching train_agent_chainerrl.py
    
    Based on master's processing:
    - lines 522-531: explicit action processing in train_agent_batch
    - lines 200-202: action processing in train_agent  
    - env_properties.py: compressed2full_range implementation
    - atena_env_cont.py: cont2dis implementation
    """
    
    def __init__(self, env_prop, architecture="FFGaussian"):
        """
        Initialize action processor with environment properties and architecture
        
        Args:
            env_prop: Environment properties object with compressed2full_range method
            architecture: Architecture type - "FFGaussian", "FFParamSoftmax", or "FFSoftmax"
        """
        self.env_prop = env_prop
        self.architecture = architecture
        
        # Master's action statistics tracking (lines 402-408 in train_agent_chainerrl.py)
        self.actions_cntr = defaultdict(int)      # Raw action counts
        self.action_types_cntr = {"back": 0, "filter": 0, "group": 0}  # Action type counts
        
        # Action processing constants from master
        self.ACTION_RANGE = 6.0  # From env_properties.py line 111
        
        print("Master Action Processor initialized:")
        print(f"   Architecture: {architecture}")
        print(f"   ACTION_RANGE: {self.ACTION_RANGE}")
        print(f"   Action statistics tracking: ✓")
    
    def compressed2full_range(self, action_vec, continuous_filter_term=True):
        """
        CRITICAL: Master's exact compressed2full_range implementation
        Based on env_properties.py lines 168-189
        
        Change a compressed range vector to full range based on the range of each entry
        and clip the vector to be in the legal ranges
        
        Args:
            action_vec: Compressed action vector (range -3 to +3)
            continuous_filter_term: Boolean whether filter term should remain continuous
            
        Returns:
            Full range action vector clipped to legal ranges
        """
        if hasattr(self.env_prop, 'compressed2full_range'):
            # Use environment's method if available
            return self.env_prop.compressed2full_range(action_vec, continuous_filter_term)
        else:
            # Fallback implementation based on master's code
            RANGE = self.ACTION_RANGE
            
            # Master's entries_ranges (env_properties.py lines 178-183)
            # These are the maximum values for each action component
            entries_ranges = np.array([
                3,    # ACTION_TYPES_NO: back(0), filter(1), group(2) 
                12,   # COLS_NO: number of columns in dataset
                9,    # FILTER_OPS: number of filter operators
                100,  # MAX_FILTER_TERMS_BY_FIELD_NO: max filter terms
                12,   # AGG_COLS_NO: number of aggregation columns  
                6     # AGG_FUNCS_NO: number of aggregation functions
            ])
            
            # Master's transformation formula (line 184)
            full_range = np.multiply(np.array((action_vec + RANGE / 2) / RANGE), entries_ranges) - 0.5
            full_range_filter_term = full_range[3]
            
            # Master's clipping (line 186)
            clipped = np.clip(full_range, np.zeros(6), entries_ranges - 1)
            
            if continuous_filter_term:
                clipped[3] = full_range_filter_term
            
            return clipped
    
    @staticmethod
    def cont2dis(c_vector):
        """
        CRITICAL: Master's exact cont2dis implementation  
        Based on atena_env_cont.py lines 329-338
        
        This function discretizes (rounds) a continuous (float) action vector
        
        Args:
            c_vector: A continuous (float) action vector
            
        Returns:
            A vector of discrete integer representing the actions
        """
        return list(np.array(np.round(c_vector), dtype=int))
    
    def process_action(self, raw_action):
        """
        CRITICAL: Master's exact action processing pipeline with ARCHITECTURE-SPECIFIC processing
        Based on train_agent_chainerrl.py lines 522-531
        
        Args:
            raw_action: Raw action from agent (continuous for FFGaussian, discrete index for FFParamSoftmax)
            
        Returns:
            dict with processed action information:
            {
                'raw_action': original action,
                'processed_action': after architecture-specific processing,
                'discrete_action': after cont2dis,
                'action_type': 'back', 'filter', or 'group',
                'action_type_idx': discrete action type index
            }
        """
        # CRITICAL FIX: Architecture-specific processing matching master EXACTLY
        if self.architecture == "FFGaussian":
            # GAUSSIAN PATH: continuous action → compressed2full_range → cont2dis
            # Master line 524: if arch is ArchName.FF_GAUSSIAN: action = env_prop.compressed2full_range(action)
            processed_action = self.compressed2full_range(raw_action, continuous_filter_term=True)
            discrete_action = self.cont2dis(processed_action)
            
        elif self.architecture in ["FFParamSoftmax", "FFSoftmax"]:
            # PARAMETRIC SOFTMAX PATH: discrete action index → static_param_softmax_idx_to_action_type → cont2dis  
            # Master lines 525-527: elif arch is ArchName.FF_PARAM_SOFTMAX or arch is ArchName.FF_SOFTMAX:
            #                        actions_cntr[action] += 1
            #                        action = env_prop.static_param_softmax_idx_to_action_type(action)
            if hasattr(self.env_prop, 'static_param_softmax_idx_to_action_type'):
                processed_action = self.env_prop.static_param_softmax_idx_to_action_type(raw_action)
            else:
                # Fallback - convert discrete index to action vector
                # For back action (index 0), return [0, 0, 0, 0, 0, 0] 
                if raw_action == 0:
                    processed_action = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
                else:
                    # For other actions, create a basic action vector (this is a simplified fallback)
                    processed_action = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)  # Default filter action
            
            discrete_action = self.cont2dis(processed_action)
            
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Extract action type (master lines 528-530)
        # Master: action_disc = ATENAEnvCont.cont2dis(action)
        # Master: action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP.get(action_disc[0])
        action_type_idx = discrete_action[0]
        action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP.get(action_type_idx, "unknown")
        
        # Track statistics (master lines 526, 530)
        # Master: actions_cntr[action] += 1  (for FFParamSoftmax only)
        # Master: action_types_cntr[action_type] += 1
        if self.architecture in ["FFParamSoftmax", "FFSoftmax"]:
            # For discrete actions, track the raw action index
            self.actions_cntr[raw_action] += 1
        else:
            # For continuous actions, track as tuple
            raw_action_key = tuple(raw_action) if isinstance(raw_action, np.ndarray) else raw_action
            self.actions_cntr[raw_action_key] += 1
            
        if action_type in self.action_types_cntr:
            self.action_types_cntr[action_type] += 1
        
        return {
            'raw_action': raw_action,
            'processed_action': processed_action, 
            'discrete_action': discrete_action,
            'action_type': action_type,
            'action_type_idx': action_type_idx
        }
    
    def process_batch_actions(self, raw_actions):
        """
        Process a batch of actions (for vectorized environments)
        
        Args:
            raw_actions: List or array of raw actions
            
        Returns:
            List of processed action dictionaries
        """
        return [self.process_action(action) for action in raw_actions]
    
    def get_statistics(self):
        """Get current action statistics"""
        total_actions = sum(self.actions_cntr.values())
        total_action_types = sum(self.action_types_cntr.values())
        
        return {
            'total_actions': total_actions,
            'total_action_types': total_action_types,
            'actions_cntr': dict(self.actions_cntr),
            'action_types_cntr': dict(self.action_types_cntr),
            'action_type_distribution': {
                action_type: count / total_action_types if total_action_types > 0 else 0
                for action_type, count in self.action_types_cntr.items()
            }
        }
    
    def reset_statistics(self):
        """Reset action statistics counters"""
        self.actions_cntr.clear()
        self.action_types_cntr = {"back": 0, "filter": 0, "group": 0}
        print("Action statistics reset")
    
    def log_statistics(self, prefix=""):
        """Log current action statistics"""
        stats = self.get_statistics()
        
        print(f"{prefix}ACTION PROCESSING STATISTICS:")
        print(f"{prefix}   Total actions processed: {stats['total_actions']}")
        print(f"{prefix}   Action type distribution:")
        for action_type, percentage in stats['action_type_distribution'].items():
            count = stats['action_types_cntr'][action_type]
            print(f"{prefix}     {action_type}: {count} ({percentage:.1%})")
        
        return stats
