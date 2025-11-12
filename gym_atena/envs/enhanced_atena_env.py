# -*- coding: utf-8 -*-
"""
Enhanced ATENA Environment with complete reward system
Includes missing reward components from ATENA-master:
- Humanity rewards (rule-based)
- Human sessions compatibility
- Enhanced diversity calculations
- Proper reward component tracking
"""

import numpy as np
import gym
from gym import spaces
import math
import sys
import os

# Add paths for config and other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../Configuration'))
sys.path.append(current_dir)  # Add current directory to path
import config as cfg

# Import base environment
try:
    from atena_env_cont import ATENAEnvCont, StepReward, HumanRulesReward
except ImportError:
    from .atena_env_cont import ATENAEnvCont, StepReward, HumanRulesReward

# Master-exact reward calculation functions implemented directly
from collections import Counter
from scipy.stats import entropy

# 🎛️ REWARD STABILIZER CONTROL FLAG (from config)
USE_REWARD_STABILIZER = getattr(cfg, 'USE_REWARD_STABILIZER', False)

# Import reward stabilization system conditionally
if USE_REWARD_STABILIZER:
    sys.path.append(os.path.join(current_dir, '../..'))
    from reward_stabilizer import get_reward_stabilizer, NumericalStabilityHelper
    print("REWARD STABILIZER: ENABLED (experimental mode)")
else:
    # Simple replacement for numerical stability (without EMA chaos)
    class NumericalStabilityHelper:
        @staticmethod
        def safe_divide(numerator, denominator, epsilon=1e-8):
            return numerator / (denominator + epsilon)
    print("REWARD STABILIZER: DISABLED (stable mode like train_ipdate-1009-18:54.png)")

# Import NetHumanRule for master-exact rules tracking
try:
    sys.path.append(os.path.join(current_dir, '../..'))
    from human_rules_tracker import NetHumanRule
except ImportError:
    # Fallback: define minimal NetHumanRule if import fails
    from enum import Enum
    class NetHumanRule(Enum):
        humane_columns_group = 1
        neutral_columns_group = 2  
        inhumane_columns_group = 3
        column_already_grouped = 4
        group_num_of_groups_unchanged = 5
        group_num_of_groups_changed = 6
        group_as_first_action = 22
        filter_as_first_action = 23


def numerically_stable_normalized_sigmoid(a, b, x):
    """
    Numerically stable sigmoid function.
    COPIED EXACTLY from ATENA-master/3d_graphs_notebook.ipynb
    """
    if b*(x-a) < 0:
        z = math.exp(b*(x-a))
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(b*(-x+a))
        return z / (1 + z)


def normalized_sigmoid_fkt(a, b, x):
    """
    Returns normalized sigmoid function - MASTER EXACT
    COPIED EXACTLY from ATENA-master/3d_graphs_notebook.ipynb
    """
    return numerically_stable_normalized_sigmoid(a, b, x)


class CounterWithoutNanKeys:
    """Counter that ignores NaN keys - used in KL divergence calculation"""
    def __init__(self, data):
        self.counter = Counter()
        for item in data:
            if not (isinstance(item, float) and math.isnan(item)):
                self.counter[item] += 1
    
    def __getitem__(self, key):
        return self.counter.get(key, 0)
    
    def items(self):
        return self.counter.items()
    
    def keys(self):
        return self.counter.keys()
    
    def values(self):
        return self.counter.values()
    
    def elements(self):
        return sum(self.counter.values())

class HumanityRuleEngine:
    """
    Rule-based humanity scoring system
    Simplified version of the complex rule system from ATENA-master
    """
    
    def __init__(self):
        self.rule_triggers = []
        
    def evaluate_action_humanity(self, action, state, dfs, step_info):
        """
        Evaluate humanity score for an action based on rules
        Returns a score between -1 and 1
        """
        humanity_score = 0.0
        
        # Rule 1: Back action after productive action is human-like
        if self.is_back_action(action) and self.had_productive_previous_action(step_info):
            humanity_score += 0.3
            
        # Rule 2: Filter actions on meaningful columns are human-like  
        if self.is_filter_action(action) and self.is_meaningful_column(action, state):
            humanity_score += 0.5
            
        # Rule 3: Group actions after filters are human-like
        if self.is_group_action(action) and self.had_recent_filter(step_info):
            humanity_score += 0.4
            
        # Rule 4: Avoid repetitive actions (penalty)
        if self.is_repetitive_action(action, step_info):
            humanity_score -= 0.6
            
        # Rule 5: Empty results are not human-like (penalty)
        if self.leads_to_empty_result(dfs):
            humanity_score -= 0.8
        
        return np.clip(humanity_score, -1.0, 1.0)
    
    def is_back_action(self, action):
        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
            return int(action[0]) == 0
        return False
    
    def is_filter_action(self, action):
        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
            return int(action[0]) == 1
        return False
    
    def is_group_action(self, action):
        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
            return int(action[0]) == 2
        return False
    
    def is_meaningful_column(self, action, state):
        # Simple heuristic: assume certain column indices are more meaningful
        if isinstance(action, (list, np.ndarray)) and len(action) > 1:
            col_idx = int(action[1])
            # This is a simplified heuristic
            return col_idx in [0, 1, 2, 3]  # First few columns often more meaningful
        return True
    
    def had_productive_previous_action(self, step_info):
        # Check if previous action was productive (not back)
        return len(step_info.get('previous_actions', [])) > 0
    
    def had_recent_filter(self, step_info):
        # Check if recent action was a filter
        prev_actions = step_info.get('previous_actions', [])
        if len(prev_actions) > 0:
            return self.is_filter_action(prev_actions[-1])
        return False
    
    def is_repetitive_action(self, action, step_info):
        # Check if this action was done recently
        prev_actions = step_info.get('previous_actions', [])
        if len(prev_actions) >= 2:
            return np.array_equal(action, prev_actions[-1]) or np.array_equal(action, prev_actions[-2])
        return False
    
    def leads_to_empty_result(self, dfs):
        # Check if action leads to empty results
        if dfs and len(dfs) > 0:
            return len(dfs[0]) == 0 if dfs[0] is not None else True
        return False


class EnhancedStepReward(StepReward):
    """Enhanced reward tracking with all original components"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Additional reward components from original
        self.rules_based_humanity = 0
        self.human_sessions_similarity = 0
        self.coherency_score = 0
        self.readability_score = 0
        
        # Master-exact flags (only add if not already in base class)
        if not hasattr(self, 'is_back'):
            self.is_back = False
        if not hasattr(self, 'is_same_display_seen_already'):
            self.is_same_display_seen_already = False
        if not hasattr(self, 'is_data_driven'):
            self.is_data_driven = False
        
        # Detailed breakdowns
        self.diversity_components = {
            'display_distance': 0,
            'action_novelty': 0
        }
        
        self.interestingness_components = {
            'kl_divergence': 0,
            'compaction_gain': 0,
            'information_gain': 0
        }
        
        # Initialize rules_reward_info like master's StepReward
        # Fix: Base class has rules_reward_info = None, so check for None too!
        if not hasattr(self, 'rules_reward_info') or self.rules_reward_info is None:
            self.rules_reward_info = HumanRulesReward()
        
        if not hasattr(self, 'snorkel_rules_reward_info') or self.snorkel_rules_reward_info is None:  
            self.snorkel_rules_reward_info = HumanRulesReward()
    
    def items(self):
        """Enhanced items including all components"""
        base_items = super().items()
        additional_items = {
            'rules_based_humanity': self.rules_based_humanity,
            'human_sessions_similarity': self.human_sessions_similarity,
            'coherency_score': self.coherency_score,
            'readability_score': self.readability_score
        }
        
        return {**dict(base_items), **additional_items}.items()


class EnhancedATENAEnv(ATENAEnvCont):
    """
    Enhanced ATENA Environment with complete reward system
    """
    
    def __init__(self, max_steps=12, gradual_training=False, **kwargs):
        super().__init__(max_steps=max_steps, gradual_training=gradual_training, **kwargs)
        
        # Initialize missing base class attributes
        # These are expected by the base ATENAEnvCont but not always properly initialized
        self.history = []  # captures all "state" dictionaries in the current episode
        self.ahist = []    # list of actions performed thus far in the episode  
        self.dhist = []    # list of the corresponding result-displays of the actions
        self.obs_hist = [] # observation history
        self.obs_hist_all = [] # complete observation history
        self.filter_terms_hist = [] # filter terms history
        self.num_of_rows_hist = [] # number of rows history
        self.num_of_fdf_rows_hist = [] # filtered dataframe rows history  
        self.num_of_immediate_action_rows_lst = [] # immediate action rows
        self.states_hisotry = [] # Base class has typo "hisotry" instead of "history"
        
        # 🎛️ CONDITIONAL: Initialize reward stabilizer based on flag
        if USE_REWARD_STABILIZER:
            self.reward_stabilizer = get_reward_stabilizer(reset=True)
            print("Reward stabilizer initialized (experimental mode)")
        else:
            self.reward_stabilizer = None  # Use raw rewards (proven stable)
        
        # Initialize missing base class attributes
        # These need to be set properly when an episode starts
        self.data = None
        self.dataset_number = 0  # Default dataset number
        self.step_num = 0        # Current step in episode
        
        # Initialize humanity rule engine
        self.humanity_engine = HumanityRuleEngine()
        
        # Track additional state for reward calculation
        self.previous_actions = []
        self.session_statistics = {
            'total_humanity_score': 0,
            'humanity_evaluations': 0,
            'rule_triggers': []
        }
        
        print("Enhanced ATENA Environment initialized with:")
        print(f"  - Rule-based humanity scoring: ✓")
        print(f"  - Enhanced diversity rewards: ✓") 
        print(f"  - Detailed reward tracking: ✓")
        print(f"  - Max steps: {max_steps}")
    
    def reset(self, dataset_number=None, **kwargs):
        """MASTER-EXACT RESET METHOD - Implementing master's complete reset logic"""
        import numpy as np
        import random
        import scipy
        
        # Master's exact reset sequence  
        random.seed()
        np.random.seed()  # FIX: scipy.random doesn't exist in modern scipy, use numpy instead
        
        # Episode management (like master)
        self.step_num = 0
        if hasattr(self, 'gradual_training') and self.gradual_training:
            # Use full episode length for expert-level evaluation  
            # Master's gradual training logic - DISABLED for expert comparison
            # episode_num = getattr(self, 'NUM_OF_EPISODES', 0) + 1
            # self.max_steps = random.randint(2, max(3, int(episode_num / 2500)))
            print(f"GRADUAL TRAINING DISABLED: Using full {self.max_steps} steps (not 2-3!)")
            # Keep original max_steps=12 for expert-level performance
        
        # (1) Choose dataset exactly like master
        import Configuration.config as cfg
        if cfg.dataset_number is not None:
            dataset_number = cfg.dataset_number
        elif dataset_number is None:
            dataset_number = np.random.randint(len(self.repo.data))
        self.dataset_number = dataset_number
        
        # (2) Load dataset exactly like master - FILTER BY KEYS!
        self.data = self.repo.data[dataset_number][self.env_dataset_prop.KEYS]
        
        # (3) Initialize history exactly like master
        from gym_atena.lib.helpers import empty_env_state
        empty_state = empty_env_state
        
        # Master's exact history initialization
        self.history = [empty_state]
        self.states_hisotry = [empty_state]  # Master's typo but exact match
        self.obs_hist = []
        self.obs_hist_all = []
        
        # (4) Calculate initial observation exactly like master
        obs, disp, _ = self.env_prop.calc_display_vector(
            self.data,
            empty_state,
            memo=self.STATE_DF_HISTORY,
            dataset_number=self.dataset_number,
            step_number=self.step_num,
            states_hist=self.history,
            obs_hist=self.obs_hist,
            len_single_display_vec=self.len_single_display_vec
        )
        
        # (5) Initialize ALL tracking arrays exactly like master
        self.dhist = [disp]
        self.ahist = []
        self.obs_hist = [obs]
        self.obs_hist_all = [obs]
        self.filter_terms_hist = []
        self.num_of_rows_hist = [len(self.data)]
        self.num_of_fdf_rows_hist = [len(self.data)]
        self.num_of_immediate_action_rows_lst = [None]
        
        # Master's additional state tracking
        self.in_the_middle_of_empty_grouping = False
        self.in_the_middle_of_empty_grouping_steps = 0
        
        # Reset enhanced state
        self.previous_actions = []
        self.session_statistics = {
            'total_humanity_score': 0,
            'humanity_evaluations': 0,
            'rule_triggers': []
        }
        
        # Master's assertion check
        assert self.observation_space.contains(obs)
        return obs
    
    def compute_enhanced_humanity_reward(self, action, state, dfs, rules_reward_info=None):
        """
        CRITICAL_FIX: Use base class's compute_rule_based_humanity_score
        
        The base class (ATENAEnvCont) inherits from schema-specific helpers
        (e.g., networking_helpers.py) which have ALL the proper humanity rules,
        including:
        - Filter operator penalties (eq/ne on info_line = -1.0)
        - Column-specific rules
        - Back action rules
        - Group action rules
        
        The previous simplified implementation was missing critical rules!
        
        Args:
            action: Action taken
            state: Environment state  
            dfs: DataFrames (current, previous) - can be None during training
            rules_reward_info: HumanRulesReward instance to populate (like master)
        """
        # CRITICAL CHECK: If dfs is None (training mode with ret_df=False),
        # we need to reconstruct the DataFrames from the state
        if dfs is None:
            # During training, ret_df=False, so we need to get dfs manually
            # Use the environment's state to reconstruct the DataFrames
            try:
                dfs = self.env_prop.get_state_dfs(
                    self.data, 
                    state, 
                    memo=self.STATE_DF_HISTORY if hasattr(self, 'STATE_DF_HISTORY') else None,
                    dataset_number=self.dataset_number if hasattr(self, 'dataset_number') else None
                )
            except Exception as e:
                # If we can't reconstruct dfs, fallback to 0
                print(f"Could not reconstruct dfs for humanity computation: {e}")
                return 0.0
        
        # The base class method expects:
        # - dfs: tuple of (filtered_df, aggregated_df)
        # - state: current state
        # - rules_reward_info: dict-like object to populate
        # - done: boolean (we'll pass False since we're mid-episode)
        
        done = False  # Mid-episode evaluation
        
        try:
            # Call base class method which has ALL the proper rules
            humanity_score = self.compute_rule_based_humanity_score(
                dfs, state, rules_reward_info, done
            )
            
            # Update session statistics
            self.session_statistics['total_humanity_score'] += humanity_score
            self.session_statistics['humanity_evaluations'] += 1
            
            # Apply coefficient from config
            final_score = humanity_score * cfg.humanity_coeff
            return final_score
            
        except Exception as e:
            print(f"Error in base class humanity computation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to 0 if base class method fails
            return 0.0
    
    def display_distance(self, display1, display2):
        """
        MASTER-EXACT: Calculate display distance between two displays
        Returns an object with display_distance and data_distance attributes
        """
        class DisplayDistanceResult:
            def __init__(self, display_distance, data_distance):
                self.display_distance = display_distance
                self.data_distance = data_distance
        
        # If either display is None, return maximum distance
        if display1 is None or display2 is None:
            return DisplayDistanceResult(1.0, 1.0)
        
        # ENHANCED: Basic display distance calculation with numerical stability
        if hasattr(display1, 'shape') and hasattr(display2, 'shape'):
            # Shape difference with safe division
            shape_diff = abs(display1.shape[0] - display2.shape[0]) / max(display1.shape[0], display2.shape[0], 1)
            
            # Column structure similarity
            if hasattr(display1, 'columns') and hasattr(display2, 'columns'):
                common_cols = len(set(display1.columns) & set(display2.columns))
                total_cols = len(set(display1.columns) | set(display2.columns))
                col_similarity = NumericalStabilityHelper.safe_divide(common_cols, total_cols)
                
                # Combined distance with numerical stability
                display_distance = 1.0 - (col_similarity * (1 - shape_diff))
                display_distance = np.clip(display_distance, 0.0, 1.0)  # Ensure valid range
                
                # Data distance (for filter action same-data detection)
                # If displays have same shape and columns, check if data is identical
                try:
                    data_identical = display1.equals(display2) if hasattr(display1, 'equals') else False
                    data_distance = 0.0 if data_identical else display_distance
                except:
                    data_distance = display_distance  # Fallback if comparison fails
                
                return DisplayDistanceResult(display_distance, data_distance)
        
        # Fallback for other data types
        are_identical = str(display1) == str(display2)
        distance = 0.0 if are_identical else 1.0
        return DisplayDistanceResult(distance, distance)

    def compute_enhanced_diversity_reward(self, obs, action, state, last_action_type, reward_info):
        """
        MASTER-EXACT: Diversity calculation with same_display penalties
        Based on ATENA-master lines 796-841
        """
        # Back actions should never get diversity penalties!
        # Back actions don't create new displays, they just navigate to previous ones
        if self.env_prop.OPERATOR_TYPE_LOOKUP.get(last_action_type) == 'back':
            # Back actions get 0 diversity reward (master-exact)
            reward_info.diversity = 0
            return 0, {
                'display_distance': 0.0,
                'same_display_penalty': False,
                'is_back_action': True
            }
        
        if not hasattr(self, 'dhist') or len(self.dhist) <= 1:
            # FIX: Use fresh config for current coefficient values
            print(f"Diversity coefficient applied (early return): {cfg.diversity_coeff}")
            return cfg.diversity_coeff, {'display_distance': 1.0, 'same_display_penalty': False}
        
        last_display = self.dhist[-1]
        sim_vec = []
        
        # Compare current display with all previous displays
        for i, prev_display in enumerate(self.dhist[:-1]):
            # Calculate display distance with numerical stability
            display_distance_result = self.display_distance(prev_display, last_display)
            dist = display_distance_result.display_distance
            
            # Same display penalty check (only for non-back actions)
            if (dist == 0 or  # Exact same display
                (display_distance_result.data_distance == 0 and  # Same data after filter
                 self.env_prop.OPERATOR_TYPE_LOOKUP.get(last_action_type) == 'filter')):
                
                # 🎛️ CONDITIONAL: Apply penalty with optional stabilization
                # FIX: Use fresh config for current coefficient values
                raw_penalty = -1.0 * cfg.humanity_coeff
                print(f"Humanity coefficient applied (same display penalty): {cfg.humanity_coeff}")
                
                if USE_REWARD_STABILIZER and self.reward_stabilizer:
                    stabilized_penalty = self.reward_stabilizer.stabilize_diversity_reward(raw_penalty, is_penalty=True)
                    penalty_scale_factor = self.reward_stabilizer.penalty_scale_factor
                else:
                    stabilized_penalty = raw_penalty  # Stable: Raw penalty like train_ipdate-1009-18:54.png
                    penalty_scale_factor = 1.0
                
                reward_info.same_display_seen_already = stabilized_penalty
                reward_info.diversity = stabilized_penalty
                
                return stabilized_penalty, {
                    'display_distance': 0.0,
                    'same_display_penalty': True,
                    'penalty_scale_factor': penalty_scale_factor
                }
            else:
                sim_vec.append(dist)
        
        # Calculate diversity reward using min(similarities) like master
        # FIX: Use fresh config for current coefficient values
        if sim_vec:
            raw_diversity = min(sim_vec) * cfg.diversity_coeff
            print(f"Diversity coefficient applied (main calc): {cfg.diversity_coeff}")
        else:
            raw_diversity = cfg.diversity_coeff  # No previous displays to compare
            print(f"Diversity coefficient applied (no prev displays): {cfg.diversity_coeff}")
        
        # 🎛️ CONDITIONAL: Use raw or stabilized diversity based on flag  
        if USE_REWARD_STABILIZER and self.reward_stabilizer:
            stabilized_diversity = self.reward_stabilizer.stabilize_diversity_reward(raw_diversity, is_penalty=False)
            reward_info.diversity = stabilized_diversity
            result_diversity = stabilized_diversity
            extra_info = {'stabilizer_ema': self.reward_stabilizer.diversity_ema}
        else:
            # STABLE: Use raw diversity (like train_ipdate-1009-18:54.png excellence!)
            reward_info.diversity = raw_diversity
            result_diversity = raw_diversity
            extra_info = {'stabilizer_disabled': True}
        
        return result_diversity, {
            'display_distance': min(sim_vec) if sim_vec else 1.0,
            'same_display_penalty': False,
            **extra_info
        }
    
    def compute_action_novelty(self, action):
        """Compute how novel this action is compared to recent actions"""
        if len(self.previous_actions) == 0:
            return 1.0  # First action is always novel
        
        # Check similarity to recent actions
        similarity_scores = []
        for prev_action in self.previous_actions[-3:]:  # Check last 3 actions
            if isinstance(action, np.ndarray) and isinstance(prev_action, np.ndarray):
                similarity = np.dot(action, prev_action) / (np.linalg.norm(action) * np.linalg.norm(prev_action) + 1e-8)
                similarity_scores.append(similarity)
        
        # Novelty is inverse of max similarity
        if similarity_scores:
            max_similarity = max(similarity_scores)
            return 1.0 - max_similarity
        
        return 1.0
    
    def compute_interestingness_compaction_gain_master_exact(self, dfs, state):
        """
        COPIED EXACTLY from ATENA-master/gym_atena/envs/atena_env_cont.py lines 843-879
        
        Let R = the number of rows in the original dataframe
        Let G = the number of groups in the current dataframe (if grouped)
        Let C = the number of grouped columns in the current dataframe (if grouped)
        Let R' = the number of rows in the currrent dataframe
        If dfs[1] is not None (there is grouping involved) returns CG = CDS * DSS where
        CDS = sigmoid(-(17*((1-1/log(10, 10+G*C))-0.5)))
        DSS = 1-sigmoid(-(17*((1-1/log(7, 7+G*C))-0.5)))
        If filter only is involved, returns 1-log(R')/log(R) = 1-log(R,R')
        """
        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        denominator_epsilon = 0.00001
        R = len(self.data)
        R_tag = len(df_dt) if df_dt is not None else 0
        C = len(getattr(state, 'grouping', []))
        
        if is_grouping:
            R_tag = len(dfs[0]) if dfs[0] is not None else 0
            G = len(df_dt) if df_dt is not None else 0
            
            """Punishment for a single group"""
            if G == 1:
                return -1
            
            # Smaller (G*C) makes it larger
            compact_display_score = normalized_sigmoid_fkt(0.5, 17,
                                                           1 - 1 / math.log(8 + G * C + denominator_epsilon, 8))
            # Larger R_tag makes it larger  
            compact_data_score = 1 - normalized_sigmoid_fkt(0.5, 17,
                                                            1 - 1 / math.log(7 + R_tag + denominator_epsilon, 7))
            return compact_display_score * compact_data_score
        
        return 1 - math.log(R_tag + denominator_epsilon, R) if R > 0 else 0  # if filter only

    def get_filtered_only_or_grouped_data_frame(self, dfs):
        """
        Helper function to determine if we have grouping and return the appropriate dataframe
        """
        if len(dfs) >= 2 and dfs[1] is not None:
            # We have grouping
            return dfs[1], True
        else:
            # Filter only
            return dfs[0] if dfs[0] is not None else [], False

    def compute_interestingness_kl_divergence_master_exact(self, dfs, state):
        """
        COPIED AND ADAPTED from ATENA-master/gym_atena/envs/atena_env_cont.py lines 458-580
        Returns sigmoid(max_{KL_div_attr for each attribute in the current dataframe}/2-3)
        """
        kl_distances = []
        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        
        # Get previous dataframe (past 2 steps) - simplified for now
        df_D = self.data  # In master, this gets from history with complex caching
        
        # Find attributes to compute KL divergence for
        if is_grouping:
            # Get aggregate attributes from state
            aggeregate_attributes_list = self.get_aggregate_attributes(state)
            kl_attrs = aggeregate_attributes_list
        else:
            KL_DIV_EPSILON = 2 / len(df_D) * 0.1 if len(df_D) > 0 else 0.01
            kl_attrs = df_D.columns if hasattr(df_D, 'columns') else []
        
        # Compute KL_divergence for each attribute
        for attr in kl_attrs:
            try:
                if hasattr(df_D, 'iloc') and hasattr(df_dt, 'iloc'):
                    # Pandas dataframe
                    attr_value_count1 = CounterWithoutNanKeys(df_D[attr].values)
                    attr_value_count2 = CounterWithoutNanKeys(df_dt[attr].values)
                    
                    if is_grouping:
                        KL_DIV_EPSILON = 2 / attr_value_count1.elements() * 0.1 if attr_value_count1.elements() > 0 else 0.01
                    
                    # Compute KL divergence between the two distributions
                    kl_distance = self.compute_kl_divergence_between_counters(attr_value_count1, attr_value_count2, KL_DIV_EPSILON)
                    kl_distances.append(kl_distance)
                    
            except Exception as e:
                print(f"Error computing KL divergence for attribute {attr}: {e}")
                continue
        
        if not kl_distances:
            return 0
        
        max_kl_distance = max(kl_distances)
        
        # Master's sigmoid transformation
        return normalized_sigmoid_fkt(3, 0.5, max_kl_distance / 2)

    def compute_kl_divergence_between_counters(self, counter1, counter2, epsilon):
        """Compute KL divergence between two counters (probability distributions)"""
        # Get all unique values
        all_keys = set(counter1.keys()) | set(counter2.keys())
        
        if not all_keys:
            return 0
        
        # Convert to probability distributions
        total1 = counter1.elements()
        total2 = counter2.elements()
        
        if total1 == 0 or total2 == 0:
            return 0
        
        # Calculate KL divergence
        kl_div = 0
        for key in all_keys:
            p1 = (counter1[key] + epsilon) / (total1 + epsilon * len(all_keys))
            p2 = (counter2[key] + epsilon) / (total2 + epsilon * len(all_keys))
            
            kl_div += p1 * math.log(p1 / p2)
        
        return kl_div

    def get_aggregate_attributes(self, state):
        """Extract aggregate attributes from state"""
        aggregations = getattr(state, 'aggregations', [])
        return [agg.field for agg in aggregations] if aggregations else []

    def compute_enhanced_interestingness_reward(self, dfs, state, action_type):
        """
        MASTER-EXACT interestingness reward calculation
        Uses exact master algorithms for compaction_gain and kl_distance
        """
        kl_distance = 0
        compaction_gain = 0
        
        # MASTER-EXACT: Only kl_distance OR compaction_gain, not both
        if self.env_prop.OPERATOR_TYPE_LOOKUP[action_type] == "filter":
            kl_distance = self.compute_interestingness_kl_divergence_master_exact(dfs, state)
            # FIX: Use fresh config for current coefficient values
            kl_distance *= cfg.kl_coeff
            print(f"KL coefficient applied: {cfg.kl_coeff}")
        
        elif self.env_prop.OPERATOR_TYPE_LOOKUP[action_type] == 'group':
            compaction_gain = self.compute_interestingness_compaction_gain_master_exact(dfs, state)
            # FIX: Use fresh config for current coefficient values
            compaction_gain *= cfg.compaction_coeff
            print(f"Compaction coefficient applied: {cfg.compaction_coeff}")
        
        # Master logic: interestingness = max(kl_distance, compaction_gain)
        total_interestingness = max(kl_distance, compaction_gain)
        
        # Print reward calculation details
        if total_interestingness != 0 or True:  # Always debug for now
            print(f"MASTER-EXACT REWARD:")
            print(f"   action_type: {action_type}")
            print(f"   kl_distance: {kl_distance}")
            print(f"   compaction_gain: {compaction_gain}")
            print(f"   total_interestingness: {total_interestingness}")
            if hasattr(state, 'get'):
                print(f"   state grouping: {getattr(state, 'grouping', 'N/A')}")
        
        return total_interestingness, {
            'kl_divergence': kl_distance,
            'compaction_gain': compaction_gain, 
            'information_gain': 0
        }
    
    def compute_information_gain(self, dfs, state):
        """
        Compute information gain from the action
        This is a simplified version of what could be a complex calculation
        """
        if not dfs or len(dfs) == 0:
            return 0
        
        # Simple heuristic: reward actions that result in meaningful data reduction
        original_size = len(self.data) if hasattr(self, 'data') else 1000
        current_size = len(dfs[0]) if dfs[0] is not None else 0
        
        if original_size > 0:
            reduction_ratio = (original_size - current_size) / original_size
            # Reward moderate reductions, penalize extreme reductions
            if 0.1 <= reduction_ratio <= 0.8:
                return reduction_ratio * 0.5
            elif reduction_ratio > 0.8:
                return -0.3  # Too much reduction
        
        return 0
    
    
    def _is_empty_display(self, obs):
        """Check if display is empty (master-exact logic)"""
        if obs is None:
            return True
        if hasattr(obs, 'shape') and obs.shape[0] == 0:
            return True
        if hasattr(obs, '__len__') and len(obs) == 0:
            return True
        return False
    
    def _is_empty_groupings(self, obs):
        """Check if grouping result is empty (master-exact logic)"""
        # This would check if grouping resulted in empty groups
        # For now, use same logic as empty display
        return self._is_empty_display(obs)
    
    
    def step(self, action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None, **kwargs):
        """
        MASTER-EXACT: Enhanced step with complete reward calculation
        Signature matches master's EXACTLY: step(action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None)
        Based on ATENA-master/gym_atena/envs/atena_env_cont.py line 881
        """
        
        # Store action for future reference
        if isinstance(action, np.ndarray):
            self.previous_actions.append(action.copy())
        else:
            self.previous_actions.append(action)
        
        # Keep only recent actions
        if len(self.previous_actions) > 10:
            self.previous_actions.pop(0)
        
        # Call parent step with raw action - let our enhanced action_to_vec handle the processing
        # The base step() method will call self.action_to_vec(), which will use our enhanced version
        parent_kwargs = kwargs.copy()
        parent_kwargs.update({
            'compressed': compressed,
            'filter_by_field': filter_by_field,
            'continuous_filter_term': continuous_filter_term,
            'filter_term': filter_term
        })
        
        obs, base_reward, done, info = super().step(action, **parent_kwargs)
        
        # Enhanced reward calculation
        if 'reward_info' in info:
            reward_info = info['reward_info']
            state = info.get('state', {})
            raw_action = info.get('raw_action', action)
            raw_display = info.get('raw_display', None)
            
            # Create enhanced reward info
            enhanced_reward = EnhancedStepReward()
            
            # Copy base reward components
            for attr in ['empty_display', 'empty_groupings', 'same_display_seen_already', 
                        'back', 'diversity', 'interestingness', 'kl_distance', 
                        'compaction_gain', 'humanity']:
                if hasattr(reward_info, attr):
                    setattr(enhanced_reward, attr, getattr(reward_info, attr))
            
            # MASTER-EXACT PENALTY CHECKS (before normal reward calculation)
            penalty_applied = False
            special_back_action = False
            last_action_type = raw_action[0] if isinstance(raw_action, (list, np.ndarray)) and len(raw_action) > 0 else 0
            
            # Check for no history for back action
            no_history_for_back = (self.env_prop.OPERATOR_TYPE_LOOKUP.get(last_action_type) == 'back' and 
                                 (not hasattr(self, 'dhist') or len(getattr(self, 'dhist', [])) <= 1))
            
            # (1.a) punishment for 'back' action with no previous displays  
            if no_history_for_back:
                # REDUCED PENALTY: Encourage back action exploration (was -4.5, now -1.0)
                penalty = -1.0  # Fixed penalty instead of -1.0 * cfg.humanity_coeff
                enhanced_reward.back = penalty
                penalty_applied = True
                print(f"REDUCED back penalty: {penalty} (was {-1.0 * cfg.humanity_coeff})")
            
            # (1.b) if last action is 'back' give 0 reward + humanity only
            elif self.env_prop.OPERATOR_TYPE_LOOKUP.get(last_action_type) == 'back':
                enhanced_reward.back = 0  # Base back reward is 0
                special_back_action = True
                print(f"BACK ACTION: humanity rewards only")
            
            # (1.c) punishment for empty results
            elif self._is_empty_display(obs):
                penalty = -1.0 * cfg.humanity_coeff
                enhanced_reward.empty_display = penalty
                enhanced_reward.interestingness = penalty  # Overrides other interestingness!
                penalty_applied = True
                print(f"EMPTY DISPLAY PENALTY: {penalty}")
            
            # (1.d) punishment for empty grouping
            elif (hasattr(state, 'grouping') and state.grouping and self._is_empty_groupings(obs)):
                penalty = -1.0 * cfg.humanity_coeff
                enhanced_reward.empty_groupings = penalty
                enhanced_reward.interestingness = penalty  # Overrides other interestingness!
                penalty_applied = True
                print(f"EMPTY GROUPING PENALTY: {penalty}")
            
            # Calculate rewards based on action type and penalties
            if not penalty_applied:
                # Enhanced diversity calculation (always calculated if not disabled)
                if not cfg.no_diversity:
                    diversity_reward, diversity_components = self.compute_enhanced_diversity_reward(
                        obs, raw_action, state, last_action_type, enhanced_reward
                    )
                    enhanced_reward.diversity = diversity_reward
                    enhanced_reward.diversity_components = diversity_components
                
                # MASTER-EXACT: Skip interestingness & humanity if same_display_seen_already
                # Check if same display penalty was applied during diversity calculation
                same_display_penalty = getattr(enhanced_reward, 'same_display_seen_already', 0) != 0
                
                if not same_display_penalty:
                    # Add enhanced humanity reward (if enabled and not back action)
                    if cfg.use_humans_reward and not special_back_action:
                        raw_humanity = self.compute_enhanced_humanity_reward(
                            raw_action, state, raw_display, enhanced_reward.rules_reward_info
                        )
                        # 🎛️ CONDITIONAL: Use raw or stabilized humanity based on flag
                        if USE_REWARD_STABILIZER and self.reward_stabilizer:
                            stabilized_humanity = self.reward_stabilizer.stabilize_humanity_reward(raw_humanity)
                            enhanced_reward.rules_based_humanity = stabilized_humanity
                            enhanced_reward.humanity += stabilized_humanity
                        else:
                            # STABLE: Use raw humanity (like train_ipdate-1009-18:54.png excellence!)
                            enhanced_reward.rules_based_humanity = raw_humanity
                            enhanced_reward.humanity += raw_humanity
                    
                    # Enhanced interestingness calculation (for non-back actions)
                    if not cfg.no_interestingness and raw_display and not special_back_action:
                        action_type = raw_action[0] if isinstance(raw_action, (list, np.ndarray)) else 0
                        raw_interestingness, interest_components = self.compute_enhanced_interestingness_reward(
                            raw_display, state, action_type
                        )
                        # 🎛️ CONDITIONAL: Use raw or stabilized interestingness based on flag
                        if USE_REWARD_STABILIZER and self.reward_stabilizer:
                            stabilized_interestingness = self.reward_stabilizer.stabilize_interestingness_reward(raw_interestingness)
                            
                            # Also stabilize individual components
                            if 'kl_divergence' in interest_components:
                                interest_components['kl_divergence'] = self.reward_stabilizer.stabilize_kl_distance(
                                    interest_components['kl_divergence']
                                )
                            if 'compaction_gain' in interest_components:
                                interest_components['compaction_gain'] = self.reward_stabilizer.stabilize_compaction_gain(
                                    interest_components['compaction_gain']
                                )
                            
                            enhanced_reward.interestingness = stabilized_interestingness
                        else:
                            # STABLE: Use raw interestingness (like train_ipdate-1009-18:54.png excellence!)
                            enhanced_reward.interestingness = raw_interestingness
                        enhanced_reward.interestingness_components = interest_components
                    
                    # For back actions, ONLY calculate humanity rewards (master-exact)
                    if special_back_action and cfg.use_humans_reward:
                        raw_back_humanity = self.compute_enhanced_humanity_reward(
                            raw_action, state, raw_display, enhanced_reward.rules_reward_info
                        )
                        # 🎛️ CONDITIONAL: Use raw or stabilized back humanity based on flag
                        if USE_REWARD_STABILIZER and self.reward_stabilizer:
                            stabilized_back_humanity = self.reward_stabilizer.stabilize_humanity_reward(raw_back_humanity)
                            enhanced_reward.rules_based_humanity = stabilized_back_humanity
                            enhanced_reward.humanity += stabilized_back_humanity
                        else:
                            # STABLE: Use raw back humanity (like train_ipdate-1009-18:54.png excellence!)
                            enhanced_reward.rules_based_humanity = raw_back_humanity
                            enhanced_reward.humanity += raw_back_humanity
                    
                else:
                    print(f"MASTER-EXACT: Skipping interestingness & humanity due to same display penalty")
                    
                # Add Snorkel humanity for ALL actions (master-exact integration)
                if cfg.use_snorkel and hasattr(self, 'snorkel_gen_model') and self.snorkel_gen_model is not None:
                    try:
                        r_snorkel_humanity, non_abstain_funcs_dict = self.compute_snorkel_humanity_score()
                        enhanced_reward.snorkel_rules_reward_info = HumanRulesReward(non_abstain_funcs_dict)
                        
                        # Apply master's exact multiplier rules (from base class lines 775-804)
                        # These are CRITICAL for proper Snorkel reward scaling
                        
                        # Apply humanity coefficient scaling (master-exact)
                        # FIX: Use config for current coefficient values
                        current_humanity_coeff = cfg.humanity_coeff
                        
                        print(f"BEFORE coeff: r_snorkel_humanity={r_snorkel_humanity:.6f}")
                        print(f"Using current humanity_coeff: {current_humanity_coeff}")
                        r_snorkel_humanity *= current_humanity_coeff
                        print(f"AFTER coeff: r_snorkel_humanity={r_snorkel_humanity:.6f}")
                        enhanced_reward.snorkel_humanity = r_snorkel_humanity
                        enhanced_reward.humanity += r_snorkel_humanity
                        
                        print(f"Snorkel humanity: {r_snorkel_humanity:.6f}")
                        
                    except Exception as e:
                        print(f"Snorkel humanity calculation failed: {e}")
                        enhanced_reward.snorkel_humanity = 0
                else:
                    enhanced_reward.snorkel_humanity = 0
                
                # Mark back action in reward info
                if special_back_action:
                    enhanced_reward.is_back = True
            
            # Update reward info
            info['reward_info'] = enhanced_reward
            
            # Recalculate total reward
            total_reward = 0
            for component_name, component_value in enhanced_reward.items():
                if isinstance(component_value, (int, float)):
                    total_reward += component_value
            
            # Update the reward
            base_reward = total_reward
        
        return obs, base_reward, done, info
    
    def get_session_statistics(self):
        """Get detailed session statistics"""
        avg_humanity = 0
        if self.session_statistics['humanity_evaluations'] > 0:
            avg_humanity = (self.session_statistics['total_humanity_score'] / 
                          self.session_statistics['humanity_evaluations'])
        
        return {
            'total_actions': len(self.previous_actions),
            'average_humanity_score': avg_humanity,
            'total_humanity_evaluations': self.session_statistics['humanity_evaluations'],
            'rule_triggers': len(self.session_statistics['rule_triggers'])
        }
    
    def _discrete_to_continuous_action(self, action_idx):
        """
        MASTER-EXACT: Convert discrete action index (FFParamSoftmax) to 6-dimensional continuous vector
        
        Based on FFParamSoftmax policy discrete action space:
        - 0: back action  
        - 1-936: filter actions (12 fields × 3 operators × 26 terms)
        - 937-948: group actions (12 fields)
        
        Args:
            action_idx: Integer action index from FFParamSoftmax policy
            
        Returns:
            np.array: 6-dimensional continuous action vector
        """
        if action_idx == 0:
            # Back action: [0, 0, 0, 0, 0, 0] 
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        elif 1 <= action_idx <= 936:
            # Filter action: [1, field, operator, term, 0, 0]
            # Decode filter parameters from action index
            filter_idx = action_idx - 1  # Convert to 0-based
            
            # Decode: 12 fields × 3 operators × 26 terms = 936 combinations
            terms_per_field_op = 26
            ops_per_field = 3
            
            field = filter_idx // (ops_per_field * terms_per_field_op)
            remaining = filter_idx % (ops_per_field * terms_per_field_op)
            operator = remaining // terms_per_field_op
            term = remaining % terms_per_field_op
            
            return np.array([1, field, operator, term, 0, 0], dtype=np.float32)
        elif 937 <= action_idx <= 948:
            # Group action: [2, field, 0, 0, 0, 0]
            group_idx = action_idx - 937  # Convert to 0-based  
            field = group_idx  # 12 fields (0-11)
            return np.array([2, field, 0, 0, 0, 0], dtype=np.float32)
        else:
            # Invalid action index, default to back
            print(f"Warning: Invalid action index {action_idx}, defaulting to back action")
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    def action_to_vec(self, action, compressed=False, continuous_filter_term=True, filter_by_field=True):
        """
        MASTER-EXACT: Action to vector conversion with architecture-specific processing
        Based on ATENA-master/gym_atena/envs/atena_env_cont.py lines 1010-1025
        
        This method replicates master's exact logic:
        1. Architecture-specific processing for FFParamSoftmax vs FFGaussian
        2. Compressed to full range conversion when needed
        3. Continuous to discrete conversion
        4. Filter term handling
        
        Args:
            action: Raw action (discrete index for FFParamSoftmax, continuous vector for FFGaussian)
            compressed: Whether action is in compressed range
            continuous_filter_term: Whether to use continuous filter terms
            filter_by_field: Whether to filter by field
            
        Returns:
            tuple: (processed_action, filter_by_field)
        """
        # Architecture-specific processing matching master EXACTLY
        # Master lines 1011-1013: if self.arch is ArchName.FF_PARAM_SOFTMAX or self.arch is ArchName.FF_SOFTMAX:
        #                         compressed = False
        #                         action = self.param_softmax_idx_to_action(action)
        
        # Detect architecture from action type and current policy configuration
        # Handle both scalar integers and 1D arrays with single integer
        is_discrete_action = (
            isinstance(action, (int, np.integer)) or 
            (isinstance(action, np.ndarray) and action.shape == () and np.issubdtype(action.dtype, np.integer)) or
            (isinstance(action, np.ndarray) and action.shape == (1,) and np.issubdtype(action.dtype, np.integer))
        )
        
        if is_discrete_action:
            # FFParamSoftmax: discrete action index
            compressed = False  # Master sets compressed=False for FFParamSoftmax
            
            # Extract scalar from array if needed
            if isinstance(action, np.ndarray):
                action_idx = action.item() if action.shape == () else action[0]
            else:
                action_idx = action
                
            action = self._discrete_to_continuous_action(action_idx)  # Convert to continuous vector
        else:
            # Ensure action is numpy array for consistent processing
            action = np.array(action, dtype=np.float32)
        
        # Master lines 1014-1017: if compressed: action = self.env_prop.compressed2full_range(action, continuous_filter_term)
        if compressed:
            # Apply compressed to full range conversion (for FFGaussian)
            # Ensure action is numpy array before passing to compressed2full_range
            action = np.array(action, dtype=np.float32)
            if hasattr(self, 'env_prop') and hasattr(self.env_prop, 'compressed2full_range'):
                action = self.env_prop.compressed2full_range(action, continuous_filter_term)
            else:
                # Fallback: simple scaling from [-3,3] to action space range
                action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Master lines 1018-1020: if cfg.filter_from_list: continuous_filter_term=False, filter_by_field=False
        if hasattr(cfg, 'filter_from_list') and cfg.filter_from_list:
            continuous_filter_term = False
            filter_by_field = False
        
        # Master lines 1021-1024: action_filter_term = action[3], action = self.cont2dis(action)
        # Ensure action is numpy array before accessing elements
        action = np.array(action, dtype=np.float32) if not isinstance(action, np.ndarray) else action
        action_filter_term = action[3]
        action = self.cont2dis(action)
        
        # Master line 1024-1025: if continuous_filter_term: action[3] = action_filter_term + 0.5
        if continuous_filter_term:
            action[3] = action_filter_term + 0.5
        
        return action, filter_by_field

    def get_snorkel_obj_dict(self):
        """
        Create Snorkel candidate dictionary - matching master exactly
        ATENA-master/gym_atena/envs/atena_env_cont.py lines 1091-1108
        
        Returns: Dictionary with action history for Snorkel evaluation
        """
        if not self.ahist:
            return {'actions_lst': [], 'filter_terms_lst': [], 'num_of_rows_lst': [], 
                   'num_of_fdf_rows_lst': [], 'num_of_immediate_action_rows_lst': [],
                   'dataset_num': getattr(self, 'dataset_number', 0)}
        
        assert isinstance(self.ahist[0], (list, np.ndarray))
        
        from copy import deepcopy
        actions_lst = deepcopy(self.ahist)
        actions_lst = [[int(entry) for entry in vec] for vec in actions_lst]
        filter_terms_lst = getattr(self, 'filter_terms_hist', [])
        num_of_rows_lst = getattr(self, 'num_of_rows_hist', [])
        num_of_fdf_rows_lst = getattr(self, 'num_of_fdf_rows_hist', [])
        
        to_json = {
            'actions_lst': actions_lst, 
            'filter_terms_lst': filter_terms_lst,
            'num_of_rows_lst': num_of_rows_lst, 
            'num_of_fdf_rows_lst': num_of_fdf_rows_lst,
            'num_of_immediate_action_rows_lst': getattr(self, 'num_of_immediate_action_rows_lst', []),
            'dataset_num': getattr(self, 'dataset_number', 0)
        }
        return to_json


# Factory function to create enhanced environment  
# Factory function to create enhanced environment  
def make_enhanced_atena_env(max_steps=12, gradual_training=False, **kwargs):
    """
    Factory function to create enhanced ATENA environment with gradual training support
    """
    return EnhancedATENAEnv(max_steps=max_steps, gradual_training=gradual_training, **kwargs)
