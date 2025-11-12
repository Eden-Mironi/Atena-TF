"""
CRITICAL: Master's Human Rules Statistics Tracking System
Based on ATENA-master train_agent_chainerrl.py lines 412-415

Implements master's exact human rules tracking:
- human_rules_stats = {rule: (average_reward, occurrence_count) for rule in HumanRule}
- Schema-specific rule sets (NetHumanRule, FlightsHumanRule, etc.)
- Rule effectiveness analysis and logging
"""

import sys
import numpy as np
from enum import Enum
from collections import defaultdict
from typing import Dict, Tuple, Any

sys.path.append('./Configuration')
import config as cfg

# Import schema-specific human rules based on configuration
class NetHumanRule(Enum):
    """
    MASTER-EXACT: Networking schema human rules
    Based on ATENA-master/gym_atena/lib/networking_helpers.py lines 489-528
    """
    humane_columns_group = 1
    neutral_columns_group = 2
    inhumane_columns_group = 3
    column_already_grouped = 4
    group_num_of_groups_unchanged = 5
    group_num_of_groups_changed = 6
    stacking_five_groups = 7
    stacking_more_than_six_groups = 8
    filter_term_appears_in_human_session = 9
    filter_term_doesnt_appear_in_human_session = 10
    info_line_bad_filter_operators = 11
    info_line_good_filter_operators = 12
    humane_columns_filter = 13
    neutral_columns_filter = 14
    inhumane_columns_filter = 15
    filter_num_of_rows_unchanged = 16
    filter_num_of_rows_changed = 17
    back_with_no_history = 18
    back_after_single_non_back = 19
    back_as_a_function_of_stacked_non_backs = 20
    back_after_back = 21
    group_as_first_action = 22
    filter_as_first_action = 23
    stacking_more_than_two_filters = 24
    filter_from_undisplayed_column = 25
    filter_from_displayed_column = 26
    back_after_filter_readability_gain = 27
    group_results_in_single_group = 28
    filter_small_number_of_rows = 29
    filter_results_in_single_row = 30
    filter_on_the_same_column_in_subsession = 31
    end_episode_filter_readability_gain = 32
    using_not_equal_filter_operator = 33
    group_on_filtered_column_in_subsession = 34
    group_ip_dst_after_group_ip_src = 35
    highest_layer_filter_before_group = 36
    highest_layer_group_on_filtered_display = 37
    first_group_length = 38
    group_eth_src_on_filtered_display = 39

class FlightsHumanRule(Enum):
    """
    MASTER-EXACT: Flights schema human rules  
    Based on ATENA-master schema-specific rules
    """
    # Same numbering as networking for compatibility
    humane_columns_group = 1
    neutral_columns_group = 2
    inhumane_columns_group = 3
    # ... (would contain flights-specific rules)

# Schema mapping
SCHEMA_HUMAN_RULES = {
    'NETWORKING': NetHumanRule,
    'FLIGHTS': FlightsHumanRule,
    'BIG_FLIGHTS': FlightsHumanRule,  # Fallback to FlightsHumanRule for now
    'WIDE_FLIGHTS': FlightsHumanRule, # Fallback to FlightsHumanRule for now
    'WIDE12_FLIGHTS': FlightsHumanRule, # Fallback to FlightsHumanRule for now
}

class HumanRulesTracker:
    """
    MASTER-EXACT: Human Rules Statistics Tracker
    
    Based on master's system:
    - train_agent_chainerrl.py lines 412-415: human_rules_stats tracking
    - Tracks (average_reward, occurrence_count) for each human rule
    - Schema-specific rule sets
    - Rule effectiveness analysis
    """
    
    def __init__(self, schema_name: str = None):
        """
        Initialize human rules tracker for specific schema
        
        Args:
            schema_name: Schema name (NETWORKING, FLIGHTS, etc.)
        """
        self.schema_name = schema_name or cfg.schema
        
        # Get schema-specific human rules
        self.HumanRule = SCHEMA_HUMAN_RULES.get(self.schema_name, NetHumanRule)
        
        # CRITICAL MASTER TRACKING: human_rules_stats = {rule: (avg_reward, count) for rule in HumanRule}
        # Based on train_agent_chainerrl.py line 415
        self.human_rules_stats: Dict[Enum, Tuple[float, int]] = {
            rule: (0.0, 0) for rule in self.HumanRule
        }
        
        # Additional tracking for analysis
        self.rule_rewards_history = defaultdict(list)  # Track individual rewards per rule
        self.episodes_tracked = 0
        
        print(f"Human Rules Tracker initialized:")
        print(f"   Schema: {self.schema_name}")
        print(f"   Rules available: {len(self.human_rules_stats)}")
        print(f"   Rule type: {self.HumanRule.__name__}")
    
    def track_rule(self, rule: Enum, reward_value: float):
        """
        CRITICAL: Track when a human rule is triggered
        
        Updates the (average_reward, occurrence_count) tuple for the rule
        
        Args:
            rule: Human rule that was triggered
            reward_value: Reward value contributed by this rule
        """
        if rule not in self.human_rules_stats:
            print(f"Warning: Unknown rule {rule}, adding to tracker")
            self.human_rules_stats[rule] = (0.0, 0)
        
        # Get current stats
        current_avg, current_count = self.human_rules_stats[rule]
        
        # Update count
        new_count = current_count + 1
        
        # Update running average (master's incremental average formula)
        # new_avg = (old_avg * old_count + new_value) / new_count
        new_avg = (current_avg * current_count + reward_value) / new_count
        
        # Update stats
        self.human_rules_stats[rule] = (new_avg, new_count)
        
        # Track individual rewards for analysis
        self.rule_rewards_history[rule].append(reward_value)
        
        return new_avg, new_count
    
    def track_rules_from_reward_info(self, rules_reward_info):
        """
        Track multiple rules from a reward info object
        
        Args:
            rules_reward_info: Object containing triggered rules and their rewards
        """
        if hasattr(rules_reward_info, 'items'):
            # Handle reward info object with items() method
            for rule, reward_value in rules_reward_info.items():
                if isinstance(rule, Enum) and isinstance(reward_value, (int, float)):
                    self.track_rule(rule, float(reward_value))
        elif hasattr(rules_reward_info, 'triggered_rules'):
            # Handle HumanRulesReward object 
            for rule, reward_value in rules_reward_info.triggered_rules.items():
                if isinstance(rule, Enum) and isinstance(reward_value, (int, float)):
                    self.track_rule(rule, float(reward_value))
        elif isinstance(rules_reward_info, dict):
            # Handle dictionary directly
            for rule, reward_value in rules_reward_info.items():
                if isinstance(rule, Enum) and isinstance(reward_value, (int, float)):
                    self.track_rule(rule, float(reward_value))
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rule statistics"""
        total_triggers = sum(count for _, count in self.human_rules_stats.values())
        
        # Sort rules by effectiveness (average reward * frequency)
        rule_effectiveness = {}
        for rule, (avg_reward, count) in self.human_rules_stats.items():
            if count > 0:
                effectiveness = avg_reward * count  # Total contribution
                rule_effectiveness[rule.name] = {
                    'average_reward': avg_reward,
                    'occurrence_count': count,
                    'total_contribution': effectiveness,
                    'trigger_frequency': count / total_triggers if total_triggers > 0 else 0
                }
        
        # Sort by total contribution
        sorted_rules = sorted(rule_effectiveness.items(), 
                            key=lambda x: abs(x[1]['total_contribution']), 
                            reverse=True)
        
        return {
            'total_rule_triggers': total_triggers,
            'active_rules': len([r for r, (_, count) in self.human_rules_stats.items() if count > 0]),
            'total_rules': len(self.human_rules_stats),
            'rule_effectiveness': dict(sorted_rules),
            'schema': self.schema_name
        }
    
    def log_rule_statistics(self, prefix="", top_n=10):
        """Log rule statistics"""
        stats = self.get_rule_statistics()
        
        print(f"{prefix}HUMAN RULES STATISTICS ({self.schema_name}):")
        print(f"{prefix}   Total triggers: {stats['total_rule_triggers']}")
        print(f"{prefix}   Active rules: {stats['active_rules']}/{stats['total_rules']}")
        
        if stats['rule_effectiveness']:
            print(f"{prefix}   Top {top_n} most effective rules:")
            for i, (rule_name, rule_stats) in enumerate(list(stats['rule_effectiveness'].items())[:top_n]):
                avg_reward = rule_stats['average_reward']
                count = rule_stats['occurrence_count']
                contribution = rule_stats['total_contribution']
                frequency = rule_stats['trigger_frequency']
                
                print(f"{prefix}     {i+1}. {rule_name}:")
                print(f"{prefix}        Avg reward: {avg_reward:.3f}, Count: {count}, Contribution: {contribution:.3f} ({frequency:.1%})")
        
        return stats
    
    def reset_statistics(self):
        """Reset all rule statistics"""
        self.human_rules_stats = {rule: (0.0, 0) for rule in self.HumanRule}
        self.rule_rewards_history.clear()
        self.episodes_tracked = 0
        print(f"Human rules statistics reset for {self.schema_name} schema")
    
    def save_statistics(self, filepath: str):
        """Save rule statistics to file (for analysis like master)"""
        import pickle
        
        # Convert enum keys to string keys for serialization (like master does)
        serializable_stats = {rule.name: stats for rule, stats in self.human_rules_stats.items()}
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Human rules statistics saved to {filepath}")
    
    def load_statistics(self, filepath: str):
        """Load rule statistics from file"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            print(f"Warning: Statistics file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                loaded_stats = pickle.load(f)
            
            # Convert string keys back to enum keys
            for rule_name, stats in loaded_stats.items():
                try:
                    rule = getattr(self.HumanRule, rule_name)
                    self.human_rules_stats[rule] = stats
                except AttributeError:
                    print(f"Warning: Unknown rule {rule_name} in loaded statistics")
            
            print(f"Human rules statistics loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load statistics: {e}")
            return False
