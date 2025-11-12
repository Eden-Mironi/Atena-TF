"""
CRITICAL MISSING COMPONENT: Periodic Humanity Evaluation System
Based on ATENA-master/humanity_reward.py

Evaluates agent's ability to act like humans by comparing actions to recorded human behavior patterns
"""

import random
from collections import defaultdict
import pickle
from copy import deepcopy
import numpy as np
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'gym_atena/lib'))
sys.path.append(os.path.join(current_dir, 'Utilities/Envs'))

try:
    import gym_atena.lib.helpers as ATENAUtils
except ImportError:
    # Fallback for import issues
    class ATENAUtils:
        OPERATOR_TYPE_LOOKUP = {0: 'back', 1: 'filter', 2: 'group'}

def decompress_and_discretize_actions(agent_actions):
    """
    Convert continuous agent actions to discrete format for comparison with human actions
    Based on master's Utilities/Envs/Envs_Utilities.py
    """
    discrete_actions = []
    
    for action in agent_actions:
        if isinstance(action, (list, np.ndarray)):
            # Convert to discrete action format [action_type, column, filter_op, filter_value, agg_column, agg_func]
            discrete_action = [int(round(val)) if i < 3 else val for i, val in enumerate(action)]
            # Ensure proper bounds for action type (0, 1, 2 for back, filter, group)
            discrete_action[0] = max(0, min(2, discrete_action[0]))
            discrete_actions.append(discrete_action)
        else:
            discrete_actions.append([int(round(action)), 0, 0, 0, 0, 0])
    
    return discrete_actions

def eval_agent_humanity(human_displays_actions_clusters, human_obss, agent_actions, pad_length):
    """
    CRITICAL FUNCTION: Compare agent actions to human behavior patterns
    
    Returns the rate of agent actions that are also performed by humans
    w.r.t. the same observation and also return statistics info

    Based on ATENA-master/humanity_reward.py lines 14-53
    
    :param human_displays_actions_clusters: Dictionary with humans obs-act_lst pairs
    :param human_obss: List of observations to evaluate the agent on
    :param agent_actions: List of actions of the agent corresponding to observations
    :param pad_length: Length of padding to the agent_actions list
    :return: (agent_humanity_rate, detailed_info_dict)
    """
    info = {
        'success_obs': [],
        'failure_obs': [],
        'success_count_per_action_type': defaultdict(int),
        'failure_count_per_action_type': defaultdict(int)
    }

    # Remove padded observations and actions
    agent_actions = agent_actions[:len(agent_actions) - pad_length] if pad_length > 0 else agent_actions

    # Decompress and discretize agent actions (convert continuous to discrete)
    agent_discrete_actions = decompress_and_discretize_actions(agent_actions)

    # Compare agent to humans
    agent_success_count = 0
    
    for obs, agent_action in zip(human_obss, agent_discrete_actions):
        # Check if agent acts like a human on the current observation
        agent_action_type, is_obs_success, obs_success_score = does_agent_acts_on_obs_like_human(
            agent_action, human_displays_actions_clusters, obs)

        if is_obs_success:  # If agent took action that was taken by human for the current obs
            agent_success_count += 1
            info['success_obs'].append(obs)
            info['success_count_per_action_type'][ATENAUtils.OPERATOR_TYPE_LOOKUP[agent_action_type]] += 1
        else:
            info['failure_obs'].append(obs)
            info['failure_count_per_action_type'][ATENAUtils.OPERATOR_TYPE_LOOKUP[agent_action_type]] += 1

    # Calculate humanity rate
    humanity_rate = agent_success_count / len(human_obss) if len(human_obss) > 0 else 0.0
    
    return humanity_rate, info

def does_agent_acts_on_obs_like_human(agent_action, human_displays_actions_clusters, obs):
    """
    CRITICAL COMPARISON LOGIC: Check if agent's action matches human behavior
    
    Based on ATENA-master/humanity_reward.py lines 56-104
    
    Returns agent_action_type (0,1,2), is_obs_success (True if agent makes human action),
    success_score (customized success score for acting like human)
    
    :param agent_action: Agent's discrete action vector
    :param human_displays_actions_clusters: Dictionary of human behavior patterns
    :param obs: Current observation vector
    :return: (agent_action_type, is_obs_success, success_score)
    """
    is_obs_success = False
    obs = tuple(obs)  # Convert to tuple for dictionary lookup
    
    # Get human actions for this observation
    human_actions_for_obs = human_displays_actions_clusters.get(obs, [])
    
    if not human_actions_for_obs:
        # No human data for this observation
        return agent_action[0], False, 0.0
    
    agent_action_type = agent_action[0]
    success_score = 0
    
    for human_action in human_actions_for_obs:
        human_action_type = human_action[0]
        human_action_col_id = human_action[1] if len(human_action) > 1 else 0
        agent_action_col_id = agent_action[1] if len(agent_action) > 1 else 0
        human_action_type_str = ATENAUtils.OPERATOR_TYPE_LOOKUP.get(human_action_type, 'unknown')

        if human_action_type_str == "back":
            if human_action_type == agent_action_type:
                success_score = 1.0
                is_obs_success = True
                break

        elif human_action_type_str == "filter":
            if human_action_type == agent_action_type:
                success_score = 0.4
                # Check if same column
                if human_action_col_id == agent_action_col_id:
                    success_score += 0.9
                    is_obs_success = True
                    break

        elif human_action_type_str == "group":
            if human_action_type == agent_action_type:
                success_score = 0.0
                if human_action_col_id == agent_action_col_id:
                    success_score += 1.0
                    is_obs_success = True
                    break

        else:
            print(f"Warning: Unknown action type {human_action_type_str}")

    return agent_action_type, is_obs_success, success_score

def load_human_behavior_data():
    """
    Load human behavior patterns from pickle file
    Based on master's Utilities/Utility_Functions.py load_human_session_actions_clusters()
    """
    try:
        with open('human_actions_clusters.pickle', 'rb') as handle:
            human_displays_actions_clusters = pickle.load(handle)
        
        # Process the data (remove easy actions, filter back-only observations)
        result = {}
        for obs, action_lst in human_displays_actions_clusters.items():
            if not action_lst:
                continue
                
            # Remove `back` actions if there are other actions
            is_back_in_act_lst = [act[0] == 0 for act in action_lst]
            if not all(is_back_in_act_lst):
                action_lst = [act for i, act in enumerate(action_lst) if not is_back_in_act_lst[i]]
            
            # Skip observations that have only `back` actions
            if not action_lst or all(is_back_in_act_lst):
                continue
                
            result[obs] = action_lst
        
        print(f"Loaded human behavior data: {len(result)} observation patterns")
        return result
        
    except FileNotFoundError:
        print("Warning: human_actions_clusters.pickle not found. Humanity evaluation will be disabled.")
        return {}
    except Exception as e:
        print(f"Error loading human behavior data: {e}")
        return {}
