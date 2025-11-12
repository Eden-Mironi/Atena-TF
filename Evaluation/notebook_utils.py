"""
Notebook Utilities for ATENA-TF 2
Adapted from ATENA-master's NotebookUtils.py for TensorFlow 2 implementation
"""

from enum import Enum
import hashlib
import json
import os
import sys
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from uuid import uuid4

import gym
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.util import hash_pandas_object

import Configuration.config as cfg
from Evaluation.evaluation_measures_tf import draw_nx_display_tree
from arguments import SchemaName, ArchName
from gym_atena.envs.atena_env_cont import ATENAEnvCont
from gym_atena.envs.enhanced_atena_env import EnhancedATENAEnv
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.global_env_prop import update_global_env_prop_from_cfg
from gym_atena.lib.networking_helpers import convert_to_action_vector
from gym_atena.reactida.utils.utilities import Repository

from IPython.display import HTML, display

HumanStep = namedtuple('HumanStep', 'cur_obs action_vector next_obs')
HumanStepInfo = namedtuple('HumanStepInfo', 'cur_state action_vector action_info next_state dataset_number')

# environment name
env_d = 'ATENAcont-v0'


def get_prev_next_buttons_html_ob_for_display_num(disply_num, run_id):
    """Generate HTML for previous/next navigation buttons"""
    prev_and_next_buttons = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    a {{
      text-decoration: none;
      display: inline-block;
      padding: 8px 16px;
    }}
    a:hover {{
      background-color: #ddd;
      color: black;
    }}
    .previous {{
      background-color: #f1f1f1;
      color: black;
    }}
    .next {{
      background-color: #4CAF50;
      color: white;
    }}
    .round {{
      border-radius: 50%;
    }}
    </style>
    </head>
    <body>
    <a href="#disp{disply_num-1}_{run_id}" class="previous">&laquo; Previous</a>
    <a href="#disp{disply_num+1}_{run_id}" class="next">Next &raquo;</a>
    </body>
    </html> 
    """
    return HTML(prev_and_next_buttons)


def get_new_action_html_obj(display_num, run_id):
    """Generate HTML for new action header"""
    new_action = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    </style>
    </head>
    <body>
    <br style="line-height:5;">
    <hr>
    <h2><a name="disp{display_num}_{run_id}">Action No. {display_num}</a></h2>
    </body>
    </html> 
    """
    return HTML(new_action)


def get_dataset_number_html_obj(dataset_number, run_id):
    """Generate HTML for dataset number display"""
    dataset_number_html_txt = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    div.xlarge{{
        font-size:x-large;
        line-height: 1.2;
    }}
    div.large{{
        font-size:large;
        line-height: 1.2;
    }}
    u {{ 
      text-decoration: underline;
    }}
    </style>
    </head>
    <body>
    <a name="disp0_{run_id}"></a>
    <div class=xlarge>
    <p><strong>Running Actions For Dataset No. {dataset_number}</strong></p>
    </div>
    <div class=large>
    <p><strong>The following is the initial display of the dataset:</strong></p>
    </div>
    </body>
    </html> 
    """
    return HTML(dataset_number_html_txt)


def get_back_action_html_body():
    """Generate HTML body for back action"""
    return f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>BACK</strong></p>
    </div>
    """


def get_action_html_obj_helper(html_body):
    """Helper to wrap HTML body in proper structure"""
    html_txt = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    div.large{{
        font-size:medium;
        line-height: 1.2;
    }}
    u {{ 
      text-decoration: underline;
    }}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html> 
    """
    return HTML(html_txt)


def get_group_action_html_body(grouped_attr, agg_func, agg_attr):
    """Generate HTML body for group action"""
    return f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>GROUP</strong></p>
    <ul><li>Attr: <strong>{grouped_attr}</strong>  &emsp; Agg_func: <strong>{agg_func}</strong>  &emsp; Agg_attr: <strong>{agg_attr}</strong></li></ul>
    </div>
    """


def get_filter_action_html_body(filtered_attr, filter_op, filter_term):
    """Generate HTML body for filter action"""
    filter_op_tokens = filter_op.split()
    if len(filter_op_tokens) > 1:
        filter_op = f"""<font color="red">{filter_op_tokens[0]}</font> {''.join(filter_op_tokens[1:])}"""

    return f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>FILTER</strong></p>
    <ul><li>Attr: <strong>{filtered_attr}</strong>  &emsp; OP: <strong>{filter_op}</strong>  &emsp; Term: <strong>{filter_term}</strong></li></ul>
    </div>
    """


def get_filtering_state_html_obj(filtering_state):
    """Generate HTML object for filtering state"""
    filtering_lst = get_filtering_lst_from_state(filtering_state)
    if not filtering_lst:
        return None
    html_body = get_filtering_state_html_body(filtering_lst)
    return get_action_html_obj_helper(html_body)


def get_filtering_lst_from_state(filtering_state):
    """Extract filtering list from state"""
    filtering_lst = []
    for filtering_tuple in filtering_state:
        filtered_attr = update_global_env_prop_from_cfg().env_dataset_prop.KEYS_MAP_ANALYST_STR[filtering_tuple.field]
        filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_PRETTY_STR[filtering_tuple.condition]
        filter_term = filtering_tuple.term
        filtering_lst.append((filtered_attr, filter_op, filter_term))
    return filtering_lst


def get_filtering_state_html_body(filtering_lst):
    """Generate HTML body for filtering state"""
    if not filtering_lst:
        return ""

    for i, (filtered_attr, filter_op, filter_term) in enumerate(filtering_lst):
        filter_op_tokens = filter_op.split()
        if len(filter_op_tokens) > 1:
            filter_op = f"""<font color="red">{filter_op_tokens[0]}</font> {''.join(filter_op_tokens[1:])}"""
        filtering_lst[i] = (filtered_attr, filter_op, filter_term)

    filter_state = f"""
    <div class=large>
    <u><strong>Filtering State:</strong></u>
    <ul>"""

    for (filtered_attr, filter_op, filter_term) in filtering_lst:
        filter_state += f"""
    <li>Attr: <strong>{filtered_attr}</strong>  &emsp; OP: <strong>{filter_op}</strong>  &emsp; Term: <strong>{filter_term}</strong></li>
    """

    filter_state += """
    </ul>
    </div>
    </body>
    """
    return filter_state


def get_sessions_dfs_from_session_ids(repo, session_ids):
    """
    Return DataFrame of all actions from given session ids
    """
    solutions_df = repo.actions[repo.actions["session_id"].isin(session_ids)]
    return solutions_df


def get_solution_sessions_dfs_and_id(repo):
    """
    Return tuple (solutions_df, solutions_ids) of all human sessions with solution == True
    """
    solutions_df = repo.actions[repo.actions["solution"] == True]
    solutions_ids = set(solutions_df["session_id"].values)
    return solutions_df, solutions_ids


def load_gold_labels_test_dataset():
    """
    Load the gold-labeled test dataset created by add_to_snorkel_testset()
    
    Returns:
        List of dicts with keys:
            - 'snorkel_data_obj': Snorkel data object
            - 'gold_humanity_score': Analyst-provided humanity score (float)
    """
    from gym_atena.envs.atena_snorkel.data_loader import DataLoaderWithLabels
    
    # Load test data using DataLoaderWithLabels
    loader = DataLoaderWithLabels.load_test_data()
    
    # Convert to list of dicts for easy iteration
    test_examples = []
    for snorkel_obj, label in zip(loader.data, loader.labels):
        test_examples.append({
            'snorkel_data_obj': snorkel_obj,
            'gold_humanity_score': label
        })
    
    return test_examples


def add_to_snorkel_testset(env, dataset_number, action_vecs, analyst_humanity_scores):
    """
    Add samples to Snorkel test set
    
    Args:
        env: ATENA environment
        dataset_number: Dataset to use
        action_vecs: List of action vectors
        analyst_humanity_scores: List of analyst-provided humanity scores (floats)
    
    Returns:
        None
    """
    from gym_atena.envs.atena_snorkel.data_loader import save_gold_labels_test_dataset
    
    # Run episode and log to Snorkel test dataset
    info_hist, r_sum = run_episode(dataset_number=dataset_number,
                                   env=env,
                                   compressed=False,
                                   filter_by_field=True,
                                   continuous_filter_term=True,
                                   actions_lst=action_vecs,
                                   log_snorkel_test=True,
                                   verbose=False)
    
    assert len(action_vecs) == len(analyst_humanity_scores), \
        f"Number of actions ({len(action_vecs)}) must match number of humanity scores ({len(analyst_humanity_scores)})"
    
    # Write gold labels to file
    save_gold_labels_test_dataset(analyst_humanity_scores)


def run_episode(agent=None,
                dataset_number=None,
                env=None,
                compressed=False,
                filter_by_field=True,
                continuous_filter_term=True,
                actions_lst=None,
                most_probable=False,
                log_snorkel_test=False,
                verbose=True):
    """
    Run a single episode with given actions or agent
    
    Args:
        agent: TF2 agent (optional)
        dataset_number: Dataset to use
        env: Environment (created if None)
        compressed: Use compressed observations
        filter_by_field: Filter by field
        continuous_filter_term: Continuous filter terms
        actions_lst: List of actions (if None, agent chooses)
        most_probable: Use most probable actions
        log_snorkel_test: If True, log each step to Snorkel test dataset
        verbose: Print debug info
    
    Returns:
        (info_hist, r_sum): History and total reward
    """
    if most_probable:
        assert agent

    info_hist = []

    # Create env if needed
    if env is None:
        env = gym.make(env_d)
    
    # Unwrap environment if it's wrapped by gym.wrappers (e.g., TimeLimit)
    # This must happen BEFORE any type checks
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env
    
    # Verify we have the correct environment type (either base or enhanced)
    assert isinstance(base_env, (ATENAEnvCont, EnhancedATENAEnv)), \
        f"Expected ATENAEnvCont or EnhancedATENAEnv, got {type(base_env)}. Use env.unwrapped if wrapped."
    
    # Use the unwrapped environment for all operations
    env = base_env
    
    env.render()
    env.reset()

    # Set number of steps
    num_of_steps = cfg.MAX_NUM_OF_STEPS
    if actions_lst is not None:
        num_of_steps = len(actions_lst)

    s = env.reset(dataset_number=dataset_number)
    env.max_steps = num_of_steps

    r_sum = 0
    for ep_t in range(num_of_steps):
        if actions_lst is not None:
            a = actions_lst[ep_t]
        elif not agent:
            a = env.action_space.sample()
        else:
            if most_probable:
                # For TF2 agent, use model directly
                a = agent.get_action(s, deterministic=True)
            else:
                a = agent.get_action(s, deterministic=False)
        
        if verbose:
            print(a)
        
        s_, r, done, info = env.step(a, compressed=compressed, filter_by_field=filter_by_field,
                                     continuous_filter_term=continuous_filter_term)
        info_hist.append((info, r))

        if log_snorkel_test:
            # Create Snorkel candidate
            to_json = env.get_snorkel_obj_dict()
            
            # Log candidate to test dataset
            with open("snorkel_test_dataset.jsonl", mode='a', encoding='utf-8') as snorkel_f:
                snorkel_f.write(json.dumps(to_json, sort_keys=True) + '\n')

        if verbose:
            display(info["action"])
        
        s = s_
        r_sum += r
        if done:
            break
    
    return info_hist, r_sum


def get_action_html_obj(raw_action, filter_term):
    """Generate HTML object for action"""
    html_body = get_action_html_body(raw_action, filter_term)
    return get_action_html_obj_helper(html_body)


def get_action_html_body(raw_action, filter_term):
    """Generate HTML body for action"""
    env_prop = update_global_env_prop_from_cfg()
    act_string = ATENAUtils.OPERATOR_TYPE_LOOKUP[raw_action[0]]
    
    if act_string == "back":
        html_body = get_back_action_html_body()
    elif act_string == "group":
        grouped_attr = env_prop.env_dataset_prop.KEYS_ANALYST_STR[raw_action[1]]
        agg_func = ATENAUtils.AGG_MAP_ATENA_STR[raw_action[4]]
        agg_attr = env_prop.env_dataset_prop.AGG_KEYS_ANALYST_STR[raw_action[5]]
        html_body = get_group_action_html_body(grouped_attr, agg_func, agg_attr)
    elif act_string == "filter":
        filtered_attr = env_prop.env_dataset_prop.KEYS_ANALYST_STR[raw_action[1]]
        filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_PRETTY_STR[raw_action[2]]
        html_body = get_filter_action_html_body(filtered_attr, filter_op, filter_term)
    
    return html_body


def run_episode_analyst_view(agent=None,
                dataset_number=None,
                env=None,
                compressed=False,
                filter_by_field=True,
                continuous_filter_term=True,
                actions_lst=None,
                filter_terms_lst=None,
                verbose=True):
    """
    Run episode with HTML-based analyst view
    """
    if filter_terms_lst is not None:
        assert len(filter_terms_lst) == len(actions_lst)
    
    track_filter_terms_lst = []
    track_actions_lst = []

    run_id = uuid4()
    info_hist = []

    # Create env if needed
    if env is None:
        env = gym.make(env_d)
    
    # Unwrap environment if it's wrapped by gym.wrappers (e.g., TimeLimit)
    # This must happen BEFORE any type checks
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env
    
    # Verify we have the correct environment type (either base or enhanced)
    assert isinstance(base_env, (ATENAEnvCont, EnhancedATENAEnv)), \
        f"Expected ATENAEnvCont or EnhancedATENAEnv, got {type(base_env)}. Use env.unwrapped if wrapped."
    
    # Use the unwrapped environment for all operations
    env = base_env
    
    env.render()
    env.reset()

    # Set number of steps
    num_of_steps = cfg.MAX_NUM_OF_STEPS
    if actions_lst is not None:
        num_of_steps = len(actions_lst)

    s = env.reset(dataset_number=dataset_number)
    env.max_steps = num_of_steps

    r_sum = 0
    if verbose:
        display(get_dataset_number_html_obj(dataset_number, run_id))
        display(get_prev_next_buttons_html_ob_for_display_num(0, run_id))
        display(env.data)
    
    for ep_t in range(num_of_steps):
        if actions_lst is not None:
            a = actions_lst[ep_t]
        elif not agent:
            a = env.action_space.sample()
        else:
            a = agent.get_action(s)

        s_, r, done, info = env.step(a, compressed=compressed, filter_by_field=filter_by_field,
                                     continuous_filter_term=continuous_filter_term,
                                     filter_term=None if filter_terms_lst is None else filter_terms_lst[ep_t])

        track_actions_lst.append(info['raw_action'])
        track_filter_terms_lst.append(info['filter_term'])
      
        if verbose:
            # display action
            display(get_new_action_html_obj(ep_t+1, run_id))
            display(get_action_html_obj(info["raw_action"], info["filter_term"]))
            display(get_prev_next_buttons_html_ob_for_display_num(ep_t+1, run_id))
            
            # display filtering state
            filtering_state = env.states_hisotry[-1].filtering
            filtering_state_html_obj = get_filtering_state_html_obj(filtering_state)
            if filtering_state_html_obj:
                display(filtering_state_html_obj)

            # display tree
            draw_nx_display_tree(track_actions_lst, dataset_number=dataset_number, 
                                filter_terms_lst=track_filter_terms_lst if filter_terms_lst is None else filter_terms_lst[:ep_t+1])
            
            # display result
            if ATENAUtils.OPERATOR_TYPE_LOOKUP[info["raw_action"][0]] != "back":
                f, g = info["raw_display"]
                df_to_display = g if g is not None else f
                display(df_to_display)
        
        s = s_
        r_sum += r
        if done:
            break
    
    return info_hist, r_sum


def info_hist_to_raw_actions_lst(info_hist):
    """Convert info history to raw actions list"""
    actions_lst = []
    for info, _ in info_hist:
        info = deepcopy(info)
        info["raw_action"][3] -= 0.5
        actions_lst.append(info["raw_action"])
    return actions_lst


def simulate(info_hist, displays=False, verbose=True):
    """Display details about actions and rewards"""
    if verbose:
        for info, r in info_hist:
            info["raw_action"][3] -= 0.5
            print(f'{info["raw_action"]},')

    r_sum = 0
    for i, reward in info_hist:
        if verbose:
            print(f'action: {i["action"]} , reward: {reward}')
            print(f'raw action: {i["raw_action"]}')
            print(str(i["reward_info"]))
            print()
        else:
            print(i["action"])
        
        if displays:
            f, g = i["raw_display"]
            df_to_display = g if g is not None else f
            display(df_to_display)

        r_sum += reward
    
    if verbose:
        print("Total Reward:", r_sum)


def run_human_session(env, actions_lst, dataset_number, filter_terms_lst=None, verbose=True):
    """
    Run a human session with given actions
    
    Args:
        env: ATENA environment
        actions_lst: List of action vectors
        dataset_number: Dataset to use
        filter_terms_lst: Optional list of filter terms (one per action, None if no filter)
        verbose: Print debug info
    
    Returns:
        (info_hist, r_sum): History and total reward
    """
    # If filter terms are provided, use run_episode_analyst_view
    if filter_terms_lst is not None:
        info_hist, r_sum = run_episode_analyst_view(
            dataset_number=dataset_number,
            env=env,
            compressed=False,
            filter_by_field=False,
            continuous_filter_term=False,
            actions_lst=actions_lst,
            filter_terms_lst=filter_terms_lst,
            verbose=verbose
        )
    else:
        info_hist, r_sum = run_episode(
            dataset_number=dataset_number,
            env=env,
            compressed=False,
            filter_by_field=False,
            continuous_filter_term=False,
            actions_lst=actions_lst,
            verbose=verbose
        )
    return info_hist, r_sum


def analyze_reward(info_hist, actions_lst, summary_reward_data, verbose=True):
    """
    Analyze rewards and return DataFrame with detailed description
    """
    actions_info = [info["action"] for info, r in info_hist]
    reward_infos_dict = defaultdict(list)
    reward_infos_dict["action"] = actions_lst
    
    for info, r in info_hist:
        for key, val in info["reward_info"].items():
            reward_infos_dict[key].append(val)
    
    for info, r in info_hist:
        reward_infos_dict["total_reward"].append(r)
    
    reward_infos_dict["action_info"] = actions_info
    reward_df = DataFrame(reward_infos_dict)

    # Calculate averages
    rewards_list = [r for info, r in info_hist]
    rewards_list_no_back = [r for info, r in info_hist if info["action"] != "Back"]
    average_reward_per_action = np.mean(rewards_list)
    average_reward_per_non_back_action = np.mean(rewards_list_no_back)
    
    summary_reward_data['total_reward'].append(sum(rewards_list))
    summary_reward_data['avg_reward_per_action'].append(average_reward_per_action)
    summary_reward_data['avg_reward_per_non_back_action'].append(average_reward_per_non_back_action)
    summary_reward_data['num_of_actions'].append(len(rewards_list))
    
    if verbose:
        display(reward_df)
    
    return reward_df


def get_solution_details(sol_id, solutions_df, verbose=True):
    """
    Extract solution details from solutions DataFrame
    
    Returns:
        (action_vecs, actions_lst, dataset_id, num_of_steps)
    """
    # Get all human actions in session
    session_df = Repository.static_get_session_actions_by_id_all_rows(sol_id, solutions_df)

    # Get action parameters
    actions_params = Repository.get_all_actions_params_of_session(session_df)

    # Get action types
    actions_lst = Repository.get_all_actions_of_session(session_df)
    num_of_steps = len(actions_lst)

    # Get dataset_id
    dataset_id = Repository.get_dataset_number(session_df) - 1
    if verbose:
        print("Dataset id is %d" % dataset_id)

    # Convert to action vectors
    action_vecs = []
    for action, action_params in zip(actions_lst, actions_params):
        if verbose:
            print(action_params)
        action_vec = convert_to_action_vector(action, action_params)
        action_vecs.append(np.array(action_vec))
    
    return action_vecs, actions_lst, dataset_id, num_of_steps


def add_human_session_to_clusters(env, sol_id, solutions_df,
                                  human_displays_actions_clusters,
                                  human_displays_actions_clusters_obs_based,
                                  observation_collisions):
    """
    Add human session to observation clusters
    """
    # Extract solution details
    action_vecs, actions_lst, dataset_id, num_of_steps = get_solution_details(sol_id, solutions_df)

    # Add session to clusters
    add_human_session_to_clusters_helper(env, action_vecs, num_of_steps, dataset_id,
                                        human_displays_actions_clusters,
                                        human_displays_actions_clusters_obs_based,
                                        observation_collisions)


def add_human_session_to_clusters_helper(env, actions, max_steps, dataset_number,
                                         human_displays_actions_clusters,
                                         human_displays_actions_clusters_obs_based,
                                         observation_collisions,
                                         filter_by_field=False):
    """
    Helper to add session observations and actions to clusters
    """
    # Unwrap environment if it's wrapped by gym.wrappers (e.g., TimeLimit)
    # to access the custom step() method with compressed, filter_by_field, etc.
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    
    info_hist = []
    env.max_steps = max_steps
    env.render()
    s = env.reset(dataset_number=dataset_number)
    r_sum = 0
    
    for ep_t in range(max_steps):
        a = actions[ep_t]

        # get current state
        cur_state = deepcopy(env.history[-1])

        # make step
        s_, r, done, info = env.step(a, compressed=False, filter_by_field=filter_by_field,
                                     continuous_filter_term=False)

        info_hist.append((info, r))

        # get current and next DataFrames
        if ep_t == 0:
            cur_df_t = env.data
        else:
            cur_df_t = next_df_t

        next_df_t = info["raw_display"][1] if info["raw_display"][1] is not None else info["raw_display"][0]
        next_state = deepcopy(info["state"])

        # hash dataframe
        cur_df_t_hash = hashlib.sha256(pd.util.hash_pandas_object(cur_df_t, index=True).values).hexdigest()

        # add to cluster
        human_displays_actions_clusters[cur_df_t_hash][
            HumanStep(cur_obs=tuple(s), action_vector=tuple(a), next_obs=tuple(s_))] = (cur_state, dataset_number)
        
        human_displays_actions_clusters_obs_based[tuple(s)].append(a)

        # check collisions
        if tuple(s) in observation_collisions:
            print("collision")
            print(cur_df_t_hash)
            print(ep_t)
            display(cur_df_t)
            print(cur_state)

        # display action
        print(a)
        display(info["action"])

        s = s_
        r_sum += r
        if done:
            break
    
    return info_hist, r_sum

