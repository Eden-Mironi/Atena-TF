import json
import math
import logging
import random
from copy import deepcopy
from functools import lru_cache

import numpy as np

from cachetools import LRUCache

import gym
from gym import spaces

from scipy.stats import entropy
import scipy

from gym_atena.lib.tokenization import tokenize_column, get_nearest_neighbor_token, gep
#from gym_atena.reactida.utils.utilities import Repository
from gym_atena.reactida.utils.distance import display_distance
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.lib.helpers import (
    normalized_sigmoid_fkt,
    get_aggregate_attributes,
    empty_env_state,
    AggregationTuple,
    FilteringTuple,
)

# ENVIRONMENT FIX: Add missing imports to match master
try:
    from gym_atena.envs.atena_snorkel.atena_snorkel_networking_lfs import SnorkelNetRule
    from gym_atena.envs.atena_snorkel.atena_snorkel_flights_lfs import SnorkelFlightsRule  
    from gym_atena.envs.atena_snorkel.atena_snorkel_big_flights_lfs import SnorkelBigFlightsRule
    from gym_atena.envs.atena_snorkel.atena_snorkel_wide_flights_lfs import SnorkelWideFlightsRule
    from gym_atena.envs.atena_snorkel.atena_snorkel_wide12_flights_lfs import SnorkelWide12FlightsRule
    SNORKEL_AVAILABLE = True
except ImportError:
    # Snorkel components not available - continue without them
    SNORKEL_AVAILABLE = False
    print("Snorkel components not available - running without them")

from arguments import ArchName, FilterTermsBinsSizes
from Collections.Counter_Without_Nans import CounterWithoutNanKeys
import Configuration.config as cfg

logger = logging.getLogger(__name__)


class ATENAEnvCont(gym.Env):
    """The main Atena environment class


    Attributes:
        max_steps (int): The number of steps in each episode
        repo (Obj): The repository of human session
        observation_space (Obj): The observation space (Box)
        action_space (Obj): The action space (Box)
        ret_df (bool): Wheter to return the dataframes in each observation
        data (obj): Dataframe containing the current dataset of the episode
        history (list): captures all "state" dictionaries in the current episode, each one is equivalent to a query
        ahist (list): a list of actions performed thus far in the episode
        dhist (list): a list of the corresponding result-displays of the actions

    """

    LOG_INTERVAL = 100

    NUM_OF_EPISODES = 0

    metadata = {
        'render.modes': ['human'],
    }

    # cache where key is (dataset_num, state) and value is the tuple (observation, display, dfs)
    STATE_DF_HISTORY = None

    # cache where key is (dataset_num, state, col) and value is the tuple (sorted_token_frequency_pairs_lst, sorted_frequencies_lst)
    COL_TOKENIZATION_HISTORY = None

    # cache where key is (dataset_number, state1, state2) and value is the distance between the displays they are representing
    # Note that you key should be ordered such that str(state1) <= str(state2) since the distance is symmetric and we don't
    # want duplicates
    STATES_DISP_DISTANCE_HISTORY = None

    # architecture
    arch = ArchName(cfg.arch)

    # length of a single display
    len_single_display_vec = None

    # a static variable env for various uses so that we won't have to create a new environment
    # which is expensive
    static_env = None

    def __init__(self, max_steps=cfg.MAX_NUM_OF_STEPS, ret_df=False, gradual_training=False):
        """When initializing the environment class the following happens:
        (1) The data files and the human session repositories are loaded
        (2) The action space and observation space are devined

        Args:
            max_steps (int): The maximum number of steps in an episodes
            env_prop (BasicEnvProp): Environment properties of allowed actions and explored dataset
        """
        self.env_prop = gep.update_global_env_prop_from_cfg()

        # reset caches if needed
        if self._is_caches_reset_needed():
            self.reset_caches()

        # (0) Initialize some attributes, others are initialized in the reset() method
        self.gradual_training = gradual_training
        self.max_steps = max_steps
        self.ret_df = ret_df
        self.env_dataset_prop = self.env_prop.env_dataset_prop

        # (1) Loading data files and session reposiotires. Note that the class Repository is taken from REACT
        self.repo = self.env_dataset_prop.repo

        # (2.a) Define the action space:
        #        0) action_type:            back[0], filter[1], group[2]
        #        1) col_id:                 [0..num_of_columns-1]
        #        2) filter_operator         [LT, GT, etc..]
        #        3) filter_term:          taken from a fixed list of tokens
        #        4) aggregation column_id:  [0..num_of_columns - 1]
        #        5) aggregation function:       [mean, count, etc.]

        # self.action_space = spaces.MultiDiscrete([ACTION_TYPES_NO,COLS_NO,FILTER_OPS,FILTER_TERMS_NO,COLS_NO, AGG_FUNCS_NO])
        # self.action_space=spaces.Box(low=np.zeros(6)-0.49,high=np.array([ACTION_TYPES_NO, COLS_NO, FILTER_OPS, FILTER_TERMS_NO, COLS_NO, AGG_FUNCS_NO])-0.51,dtype='float32')
        # Use master's symmetric action space (ACTION_RANGE = 6.0)
        # This matches master's: spaces.Box(low=[-3.0]*6, high=[3.0]*6)  
        self.action_space = spaces.Box(low=np.zeros(6) - self.env_prop.ACTION_RANGE / 2,
                                       high=np.zeros(6) + self.env_prop.ACTION_RANGE / 2, dtype='float32')

        # (2.b) Define the observation space:
        # z_step_number(optional): a binary 1-based(!) vector in size of self.max_steps with binary values
        # s.t. there is a single bit on corresponding to the current step number starting from 1
        # z: {num of Unique, num of nulls, normalized entropy} for each column
        # z2: {Grouped or aggregated state} for each column:
        #     -1 if none, [0-1] if aggregated (value means the NVE), and 2 if grouped
        # z3: {num of groups, mean group size, size variance}
        z = np.zeros(len(self.env_dataset_prop.KEYS) * 3)

        z2 = np.full(len(self.env_dataset_prop.KEYS), -1)
        z3 = np.zeros(3)

        low = np.tile(np.concatenate([z, z2, z3]), cfg.stack_obs_num)
        high = np.tile(np.concatenate([np.ones(len(z)), np.full(len(self.env_dataset_prop.KEYS), 2), np.ones(3)]), cfg.stack_obs_num)

        if cfg.obs_with_step_num:
            z_step_number = np.zeros(self.max_steps)
            low = np.concatenate([z_step_number, low])
            high = np.concatenate([np.ones(len(z_step_number)), high])
        # else:
        #    low = np.concatenate([z, z2, z3])
        #    high = np.concatenate([np.ones(len(z)), np.full(len(self.env_dataset_prop.KEYS), 2), np.ones(3)])

        self.observation_space = spaces.Box(low, high, dtype='float32')

        ATENAEnvCont.len_single_display_vec = len(z) + len(z2) + len(z3)

        # ENVIRONMENT FIX: Restore Snorkel initialization (matching master)
        if SNORKEL_AVAILABLE:
            try:
                # Load snorkel generative model
                self.snorkel_gen_model = self.env_dataset_prop.get_snorkel_gen_model()
                # Set snorkel path to write training examples  
                self.snorkel_training_path = self.env_dataset_prop.snorkel_training_path
            except Exception as e:
                print(f"Warning: Could not initialize Snorkel components: {e}")
                self.snorkel_gen_model = None
                self.snorkel_training_path = None
        else:
            self.snorkel_gen_model = None
            self.snorkel_training_path = None

        # Print dataset name if one is chosen
        if cfg.dataset_number is not None:
            print(self.repo.file_list[cfg.dataset_number])

        self._log = True

    @classmethod
    def reset_caches(cls):
        cls.STATE_DF_HISTORY = None
        if cfg.cache_dfs_size > -1:
            cls.STATE_DF_HISTORY = LRUCache(maxsize=cfg.cache_dfs_size)

        cls.COL_TOKENIZATION_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.COL_TOKENIZATION_HISTORY = LRUCache(maxsize=cfg.cache_tokenization_size)

        cls.STATES_DISP_DISTANCE_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.STATES_DISP_DISTANCE_HISTORY = LRUCache(maxsize=cfg.cache_distances_size)

    def _is_caches_reset_needed(self):
        if (self.STATES_DISP_DISTANCE_HISTORY is None
                and self.STATE_DF_HISTORY is None
                and self.COL_TOKENIZATION_HISTORY is None):
            return True
        return False

    ##############################

    # Internal Functions

    ##############################
    def translate_action(self, act_vector, filter_by_field=True, filter_term=None):
        """This function translate an action vector into a human-readable action

        Args:
            act_vector (numpy array): the action vector

        Returns:
            A human readable string that corresponds to the action vector
        """
        if type(act_vector) is not list:
            act_vector = self.cont2dis(act_vector)
        
        # ENVIRONMENT FIX: Match master's action processing
        rtype = self.env_prop.OPERATOR_TYPE_LOOKUP.get(act_vector[0])
        if rtype == "back":
            return "Back"
        elif rtype == "filter":
            col = self.data.columns[act_vector[1]]
            cond = self.env_prop.INT_OPERATOR_MAP_ATENA.get(act_vector[2])
            if not cond:
                op_num = act_vector[2]
                if op_num in [6, 7, 8]:  # op_num == 6:
                    cond = 'str.contains'
                elif op_num == 7:
                    cond = 'str.stratswith'
                elif op_num == 8:
                    cond = 'str.endswith'
            if not filter_by_field:
                term = self.env_dataset_prop.FILTER_LIST[act_vector[3]]
            else:
                """filter_field_list = FILTER_BY_FIELD_DICT.get(col)
                if filter_field_list and len(filter_field_list)-1 >= act_vector[3]:
                    term = filter_field_list[act_vector[3]]
                else:
                    term = '<UNK>'"""
                term = filter_term
            return "Filter on Column '%s', using condition '%s', with term '%s'" % (col, str(cond), term)
        elif rtype == "group":
            col = self.data.columns[act_vector[1]]
            agg_col = self.data.columns[act_vector[4]]
            agg_func = self.env_prop.AGG_MAP_ATENA.get(act_vector[5])
            return "Group on Column '%s' and aggregate with '%s' on the column '%s'" % (col, str(agg_func), agg_col)
        else:
            # ENVIRONMENT FIX: Match master's error handling
            raise NotImplementedError

    @lru_cache(maxsize=2048)
    def get_exponential_filter_term_bins(self, num_of_rows, num_of_bins):
        """
        Create `num_of_bins` bins for filter terms such that bins width grows exponentially with frequency of these bins
        Args:
            num_of_rows:
            num_of_bins:

        Returns:

        """
        assert num_of_bins >= 1
        if num_of_rows == 0:
            return [0.0] * num_of_bins + [1.0]
        B_minus1 = num_of_bins - 1
        x = (num_of_rows ** (1 / B_minus1))
        single_row_contribution = 1 / num_of_rows
        bins = [0] + [round(x ** i) / num_of_rows for i in range(num_of_bins)]
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + single_row_contribution
        return bins

    @staticmethod
    def _param_softmax_idx_to_action_helper(idx):
        raise NotImplementedError

    def param_softmax_idx_to_action(self, idx):
        """
        Maps an index that represents one off all possible discrete actions in the environment
        to a legal action in the environment (i.e. a vector of size action_space)
        Args:
            idx (int): index of an entry in the output vector of an architecture
            of type PARAM_SOFTMAX

        Returns:

        """
        result = np.zeros(6, dtype=np.float32)
        (action_type, parameters) = self.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[idx]

        result[0] = action_type
        action_type_string = self.env_prop.OPERATOR_TYPE_LOOKUP[action_type]
        if action_type_string == "back":
            pass
        elif action_type_string == "filter":
            result[1] = parameters[0]

            result[2] = (parameters[1] + 1) * 3 - 1

            # ENVIRONMENT FIX: Restore master's filter term processing
            if cfg.filter_from_list:
                result[3] = parameters[2]
            else:
                filter_terms_bin_sizes = FilterTermsBinsSizes(cfg.bins_sizes)
                if filter_terms_bin_sizes is FilterTermsBinsSizes.EQUAL_WIDTH:
                    bin_size = 1 / (self.env_prop.DISCRETE_FILTER_TERM_BINS_NUM - 1)
                    result[3] = parameters[2] * bin_size + np.random.uniform(-bin_size, bin_size) - 0.5
                elif filter_terms_bin_sizes is FilterTermsBinsSizes.CUSTOM_WIDTH:
                    lower_bin_edge = self.env_prop.bins[parameters[2]]
                    upper_bin_edge = self.env_prop.bins[parameters[2] + 1]
                    bin_size = upper_bin_edge - lower_bin_edge
                    result[3] = np.random.uniform(lower_bin_edge, upper_bin_edge) - 0.5
                elif filter_terms_bin_sizes is FilterTermsBinsSizes.EXPONENTIAL:
                    num_of_rows = self.num_of_fdf_rows_hist[-1]
                    bins = self.get_exponential_filter_term_bins(num_of_rows, cfg.exponential_sizes_num_of_bins)
                    lower_bin_edge = bins[parameters[2]]
                    upper_bin_edge = bins[parameters[2] + 1]
                    bin_size = upper_bin_edge - lower_bin_edge
                    result[3] = np.random.uniform(lower_bin_edge, upper_bin_edge) - 0.5
                else:
                    raise NotImplementedError

        elif action_type_string == "group":
            result[1] = parameters[0]
        else:
            raise ValueError("action_type should refer to back filter or group")

        return result

    @staticmethod
    def static_param_softmax_idx_to_action(idx):
        """
        Maps an index that represents one off all possible discrete actions in the environment
        to a legal action in the environment (i.e. a vector of size action_space)
        Args:
            idx (int): index of an entry in the output vector of an architecture
            of type PARAM_SOFTMAX

        Returns:

        """
        result, _, _ = ATENAEnvCont._param_softmax_idx_to_action_helper(idx)
        return result


    def compressed2full_range(self, action_vec, continuous_filter_term=True):
        """
        CRITICAL FIX: Added missing scaling function from master!
        
        Change a compressed range vector to full range based on the range of each entry
        and clip the vector to be in the legal ranges (EXACTLY matching master's implementation)
        
        Args:
            action_vec: Continuous action vector in [-3, +3] range
            continuous_filter_term: Boolean whether continuous_filter_term or discrete
        
        Returns:
            Scaled action vector ready for discretization
        """
        RANGE = self.env_prop.ACTION_RANGE  # 6.0
        entries_ranges = np.array([
            self.env_prop.ACTION_TYPES_NO,
            self.env_dataset_prop.COLS_NO,
            self.env_prop.FILTER_OPS,
            self.env_prop.MAX_FILTER_TERMS_BY_FIELD_NO,
            self.env_dataset_prop.AGG_COLS_NO,
            self.env_prop.AGG_FUNCS_NO
        ])
        
        # Master's exact scaling formula
        full_range = np.multiply(np.array((action_vec + RANGE / 2) / RANGE), entries_ranges) - 0.5
        full_range_filter_term = full_range[3]
        clipped = np.clip(full_range, np.zeros(6), entries_ranges - 1)
        
        if continuous_filter_term:
            clipped[3] = full_range_filter_term
            
        return clipped

    def cont2dis(self, c_vector):
        """This function discretizes (rounds) a continuous (float) action vector
        
        CRITICAL: Master's exact implementation - simple rounding with NO clamping!
        Clipping is handled by compressed2full_range() BEFORE this function is called.

        Args:
            C_vector (numpy array): a continuous (float) action vector

        Returns:
            A vector of discrete integer representing the actions
        """
        # EXACT MASTER MATCH: Simple rounding, no clamping (line 338 in master)
        return list(np.array(np.round(c_vector), dtype=int))

    @staticmethod
    def _is_empty_display(obs):
        start_of_cur_display_suv_vec = len(obs) - ATENAEnvCont.len_single_display_vec
        return np.array_equal(
            obs[start_of_cur_display_suv_vec:3 + start_of_cur_display_suv_vec],
            [0, 1, 0])

    @staticmethod
    def _is_empty_groupings(obs):
        return np.array_equal(obs[-3:], [0, 0, 0])

    def reset(self, dataset_number=None):
        """This function starts a new episode. It performs the following steps:
            (1) Randomly choose a dataset
            (2) Initialize the history lists: history,ahist,dhist

            :param dataset_number: if set, the dataset with this nubmer will be
            loaded, else a random dataset will be chosen
            :return: The first observation vector describing the chosen dataset
        """
        # resample a seed (MATCHING MASTER - using np.random instead of scipy.random):
        random.seed()
        np.random.seed()  # scipy.random doesn't exist in current scipy version

        self.NUM_OF_EPISODES += 1
        self.step_num = 0
        if self.gradual_training:
            # MATCHING MASTER: Variable episode lengths for diversity
            self.max_steps = random.randint(2, max(3, int(self.NUM_OF_EPISODES / 2500)))

        self._log = True if self.NUM_OF_EPISODES % self.LOG_INTERVAL == 0 else False

        # (1) Choose a dataset:
        if cfg.dataset_number is not None:
            dataset_number = cfg.dataset_number
        elif dataset_number is None:
            # MATCHING MASTER: Random dataset selection for diversity
            dataset_number = np.random.randint(len(self.repo.data))
        self.dataset_number = dataset_number
        if self._log:
            logger.info(f"Dataset number chosen is {dataset_number}, dataset name is {self.repo.file_list[dataset_number]}")

        self.data = self.repo.data[dataset_number][self.env_dataset_prop.KEYS]

        # Initialize history of states lists:
        empty_state = empty_env_state
        # history is a stack of state (back actions pop elements from list)
        self.history = [empty_state]
        # states_history is a list of all states during session
        self.states_hisotry = [empty_state]
        # stack for history of observations (back actions pop elements from list)
        self.obs_hist = []
        # obs_hist_all is a list of all observations during the session
        self.obs_hist_all = []

        # Calculate the display and the observation vector:
        obs, disp, _ = self.env_prop.calc_display_vector(self.data,
                                           empty_state,
                                           memo=self.STATE_DF_HISTORY,
                                           dataset_number=self.dataset_number,
                                           step_number=self.step_num,
                                           states_hist=self.history,
                                           obs_hist=self.obs_hist,
                                           len_single_display_vec=self.len_single_display_vec
                                           )

        # Display history will contain the first display
        self.dhist = [disp]
        self.ahist = []
        self.obs_hist = [obs]
        self.obs_hist_all = [obs]
        self.filter_terms_hist = []
        self.num_of_rows_hist = [len(self.data)]  # Number of rows or groups (if grouped) in the current display
        self.num_of_fdf_rows_hist = [len(self.data)]  # Number of rows in the current display fdf
        # Number of rows if the current action is group and the previous action is filter if we did not take the
        # filter action and make an immediate group action instead
        self.num_of_immediate_action_rows_lst = [None]

        # determine if we are in the middle of a point inside the session where the
        # a group operation was empty, and the number of operations in this subsession
        self.in_the_middle_of_empty_grouping = False
        self.in_the_middle_of_empty_grouping_steps = 0

        assert self.observation_space.contains(obs)
        return obs

    @property
    def arch(self):
        return ArchName(cfg.arch)

    def compute_interestingness_reward(self, dfs, state, last_action_type):
        kl_distance = compaction_gain = 0
        if self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == "filter":  # filter
            kl_distance = self.compute_interestingness_kl_divergence(dfs, state)
            kl_distance = kl_distance * cfg.kl_coeff
        elif self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'group':  # group
            compaction_gain = self.compute_interestingness_compaction_gain(dfs, state)
            compaction_gain = compaction_gain * cfg.compaction_coeff

        return StepReward(kl_distance=kl_distance, compaction_gain=compaction_gain)


    def compute_interestingness_kl_divergence(self, dfs, state):
        '''
        #Returns 1-log(3)^max_{KL_div_attr for each attribute in the current dataframe}(-KL_div_attr)
        Returns sigmoid(max_{KL_div_attr for each attribute in the current datafame}/2-3)
        :param dfs:
        :param state:
        :return:
        '''
        kl_distances = []
        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        df_D = self.get_previous_df(past_steps=2)
        # find attributes to compute KL divergence for
        if is_grouping:
            # calculate grouped and aggregated version of D
            # according to groupings and aggregation of d_t
            # but without filtering of d_t
            # D_state = state
            # D_state = D_state.reset_filtering()
            # D_dfs = get_state_dfs(self.data,
            #                       D_state,
            #                       self.env_dataset_prop.KEYS,
            #                       memo=self.STATE_DF_HISTORY,
            #                       dataset_number=self.dataset_number,
            #                       )
            #
            # df_D = D_dfs[1]
            aggeregate_attributes_list = get_aggregate_attributes(state)
            kl_attrs = aggeregate_attributes_list
            assert len(set(kl_attrs).difference(set(self.env_dataset_prop.AGG_KEYS))) == 0, "Only primary key is supported now, use gdf to support more aggregated attributes"
        else:
            # df_D = self.data
            KL_DIV_EPSILON = 2 / len(df_D) * 0.1
            kl_attrs = self.data.columns

        # compute KL_divergence for each attribute
        for attr in kl_attrs:
            # attr_value_count1 = df_D[attr].value_counts().to_dict()
            # attr_value_count2 = df_dt[attr].value_counts().to_dict()
            # attr_value_count1 = Counter(df_D[attr].dropna().values)
            # attr_value_count2 = Counter(df_dt[attr].dropna().values)

            attr_value_count1 = CounterWithoutNanKeys(df_D[attr].values)
            attr_value_count2 = CounterWithoutNanKeys(df_dt[attr].values)

            if is_grouping:
                elements_sum = sum(attr_value_count1.elements())
                KL_DIV_EPSILON = 2 / max(1, elements_sum) * 0.1  # FIX: Prevent division by zero

            '''if not is_grouping:
                num_of_NaNs_1 = len(df_D) - sum(attr_value_count1.values())
                num_of_NaNs_2 = len(df_dt) - sum(attr_value_count2.values())'''

            pk1 = []
            pk2 = []
            for key in attr_value_count1:
                pk1.append(attr_value_count1[key])
                if key in attr_value_count2:
                    pk2.append(attr_value_count2[key])
                else:
                    pk2.append(KL_DIV_EPSILON)

            # add the rest of attributes not in attr_value_count1
            for key in attr_value_count2:
                if key not in attr_value_count1:
                    pk2.append(attr_value_count2[key])
                    pk1.append(KL_DIV_EPSILON)

            # add NaNs number for non-grouping case
            '''if not is_grouping:
                if num_of_NaNs_1 != 0 or num_of_NaNs_2 != 0:
                    num_of_NaNs_1 = num_of_NaNs_1 if num_of_NaNs_1 != 0 else KL_DIV_EPSILON
                    num_of_NaNs_2 = num_of_NaNs_2 if num_of_NaNs_2 != 0 else KL_DIV_EPSILON
                    pk1.append(num_of_NaNs_1)
                    pk2.append(num_of_NaNs_2)'''

            attr_kl_div = entropy(pk1, pk2)
            kl_distances.append(attr_kl_div)
        # return 1-math.log(3)**(-max(kl_distances))
        return 1 / (1 + math.exp(-(max(kl_distances) / 2 - 3)))

    @staticmethod
    def get_filtered_only_or_grouped_data_frame(dfs):
        '''
        get a tuple of dataframes (dfs=[fdf, adf] as is returned from calc_display_vector())
        and returns the first dataframe if no grouping is indicated, else the second.
        Moreover, a Boolean value indicating whether a grouping was indicated is returned.

        Note: this function asssumed that if dfs[1] is None there is no grouping (which is not
        only the case. Whether a dataframe is grouped can only be determined by the state dictionary)
        :param dfs:
        :return:
        '''
        is_grouping = True if dfs[1] is not None else False
        if is_grouping:
            df_dt = dfs[1]  # df <- adf (df is filtered, grouped and aggregated)
        else:
            df_dt = dfs[0]  # df has filtering only
        return df_dt, is_grouping


    def compute_rule_based_humanity_score(self, dfs, state, rules_reward_info, done):
        """CRITICAL MISSING METHOD: Compute rule-based humanity score exactly like master

        Args:
            dfs:
            state:
            rules_reward_info (HumanRulesReward): A data structure used to collects statistics about
            the rules triggered. Passed by reference!
            done (bool): Whether or not this is the last step of the episode

        Returns: The humanity score + changes rules_reward_info

        """
        humanity_rules_based_reward = self._compute_rule_based_humanity_score(dfs, state, rules_reward_info, done)
        return humanity_rules_based_reward

    def _compute_rule_based_humanity_score(self, dfs, state, rules_reward_info, done):
        """CRITICAL MISSING METHOD: Private implementation matching master exactly"""
        try:
            return self.env_dataset_prop.rule_based_humanity_score_func(self, dfs, state, rules_reward_info, done)
        except AttributeError:
            # FALLBACK: If rule_based_humanity_score_func is missing, return 0
            # This maintains compatibility while the function is implemented
            print("Warning: rule_based_humanity_score_func not found in env_dataset_prop, using fallback")
            return 0

    def get_snorkel_obj_dict(self):
        """
        Create Snorkel candidate dictionary
        
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
    
    def create_snorkel_data_obj(self):
        """Create Snorkel data object for coherency scoring"""
        return self.env_dataset_prop.snorkel_data_obj.construct_from_dict(
            self.get_snorkel_obj_dict(), self.max_steps
        )
    
    @staticmethod
    def is_invalid_filter_term(term):
        """
        GENERIC HELPER: Check if a filter term is invalid in ANY format
        
        Catches:
        - String '<UNK>' or 'nan' (case-insensitive)
        - Float NaN (math.nan)
        - None
        - Empty string
        
        Returns:
            bool: True if term is invalid, False otherwise
        """
        import math
        
        # Check 1: None or empty string
        if term is None or term == '':
            return True
        
        # Check 2: String '<UNK>' or 'nan' (case-insensitive)
        if isinstance(term, str):
            term_lower = term.lower()
            if term_lower == '<unk>' or term_lower == 'nan':
                return True
        
        # Check 3: Float NaN (use try-except to handle any type safely)
        else:
            try:
                if math.isnan(term):
                    return True
            except (TypeError, ValueError):
                # Not a numeric type that can be NaN
                pass
        
        return False
    
    def compute_snorkel_humanity_score(self):
        """CRITICAL MISSING METHOD: Compute Snorkel humanity score matching master exactly"""
        # FIX: Check if last FILTER action had invalid term - return STRONG NEGATIVE
        # Note: filter_terms_hist contains None for non-filter actions (group/back)
        if self.filter_terms_hist and len(self.filter_terms_hist) > 0:
            last_filter_term = self.filter_terms_hist[-1]
            
            # Only check if it's NOT None (None means it was a non-filter action)
            if last_filter_term is not None and self.is_invalid_filter_term(last_filter_term):
                # Return STRONG NEGATIVE to discourage learning this behavior
                # -1.0 will be multiplied by humanity_coeff (4.5) → -4.5 total
                logger.warning(f"Snorkel: Invalid filter term '{last_filter_term}' (type: {type(last_filter_term).__name__}) - returning strong negative humanity score")
                return -1.0, {}
        
        snorkel_data_obj = self.create_snorkel_data_obj()
        
        snorkel_marginal, non_abstain_funcs_dict = self.snorkel_gen_model.get_coherency_score_prediction_and_non_abstain_funcs(
            snorkel_data_obj
        )
        
        # Change marginal from range [0,1] to [-1,1] - EXACTLY like master
        return snorkel_marginal * 2 - 1, non_abstain_funcs_dict

    def reward(self, obs, dfs, state, no_history_for_back):
        """This function determines the reward for an observation. It perfoms the following:
         (0) Check if "Done", i.e. the maximum number of steps is reached
         (1) Determines the reward (punishments + diversity based positive reward)

        Args:
            obs (obj): the observation vector

        Returns:
            bool: Is "done"
            float: A reward
            reward_info: dict of details about each reward component
        """
        reward_info = StepReward()
        rules_reward_info = HumanRulesReward()
        snorkel_rules_reward_info = HumanRulesReward()

        # (0) check if done:
        self.step_num += 1
        done = self.step_num >= self.max_steps

        # (1) Determine the reward:
        last_action_type = self.ahist[-1][0]

        # (1.a) punishment for 'back' action with no previous displays
        if no_history_for_back:
            r = -1.0 * cfg.humanity_coeff  # Match Master exactly
            reward_info.back = r
            reward_info.is_back = True

        # (1.b) if last action is 'back' - calculate reward like Master, then add bonus
        elif self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'back':
            # MATCH MASTER: Start with 0, replace with humanity/Snorkel
            r = 0
            is_data_driven = False

            # FIX CRITICAL BUG: Compute humanity for back actions like master!
            r_humanity = self.compute_rule_based_humanity_score(dfs, state, rules_reward_info, done)
            r_humanity *= cfg.humanity_coeff
            reward_info.humanity = r_humanity

            # FIX: Add master's Snorkel handling for back actions  
            r_snorkel_humanity = 0
            if cfg.use_snorkel and SNORKEL_AVAILABLE and self.snorkel_gen_model is not None:
                try:
                    r_snorkel_humanity, non_abstain_funcs_dict = self.compute_snorkel_humanity_score()
                    snorkel_rules_reward_info = HumanRulesReward(non_abstain_funcs_dict)

                    # Master's multiplier for back after bad filter (multiply by 4)
                    if (SnorkelNetRule.LF_back_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                        SnorkelFlightsRule.LF_back_after_bad_filter_readability_gain in snorkel_rules_reward_info or  
                        SnorkelBigFlightsRule.LF_back_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                        SnorkelWideFlightsRule.LF_back_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                        SnorkelWide12FlightsRule.LF_back_after_bad_filter_readability_gain in snorkel_rules_reward_info
                    ):
                        # FIX: Allow neutral (0.0) Snorkel predictions, but warn for unexpected positive values
                        if r_snorkel_humanity > 0:
                            print(f"WARNING: Expected negative/neutral Snorkel humanity for punishment rule, got {r_snorkel_humanity:.6f}")
                        # Apply 4x multiplier for back-after-bad-filter punishment (even for neutral scores)
                        r_snorkel_humanity *= 4

                    r_snorkel_humanity *= cfg.humanity_coeff
                    if is_data_driven:
                        reward_info.interestingness = r_snorkel_humanity
                    else:
                        reward_info.snorkel_humanity = r_snorkel_humanity
                except (AttributeError, TypeError) as e:
                    print(f"Snorkel computation failed for back action: {e}")
                    r_snorkel_humanity = 0

            # MATCH MASTER: REPLACE (not add) with humanity/Snorkel
            if cfg.use_humans_reward:
                if cfg.use_snorkel and not is_data_driven:
                    r = r_snorkel_humanity  # REPLACE with Snorkel (Master's way)
                else:
                    r = r_humanity  # REPLACE with humanity (Master's way)

            if cfg.use_snorkel and is_data_driven and not cfg.no_interestingness:
                r = r_snorkel_humanity  # REPLACE with Snorkel (Master's way)

            # No bonus - Snorkel should work exactly like Master
            reward_info.back = r
            reward_info.is_back = True

            if self.in_the_middle_of_empty_grouping:
                self.in_the_middle_of_empty_grouping_steps -= 1
            if self.in_the_middle_of_empty_grouping_steps == 0:
                self.in_the_middle_of_empty_grouping = False

        # (1.c) punishment for empty results.
        # Add direct DataFrame check as fallback
        # The observation-based check might miss some empty displays, so we also check the actual DataFrame
        elif self._is_empty_display(obs) or (dfs[0] is not None and len(dfs[0]) == 0):
            r = -1.0 * cfg.humanity_coeff

            reward_info.empty_display = r
            reward_info.interestingness = r
            
            # Zero out all other reward components for empty displays
            # This prevents the enhanced wrapper from summing them
            reward_info.diversity = 0
            reward_info.humanity = 0
            reward_info.snorkel_humanity = 0
            reward_info.kl_distance = 0
            reward_info.compaction_gain = 0
            
            # DEBUG: Log when empty display is detected
            if dfs[0] is not None and len(dfs[0]) == 0:
                logger.warning(f"Empty display detected via DataFrame check (len={len(dfs[0])})")

        # (1.d) punishment for empty grouping:
        elif (state['grouping'] and (dfs[1] is None or self._is_empty_groupings(obs)) or
              self.in_the_middle_of_empty_grouping):
            r = -1.0 * cfg.humanity_coeff
            self.in_the_middle_of_empty_grouping = True
            self.in_the_middle_of_empty_grouping_steps += 1

            reward_info.empty_groupings = r
            reward_info.interestingness = r
            
            # Zero out all other reward components for empty groupings
            reward_info.diversity = 0
            reward_info.humanity = 0
            reward_info.snorkel_humanity = 0
            reward_info.kl_distance = 0
            reward_info.compaction_gain = 0
            reward_info.empty_display = 0


        else:
            # Diversity
            r_div = self._compute_diversity_reward(last_action_type, reward_info)

            # calculate interestingness and humanity reward only if the
            # current display hasn't been already seen
            r_inter = 0
            r_humanity = 0
            r_snorkel_humanity = 0
            if not reward_info.is_same_display_seen_already:
                # interestingness
                interestingness_rewards = self.compute_interestingness_reward(dfs, state, last_action_type)
                reward_info.kl_distance = interestingness_rewards.kl_distance
                reward_info.compaction_gain = interestingness_rewards.compaction_gain
                r_inter = max(interestingness_rewards.values())
                reward_info.interestingness = r_inter

                # humanity
                r_humanity = self.compute_rule_based_humanity_score(dfs, state, rules_reward_info, done)
                # FIX: Use fresh config for current coefficient values
                r_humanity *= cfg.humanity_coeff
                print(f"Rule-based humanity coefficient applied: {cfg.humanity_coeff}")
                reward_info.humanity = r_humanity

                # CRITICAL MISSING COMPONENT: snorkel_humanity reward 
                if cfg.use_snorkel and SNORKEL_AVAILABLE and self.snorkel_gen_model is not None:
                    try:
                        r_snorkel_humanity, non_abstain_funcs_dict = self.compute_snorkel_humanity_score()
                        
                        snorkel_rules_reward_info = HumanRulesReward(non_abstain_funcs_dict)
                        
                        # Apply master's complex multiplier rules
                        # Multiply by 4 in case of group after non readability gain filter
                        if (SnorkelNetRule.LF_group_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelFlightsRule.LF_group_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelBigFlightsRule.LF_group_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelWideFlightsRule.LF_group_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelWide12FlightsRule.LF_group_after_bad_filter_readability_gain in snorkel_rules_reward_info
                        ):
                            # Only apply multiplier if not abstaining (r != 0)
                            if r_snorkel_humanity != 0:
                                assert r_snorkel_humanity < 0, f"Expected negative Snorkel score, got {r_snorkel_humanity}"
                                r_snorkel_humanity *= 4
                            
                        # Additional multiplier rules from master
                        if (SnorkelNetRule.LF_filter_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelFlightsRule.LF_filter_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelBigFlightsRule.LF_filter_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelWideFlightsRule.LF_filter_after_bad_filter_readability_gain in snorkel_rules_reward_info or
                            SnorkelWide12FlightsRule.LF_filter_after_bad_filter_readability_gain in snorkel_rules_reward_info
                        ):
                            # Only apply multiplier if not abstaining (r != 0)
                            if r_snorkel_humanity != 0:
                                if r_snorkel_humanity > 0:
                                    r_snorkel_humanity -= 1.0
                                assert r_snorkel_humanity < 0, f"Expected negative Snorkel score, got {r_snorkel_humanity}"
                                r_snorkel_humanity *= 4
                            
                        # Multiply by 2 for recursive group after non readability gain filter  
                        if (SnorkelNetRule.LF_group_after_recursive_bad_filter_readability_gain in snorkel_rules_reward_info
                            or SnorkelFlightsRule.LF_group_after_recursive_bad_filter_readability_gain in snorkel_rules_reward_info
                            or SnorkelBigFlightsRule.LF_group_after_recursive_bad_filter_readability_gain in snorkel_rules_reward_info
                            or SnorkelWideFlightsRule.LF_group_after_recursive_bad_filter_readability_gain in snorkel_rules_reward_info
                            or SnorkelWide12FlightsRule.LF_group_after_recursive_bad_filter_readability_gain in snorkel_rules_reward_info
                        ):
                            # Only apply multiplier if not abstaining (r != 0)
                            if r_snorkel_humanity != 0:
                                assert r_snorkel_humanity < 0, f"Expected negative Snorkel score, got {r_snorkel_humanity}"
                                r_snorkel_humanity *= 2
                            
                        # Multiply by 2.0 for good filters
                        if (r_snorkel_humanity > 0 and
                                (
                                        (SnorkelNetRule.LF_filter_terms_vulnarable_protocols in snorkel_rules_reward_info
                                         and snorkel_rules_reward_info[
                                             SnorkelNetRule.LF_filter_terms_vulnarable_protocols] > 0
                                        )
                                        or
                                        (
                                                SnorkelNetRule.LF_filter_terms_most_vulnarable_info_line in snorkel_rules_reward_info
                                                and snorkel_rules_reward_info[
                                                    SnorkelNetRule.LF_filter_terms_most_vulnarable_info_line] > 0
                                        )
                                        or
                                        (SnorkelNetRule.LF_filter_terms_vulnarable_info_line in snorkel_rules_reward_info
                                         and snorkel_rules_reward_info[
                                             SnorkelNetRule.LF_filter_terms_vulnarable_info_line] > 0
                                        )
                                )
                        ):
                            r_snorkel_humanity *= 2.0

                        r_snorkel_humanity *= cfg.humanity_coeff
                        reward_info.snorkel_humanity = r_snorkel_humanity
                        
                    except Exception as e:
                        import traceback
                        print(f"Error computing Snorkel humanity score: {e}")
                        traceback.print_exc()
                        r_snorkel_humanity = 0

            if cfg.no_diversity and not reward_info.is_same_display_seen_already:
                r_div = 0

            if cfg.no_interestingness:
                r_inter = 0

            if not cfg.use_humans_reward:
                r_humanity = 0
                r_snorkel_humanity = 0

            if cfg.use_snorkel:
                r_humanity = 0
            else:
                r_snorkel_humanity = 0

            # Include missing r_snorkel_humanity component!
            r = r_div + r_inter + r_humanity + r_snorkel_humanity
        
        reward_info.rules_reward_info = rules_reward_info
        reward_info.snorkel_rules_reward_info = snorkel_rules_reward_info
        return done, r, reward_info

    def _compute_diversity_reward(self, last_action_type, reward_info):
        last_display = self.dhist[-1]
        last_state = self.states_hisotry[-1]
        # sim_vec will contain the similarity scores of the last display and all others
        sim_vec = []

        # (1.e) Compute the diversity-based reward:
        for i, d in enumerate(self.dhist[:-1]):
            i_state = self.states_hisotry[i]
            state1 = i_state
            state2 = last_state
            if str(state1) > str(state2):
                state1, state2 = state2, state1
            display_distance_result_obj = None
            if self.STATES_DISP_DISTANCE_HISTORY is not None:
                display_distance_result_obj = self.STATES_DISP_DISTANCE_HISTORY.get(
                    (self.dataset_number, state1, state2))
            if display_distance_result_obj is None:  # if not in cache
                display_distance_result_obj = display_distance(d, last_display)
                if self.STATES_DISP_DISTANCE_HISTORY is not None:
                    self.STATES_DISP_DISTANCE_HISTORY[
                        (self.dataset_number, state1, state2)] = display_distance_result_obj
            dist = display_distance_result_obj.display_distance
            # (1.f) Punishment if the exact same display was already seen
            # (2.f) Punishment if the same data layer is seen in the same subsession after a filter action.
            # This means that the two filter action filtered the exact same rows in the same subsession
            if (dist == 0 or
                    (display_distance_result_obj.data_distance == 0 and
                     self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'filter'
                            # and len(self.obs_hist_all) - len(self.obs_hist) <= i <= len(self.obs_hist_all)
                    )
            ):
                r = -1.0 * cfg.humanity_coeff
                # print("same display is:" + str(i) + " len(dhist) is:" + str(len(self.dhist)))
                reward_info.same_display_seen_already = r
                reward_info.diversity = r
                break

            else:
                sim_vec.append(dist)
        else:
            # r = sum(sim_vec) / len(sim_vec) * 2
            r = min(sim_vec) * cfg.diversity_coeff
            reward_info.diversity = r

        return r

    def compute_interestingness_compaction_gain(self, dfs, state):
        r'''
        Let R = the number of rows in the original dataframe
        Let P = the number of rows\groups (if grouped) in the original dataframe
        Let G = the number of groups in the current dataframe (if grouped)
        Let C = the number of grouped columns in the current dataframe (if grouped)
        Let R' = the number of rows in the currrent dataframe
        If dfs[1] is not None (there is grouping involved) returns CG = CDS * DSS where
        CDS = 1/(1+e^(17*((1-1/log(10, 10+G*C))-0.5))) =  sigmoid(-(17*((1-1/log(10, 10+G*C))-0.5)))
        DSS = 1-1/(1+e^(17*((1-1/log(7, 7+G*C))-0.5))) =  1-sigmoid(-(17*((1-1/log(7, 7+G*C))-0.5)))
        If filter only is involved, returns 1-log(R')/log(R) = 1-log(R,R')
        :param dfs:
        :param state:
        :return:
        '''
        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        denominator_epsilon = 0.00001
        R = len(self.data)
        R_tag = len(df_dt)
        C = len(state["grouping"])
        if is_grouping:
            # TODO: keep track if the first is necessary because it consumes much more time
            # G = len(df_dt.index.unique())
            R_tag = len(dfs[0])
            G = len(df_dt)
            """Punishment for a single group"""
            if G == 1:
                return -1
            # return 1 / (math.log(G+10, 10) + denominator_epsilon)
            # Smaller (G*C) makes it larger
            compact_display_score = normalized_sigmoid_fkt(0.5, 17,
                                                           1 - 1 / math.log(8 + G * C + denominator_epsilon, 8))
            # Larger R_tag makes it larger
            compact_data_score = 1 - normalized_sigmoid_fkt(0.5, 17,
                                                            1 - 1 / math.log(7 + R_tag + denominator_epsilon, 7))
            return compact_display_score * compact_data_score
        return 1 - math.log(R_tag + denominator_epsilon, R)  # if filter only

    def step(self, action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None):
        """This function processes an action:
         (1) deconstruct the action to its parameters
         (2) executes the action: It computes a rolling "state" dictionary, comprising filtering,grouping and aggregations
         (3) Calculate the display vector
         (4) Update the history lists
         (5) Determine the reward

        Args:
            action (obj): Action vector

        Returns:
            obj: observation vector
            float: reward score
            bool: if done
            dict: information dict

        """

        # (1) Deconstruct the action:
        prev_action = action
        action, filter_by_field = self.action_to_vec(action, compressed, continuous_filter_term, filter_by_field)
        operator_type = self.env_prop.OPERATOR_TYPE_LOOKUP.get(action[0])
        
        # Safety check: if operator_type is None, default to 'back'
        if operator_type is None:
            print(f"Warning: Invalid action[0]={action[0]}, defaulting to 'back' action")
            operator_type = 'back'
            action[0] = 0  # Set to back action
        
        col = self.env_dataset_prop.KEYS[action[1]]
        no_history_for_back = False

        # (2) Executing an action by incrementing the state dictionary:

        if operator_type == 'back':
            # If back: pop the last element from the history and use it as the current state
            self._invalid_filter_this_step = False  # Not a filter action
            if len(self.history) > 1:
                self.obs_hist.pop()
                self.history.pop()
                new_state = self.history[-1]
            else:
                new_state = empty_env_state
                no_history_for_back = True

        elif operator_type == 'filter':
            # If filter: add the filter condition to the list of filters in the prev state
            condition = action[2]
            if filter_term is not None:
                pass
            elif not filter_by_field:
                filter_term = self.env_dataset_prop.FILTER_LIST[action[3]]
            else:
                """filter_field_list = FILTER_BY_FIELD_DICT.get(col)
                if filter_field_list and len(filter_field_list)-1 >= action[3]:
                    filter_term = filter_field_list[action[3]]
                else:
                    filter_term = '<UNK>'"""
                filter_term = self.compute_nearest_neighbor_filter_term(action, col)
            
            # If filter_term is None (no valid tokens), FORCE a back action instead!
            # This prevents <UNK> from ever being used by converting invalid filters to back actions
            if filter_term is None:
                logger.warning(f"No valid filter terms for column '{col}' - FORCING BACK ACTION instead of filter!")
                # Execute a back action instead
                if len(self.history) >= 2:
                    self.history.pop()
                    new_state = self.history[-1]
                    no_history_for_back = False
                else:
                    new_state = empty_env_state
                    no_history_for_back = True
                # Set flags
                self._invalid_filter_this_step = False
                self._forced_back_from_invalid_filter = True
                # Override operator_type for reward calculation AND update action to be a back action
                operator_type = 'back'
                # Modify action vector to reflect back action (action type 0 = back)
                action = [0, 0, 0, 0.5, 0, 0]
            else:
                # FIX: Check for invalid filter term RIGHT AFTER it's computed
                # This is where we need to catch it, not later!
                if self.is_invalid_filter_term(filter_term):
                    # col is already the column name (string), not an index
                    logger.warning(f"Invalid filter term '{filter_term}' (type: {type(filter_term).__name__}) on column '{col}'")
                    # Store flag for penalty application after reward calculation
                    self._invalid_filter_this_step = True
                    self._forced_back_from_invalid_filter = False
                    
                    # Still execute the filter (with penalty) to maintain state consistency
                    filt_tpl = FilteringTuple(field=col, term=filter_term, condition=condition)
                    new_state = self.history[-1]
                    new_state = new_state.append_filtering(filt_tpl)
                    self.history.append(new_state)
                else:
                    self._invalid_filter_this_step = False
                    self._forced_back_from_invalid_filter = False
                    
                    # Execute normal filter
                    filt_tpl = FilteringTuple(field=col, term=filter_term, condition=condition)
                    new_state = self.history[-1]
                    new_state = new_state.append_filtering(filt_tpl)
                    self.history.append(new_state)

        elif operator_type == 'group':
            # add to the grouping and aggregations lists of the prev state:
            self._invalid_filter_this_step = False  # Not a filter action
            new_state = self.history[-1]
            if col not in new_state["grouping"]:
                new_state = new_state.append_grouping(col)
            agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[action[4]], type=action[5])
            if agg_tpl not in new_state["aggregations"]:
                new_state = new_state.append_aggregations(agg_tpl)
            self.history.append(new_state)
        else:
            raise Exception("unknown operator type: " + operator_type)

        self.states_hisotry.append(new_state)

        # (3) calculate observation and update dictionaries:
        obs, disp, dfs = self.env_prop.calc_display_vector(self.data,
                                             new_state,
                                             memo=self.STATE_DF_HISTORY,
                                             dataset_number=self.dataset_number,
                                             step_number=self.step_num,
                                             states_hist=self.history,
                                             obs_hist=self.obs_hist,
                                             len_single_display_vec=self.len_single_display_vec
                                             )
        '''if dfs[1] is not None:
            display(dfs[1])
        else:
            display(dfs[0])'''

        # (4) Update the history lists:
        self.dhist.append(disp)
        self.ahist.append(action)
        self.obs_hist_all.append(obs)
        if operator_type != 'back':
            self.obs_hist.append(obs)
        # Upadte hists needed for Snorkel
        # FIX: Only append filter_term for filter actions (not group/back where it's None)
        if operator_type == 'filter':
            self.filter_terms_hist.append(filter_term)
        else:
            # For non-filter actions, append a placeholder to keep indices aligned
            self.filter_terms_hist.append(None)
        self.num_of_rows_hist.append(len(self.get_previous_df()))
        self.num_of_fdf_rows_hist.append(len(self.get_previous_fdf()))
        self.num_of_immediate_action_rows_lst.append(self.get_num_of_immediate_action_rows_after_filter())

        # (5) Get the reward
        done, reward, reward_info = self.reward(obs, dfs, new_state, no_history_for_back)
        
        # FIX: Apply STRONG penalty for filtering on invalid values (GENERIC SOLUTION)
        # This works for ANY dataset - no need to hardcode specific columns
        # Check the flag set during filter_term computation (lines 1070-1078)
        invalid_filter_used = getattr(self, '_invalid_filter_this_step', False)
        if invalid_filter_used:
            # ZERO OUT all rewards and apply strong penalty
            # This ensures invalid filters are NEVER attractive, regardless of KL/diversity
            penalty = -20.0
            reward = penalty  # REPLACE all rewards, don't just add penalty!
            if hasattr(reward_info, '__dict__'):
                reward_info.invalid_filter_penalty = penalty
                # Zero out all positive components
                reward_info.interestingness = 0
                reward_info.diversity = 0
                reward_info.humanity = 0
                reward_info.snorkel_humanity = 0
            logger.warning(f"Invalid filter term '{filter_term}' on column '{col}'. All rewards zeroed, penalty: {penalty}")
            
            # DEBUG: Track which columns trigger this (helps identify problematic columns)
            if not hasattr(self, '_invalid_filter_stats'):
                self._invalid_filter_stats = {}
            # col is already the column name (string), not an index
            self._invalid_filter_stats[col] = self._invalid_filter_stats.get(col, 0) + 1

        # validate that obs type is np.float32
        assert obs.dtype == np.float32, 'obs.dtype must be np.float32'

        if done and self._log:
            logger.info('actions:%s' % str(self.ahist))
            # logger.info('states:%s' % str(self.dhist))

        if not self.ret_df:
            dfs = None

        return obs, reward, done, {"raw_action": action,
                                   "action": self.translate_action(action, filter_by_field, filter_term),
                                   "raw_display": dfs,
                                   "reward_info": reward_info,
                                   "state": new_state,
                                   "filter_term": filter_term,
                                   }

    def action_to_vec(self, action, compressed=False, continuous_filter_term=True, filter_by_field=True):
        if self.arch is ArchName.FF_PARAM_SOFTMAX or self.arch is ArchName.FF_SOFTMAX:
            compressed = False
            action = self.param_softmax_idx_to_action(action)
        if compressed:
            # if self._log:
            # logger.info('compressed action:%s' % str(action))
            action = self.env_prop.compressed2full_range(action, continuous_filter_term)
        
        # CONDITIONAL VALIDATION: Only use if cfg.validate_actions is True (default False to match master)
        if hasattr(cfg, 'validate_actions') and cfg.validate_actions:
            # Ensure action has exactly 6 dimensions before processing
            if len(action) != 6:
                if len(action) < 6:
                    action = np.pad(action, (0, 6 - len(action)), mode='constant')
                else:
                    action = action[:6]
        
        if cfg.filter_from_list:
            continuous_filter_term = False
            filter_by_field = False
        action_filter_term = action[3]
        action = self.cont2dis(action)
        
        # CONDITIONAL VALIDATION: Only use if cfg.validate_actions is True  
        if hasattr(cfg, 'validate_actions') and cfg.validate_actions:
            # Ensure discretized action still has 6 dimensions
            if len(action) != 6:
                if len(action) < 6:
                    action.extend([0] * (6 - len(action)))
                else:
                    action = action[:6]
            
            # Ensure action values are within valid bounds
            # Action type should be 0, 1, or 2 (back, filter, group)
            action[0] = max(0, min(2, action[0]))
            
            # Column ID should be within valid range
            action[1] = max(0, min(len(self.env_dataset_prop.KEYS) - 1, action[1]))
            
            # Filter operator should be within valid range
            action[2] = max(0, min(self.env_prop.FILTER_OPS - 1, action[2]))
            
            # Filter term should be within valid range
            action[3] = max(0, min(self.env_dataset_prop.FILTER_TERMS_NO - 1, action[3]))
            
            # Aggregation column should be within valid range
            action[4] = max(0, min(self.env_dataset_prop.AGG_COLS_NO - 1, action[4]))
            
            # Aggregation function should be within valid range
            action[5] = max(0, min(self.env_prop.AGG_FUNCS_NO - 1, action[5]))
        
        if continuous_filter_term:
            action[3] = action_filter_term + 0.5
        return action, filter_by_field

    def compute_nearest_neighbor_filter_term(self, action, col):
        prev_state = self.history[-1]
        prev_state_without_group_and_agg = prev_state.reset_grouping_and_aggregations()
        if self.COL_TOKENIZATION_HISTORY is None or (
                (self.dataset_number, prev_state_without_group_and_agg,
                 col) not in self.COL_TOKENIZATION_HISTORY):
            prev_fdf = self.get_previous_fdf()
            sorted_by_freq_token_frequency_pairs, frequencies = tokenize_column(prev_fdf, col)

            # saving to cache
            # Note: we use the key prev_state_without_group_and_agg and not prev_state
            # to increase caching hit rate and due to the fact the both cases should have the same
            # column tokenization
            if self.COL_TOKENIZATION_HISTORY is not None:
                self.COL_TOKENIZATION_HISTORY[(self.dataset_number,
                                               prev_state_without_group_and_agg,
                                               col)] = (sorted_by_freq_token_frequency_pairs, frequencies)
        else:
            sorted_by_freq_token_frequency_pairs, frequencies = self.COL_TOKENIZATION_HISTORY[
                (self.dataset_number,
                 prev_state_without_group_and_agg,
                 col)]
        
        # Check if there are ANY valid tokens BEFORE calling get_nearest_neighbor_token
        # If no valid tokens, this will return <UNK>, which we want to avoid
        if not sorted_by_freq_token_frequency_pairs or len(sorted_by_freq_token_frequency_pairs) == 0:
            logger.warning(f"No valid tokens for column '{col}' - this will result in <UNK>!")
            # Return None to signal that this column can't be filtered
            # The caller should handle this by forcing a back action instead
            return None
        
        filter_term = get_nearest_neighbor_token(sorted_by_freq_token_frequency_pairs, frequencies, action[3])
        return filter_term

    def get_num_of_immediate_action_rows_after_filter(self):
        if len(self.ahist) < 2:
            return None
        cur_action = self.ahist[-1]
        prev_action = self.ahist[-2]
        cur_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[cur_action[0]]
        prev_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[prev_action[0]]

        if prev_action_type_str == 'filter':
            if cur_action_type_str == 'back':
                return None
            else:
                # Create step before filter but including group
                state_before_filter = self.states_hisotry[-3]
                if cur_action_type_str == 'group':
                    grouped_column = self.env_dataset_prop.GROUP_COLS[cur_action[1]]
                    if grouped_column not in state_before_filter["grouping"]:
                        state_before_filter = state_before_filter.append_grouping(grouped_column)
                    agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[cur_action[4]], type=cur_action[5])
                    if agg_tpl not in state_before_filter["aggregations"]:
                        state_before_filter = state_before_filter.append_aggregations(agg_tpl)
                elif cur_action_type_str == 'filter':
                    filtered_column = self.env_dataset_prop.FILTER_COLS[cur_action[1]]
                    filt_tpl = FilteringTuple(field=filtered_column, term=self.filter_terms_hist[-1],
                                              condition=cur_action[2])
                    state_before_filter = state_before_filter.append_filtering(filt_tpl)

                dfs = self.env_prop.get_state_dfs(self.data,
                                    state_before_filter,
                                    memo=self.STATE_DF_HISTORY,
                                    dataset_number=self.dataset_number,
                                    )

                df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
                return len(df_dt)

        else:
            return None


    def get_previous_fdf(self, past_steps=1):
        prev_state = self.states_hisotry[-1 * past_steps]

        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        return dfs[0]

    def get_previous_df(self, past_steps=1):
        """

        Args:
            past_steps: number of steps to go back in history, starting from 1 (!)

        Returns:

        """
        prev_state = self.states_hisotry[-1 * past_steps]
        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)

        return df_dt

    def render(self, mode='human', close=False):
        if close:
            return None
        self.ret_df = True
        # print("I will return the df in the info...")
        return None

    @classmethod
    def get_static_env(cls, max_steps):
        """
        Create a static variable env for various uses so that we won't have to create a new environment
        which is expensive
        Args:
            max_steps: Number of steps in session

        Returns:

        """
        if cls.static_env is None:
            cls.static_env = cls(max_steps=max_steps)
        else:
            cls.static_env.max_steps = max_steps
        return cls.static_env

    GET_SESSIONS_HISTS_CACHE = LRUCache(maxsize=300)

    @classmethod
    def get_sessions_hists(cls, actions_lst,
                           dataset_number,
                           compressed=False,
                           filter_by_field=True,
                           continuous_filter_term=True,
                           filter_terms_lst=None
                           ):
        """
        Returns a 2-tuple (`dhist`, `ahist`) where `dhist` is the displays histogram and `ahsit` is a histogram of actions,
        when running a session containing the actions in `actions_lst` on dataset `dataset_number`
        Args:
            actions_lst:
            dataset_number:
            compressed:
            filter_by_field:
            continuous_filter_term:
            filter_terms_lst: Note: len(filter_terms_lst) == len(actions_lst)

        Returns:

        """
        # Change architecture to gaussian
        prev_arch = cfg.arch
        cfg.arch = ArchName.FF_GAUSSIAN.value

        actions_tuple = ATENAUtils.lst_of_actions_to_tuple(actions_lst)
        if (dataset_number, actions_tuple) in cls.GET_SESSIONS_HISTS_CACHE:
            return cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)]

        if filter_terms_lst is not None:
            assert len(actions_lst) == len(filter_terms_lst)

        env = cls.get_static_env(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset(dataset_number)
        for i, a in enumerate(actions_lst):
            s_, _, done, info = env.step(a,
                                         compressed=compressed,
                                         filter_by_field=filter_by_field,
                                         continuous_filter_term=continuous_filter_term,
                                         filter_term=None if filter_terms_lst is None else filter_terms_lst[i]
                                         )  # make step in environment
            s = s_
            info_hist.append(info)
            if done:
                break
        dhist = env.dhist
        ahist = env.ahist
        cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)] = (dhist, ahist, info_hist)

        # Restore architecture
        cfg.arch = prev_arch
        return dhist, ahist, info_hist

    @classmethod
    def get_greedy_max_reward_actions_lst(cls,
                                          dataset_number,
                                          episode_length,
                                          kl_coeff,
                                          compaction_coeff,
                                          diversity_coeff,
                                          humanity_coeff,
                                          verbose=False):
        """
        Returns a 2-tuple ('actions_lst`, `total_reward`), where `actions_lst` is the list of size `epsiode_length`
        of greedy actions on dataset `dataset_nunmber` using the given coefficients for the rewards
        Args:
            dataset_number:
            episode_length:
            kl_coeff:
            compaction_coeff:
            diversity_coeff:
            humanity_coeff:
            verbose:

        Returns:

        """
        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff

        cur_env = cls(max_steps=episode_length)
        cur_env.render()
        cur_env.reset(dataset_number)
        cur_env.arch = ArchName.FF_PARAM_SOFTMAX

        actions_lst = []
        total_reward = 0
        for step in range(episode_length):
            max_reward = -math.inf
            max_action = None
            max_env = None

            for act_idx in cur_env.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION.keys():
                next_env = deepcopy(cur_env)
                obs, reward, done, info = next_env.step(act_idx)
                action_vec = info["raw_action"]
                action_vec[3] -= 0.5
                if reward > max_reward:
                    max_reward = reward
                    max_action = action_vec
                    max_env = next_env
            if verbose:
                print(max_reward)
            cur_env = max_env
            actions_lst.append(max_action)
            total_reward += max_reward
        if verbose:
            print(actions_lst)
            print(total_reward)
        return actions_lst, total_reward

    @classmethod
    def debug_actions(cls, actions_lst, dataset_number=None, compressed=False, filter_by_field=False,
                      continuous_filter_term=False, displays=False,
                      kl_coeff=cfg.kl_coeff,
                      compaction_coeff=cfg.compaction_coeff,
                      diversity_coeff=cfg.diversity_coeff,
                      humanity_coeff=cfg.humanity_coeff,
                      ):

        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff
        cfg.analysis_mode = True

        env = cls(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset()
        for i in range(1):
            # cls.reset_caches()
            if isinstance(env, ATENAEnvCont):
                s = env.reset(dataset_number)
            elif isinstance(env, gym.wrappers.Monitor):
                s = env.env.env.reset(dataset_number)
            else:
                s = env.env.reset(dataset_number)
            r_sum = 0
            for a in actions_lst:
                print(a)
                s_, r, done, info = env.step(a,
                                             compressed=compressed,
                                             filter_by_field=filter_by_field,
                                             continuous_filter_term=continuous_filter_term)  # make step in environment
                # Debug prints disabled for clean training output
                # print(str(info["action"]))
                # print("reward:" + str(r))
                # print(str(info["reward_info"]))
                # print()
                info_hist.append((info, r))
                s = s_
                r_sum += r
                # print("")
                if displays:
                    f_df, a_df = info["raw_display"]
                    if a_df is not None:
                        pass  # print(a_df)
                    else:
                        pass  # print(f_df)
                # print("---------------------------------------------------")
                if done:
                    break
        # Debug: Total reward can be uncommented if needed
        # print(f"Total reward: {r_sum}")
        return info_hist, r_sum


class StepReward(object):
    """class that contains the reward_info for a single step"""

    def __init__(self,
                 empty_display=0,
                 empty_groupings=0,
                 same_display_seen_already=0,
                 back=0,
                 diversity=0,
                 interestingness=0,
                 kl_distance=0,
                 compaction_gain=0,
                 humanity=0,
                 snorkel_humanity=0,
                 ):
        self.empty_display = empty_display
        self.empty_groupings = empty_groupings
        self.same_display_seen_already = same_display_seen_already
        self.back = back
        self.diversity = diversity
        self.interestingness = interestingness
        self.kl_distance = kl_distance
        self.compaction_gain = compaction_gain
        self.humanity = humanity
        self.snorkel_humanity = snorkel_humanity
        self.rules_reward_info = None
        self.snorkel_rules_reward_info = None

        self.is_back = False
        self.is_data_driven = False

    @property
    def is_same_display_seen_already(self):
        return self.same_display_seen_already < 0

    def items(self):
        """return (reward_type_str, reward_type_val) pairs"""
        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('rules_reward_info')
        result.pop('snorkel_rules_reward_info')
        result.pop('is_data_driven')
        return result.items()

    def values(self):
        """return reward_type_vals"""
        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('rules_reward_info')
        result.pop('snorkel_rules_reward_info')
        result.pop('is_data_driven')
        return result.values()

    def __repr__(self):
        return self.items().__repr__()


if __name__ == '__main__':
    '''actions_lst = [
    [1, 4, 3, 9, 9, 2],
    [2, 1, 3, 23, 5, 0],
    [1, 4, 5, 19, 5, 2],
    [1, 6, 1, 15, 7, 0],'''
    actions_lst = [
        [2, 1, 1, 0.3221167325973511, 0, 0],  # 1
        [2, 6, 4, 0.6, 0, 0],  # 2
        [0, 2, 8, 0.21705351769924164, 0, 0],  # 3
        [2, 2, 8, -0.49803635857105255, 0, 0],  # 4
        [2, 6, 7, -0.46546758365631, 0, 0],  # 5
        [1, 6, 4, 0.38086163997650146, 0, 0],  # 6
        [0, 3, 1, -0.40, 0, 0],  # 7
        [0, 6, 8, 0.444186806678772, 0, 0],  # 9
        [0, 6, 8, 0.444186806678772, 0, 0],  # 8
        [2, 5, 4, 0.6415030360221863, 0, 0],  # 10
        [2, 6, 4, 0.5431699156761169, 0, 0],  # 11
        [2, 2, 4, 0.8115951418876648, 0, 0],  # 12
    ]
    actions_lst = [np.array(act) for act in actions_lst]
    ATENAEnvCont.debug_actions(actions_lst, 3, compressed=False, filter_by_field=True, continuous_filter_term=True,
                               displays=True,
                               kl_coeff=2.5,
                               compaction_coeff=2.9,
                               diversity_coeff=6.0,
                               humanity_coeff=4.8,
                               # kl_coeff=1,
                               # compaction_coeff=1,
                               # diversity_coeff=1,
                               # humanity_coeff=1,
                               )
    # ATENAEnvCont.get_greedy_max_reward_actions_lst(dataset_number=1,
    #                                                episode_length=12,
    #                                                kl_coeff=3.2,
    #                                                compaction_coeff=2.0,
    #                                                diversity_coeff=6.5,
    #                                                humanity_coeff=4.5,
    #                                                verbose=True
    #                                                )

    actions_lst = [
        [2, 6, 2, 0.3221167325973511, 0, 0],  # 1
        [2, 5, 1, -0.445, 0, 0],  # 2
    ]

    actions_lst2 = [
        [2, 5, 1, -0.445, 0, 0],  # 1
        [2, 6, 2, 0.3221167325973511, 0, 0],  # 2
    ]

    # actions_lst = [np.array(act) for act in actions_lst]
class HumanRulesReward(object):
    """CRITICAL MISSING CLASS: class that contains the reward_info of human-based rules for a single step
    
    Exactly matching master implementation
    """

    def __init__(self, rules_values_dict=None):
        if rules_values_dict is None:
            self.triggered_rules = dict()
        else:
            self.triggered_rules = rules_values_dict

    def __setitem__(self, rule, value):
        self.triggered_rules[rule] = value

    def __getitem__(self, key):
        return self.triggered_rules[key]

    def __contains__(self, key):
        return key in self.triggered_rules

    def reset(self):
        self.triggered_rules = dict()

    def items(self):
        return self.triggered_rules.items()

    def __repr__(self):
        return self.items().__repr__()


    # actions_lst2 = [np.array(act) for act in actions_lst2]
    # dhist1, ahist1 = ATENAEnvCont.get_sessions_hists(actions_lst, dataset_number=0)
    # dhist2, ahist2 = ATENAEnvCont.get_sessions_hists(actions_lst, dataset_number=0)
    # print(dhist1[-1])
    # print(dhist2[-1])
    # print(str(dhist1[-1]) == str(dhist2[-1]))
    # print(len(dhist1))
    # d = dict()
    # d[str(dhist1[1])] = 1
    # print(d)