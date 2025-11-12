import sys
import os
import os.path as path
import math
import operator
from enum import Enum
from functools import lru_cache

from gym_atena.data_schemas.netowrking.columns_data import (KEYS, KEYS_ANALYST_STR, FILTER_COLS, GROUP_COLS,
    NUMERIC_KEYS, AGG_KEYS, AGG_KEYS_ANALYST_STR, FILTER_LIST, FILTER_BY_FIELD_DICT, DONT_FILTER_FIELDS)
from gym_atena.reactida.utils.utilities import Repository
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.envs.atena_snorkel.atena_snorkel_networking import SnorkelNetDataObj
from gym_atena.envs.env_properties import EnvDatasetProp, BasicEnvProp
from gym_atena.envs.atena_snorkel.snorkel_gen_model import SnorkelGenModel
from gym_atena.lib.helpers import (
    INT_OPERATOR_MAP_REACT_TO_ATENA,
    INVERSE_AGG_MAP_ATENA,
    OPERATOR_TYPE_LOOKUP,
    INT_OPERATOR_MAP_ATENA,
    AGG_MAP_ATENA,
    normalized_sigmoid_fkt)


def create_datasets_repository():
    par_dir = path.dirname(path.dirname(__file__))
    repo_path = path.join(par_dir, 'reactida/session_repositories')
    act_path = path.join(repo_path, 'actions.tsv')
    disp_path = path.join(repo_path, 'displays.tsv')
    datasets_path = path.join(par_dir, 'reactida/raw_datasets_with_pkt_nums')
    # Solving printing bug
    old_stdout = sys.stdout
    # TEMP: Don't suppress stdout so we can see preprocessing messages!
    # sys.stdout = open(os.devnull, "w")
    datasets_repository = Repository(act_path, disp_path, datasets_path)
    # sys.stdout = old_stdout
    return datasets_repository


def create_networking_env_properties():
    env_dataset_prop = EnvDatasetProp(
        create_datasets_repository(),
        KEYS,
        KEYS_ANALYST_STR,
        FILTER_COLS,
        GROUP_COLS,
        AGG_KEYS,
        AGG_KEYS_ANALYST_STR,
        NUMERIC_KEYS,
        _compute_rule_based_humanity_score,
        FILTER_LIST,
        FILTER_BY_FIELD_DICT,
        DONT_FILTER_FIELDS,
        snorkel_data_obj=SnorkelNetDataObj,
        snorkel_gen_model=SnorkelGenModel,
        snorkel_training_path="snorkel_dataset.jsonl",
    )
    return BasicEnvProp(OPERATOR_TYPE_LOOKUP,
                        INT_OPERATOR_MAP_ATENA,
                        AGG_MAP_ATENA,
                        env_dataset_prop,
                        )


def convert_to_action_vector(action, action_params):
    if action == "back":
        return [0]*6
    if action == "filter":
        col = KEYS.index(action_params["field"])
        condition = INT_OPERATOR_MAP_REACT_TO_ATENA[action_params["condition"]]
        term = FILTER_LIST.index(action_params["term"])
        return [1] + [col] + [condition] + [term] + [0, 0]
    if action == "group":
        col = KEYS.index(action_params["field"])
        if "field" in action_params["aggregations"]:
            agg_col = AGG_KEYS.index(action_params["aggregations"]["field"])
        else:
            agg_col = AGG_KEYS.index('packet_number')  # 'packet_number' col is default aggregation
        if "type" in action_params["aggregations"]:
            agg_func = INVERSE_AGG_MAP_ATENA[(action_params["aggregations"]["type"])]
        else:
            agg_func = INVERSE_AGG_MAP_ATENA['count']  # count is default aggregation
        return [2] + [col] + [0, 0] + [agg_col] + [agg_func]


def compute_normalized_readability_gain(df, prev_prev_df, num_of_grouped_cols):
    """

    Args:
        df:
        prev_prev_df:
        num_of_grouped_cols: 1 if not grouped, else number of grouped columns in df and in prev_prev_df (this is
        the same number, because this method is called after a filter action

    Returns:

    """
    num_of_grouped_cols = 1
    denominator_epsilon = 0.00001
    disp_rows_prev = len(df)
    disp_rows_prev_prev = len(prev_prev_df)

    # how compact is the resulted display
    compact_display_score = normalized_sigmoid_fkt(0.5, 17,
                                                   1 - 1 / math.log(9 + disp_rows_prev * num_of_grouped_cols, 9))
    if disp_rows_prev == 1:
        normalized_readability_gain = -1
    else:
        prev_readability = normalized_sigmoid_fkt(
            0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev * num_of_grouped_cols + denominator_epsilon, 9))
        prev_prev_readability = normalized_sigmoid_fkt(
            0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev_prev * num_of_grouped_cols + denominator_epsilon, 9))
        assert prev_readability >= prev_prev_readability

        # how compact the resulted display of the filter action relative to the display before it.
        readability_gain = 1 - prev_prev_readability / prev_readability

        # transforming the readability gain to be in range [-0.5, 0.5]
        normalized_readability_gain = -0.5 + 1.0 * normalized_sigmoid_fkt(
            0.6, 11, 1 - readability_gain * compact_display_score)
        # making negative normalized_readability_gain in the range [-2.0, 0) instead of
        # of [-0.5, 0) to 'cancel' potential gain of the filter action
        if normalized_readability_gain < 0:
            normalized_readability_gain *= 4
    return normalized_readability_gain


def _compute_rule_based_humanity_score(self, dfs, state, rules_reward_info, done):
    humanity_scores_lst = []

    def trigger_rule(rule, value, remove_all=False):
        """

        Args:
            rule:
            value:
            remove_all: Whether or not to remove all elements from  `humanity_scores_lst` and `rules_reward_info`

        Returns:

        """
        if remove_all:
            rules_reward_info.reset()
            humanity_scores_lst.clear()
        rules_reward_info[rule] = value
        humanity_scores_lst.append(value)
        return value

    last_action = self.ahist[-1]
    last_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[last_action[0]]

    # back
    if last_action_type_str == 'back':
        # SAFETY CHECK: Ensure we have at least 2 actions before accessing ahist[-2]
        if len(self.ahist) < 2:
            # First action is back (no previous action to evaluate)
            # Return neutral score
            return 0.0
        
        prev_action = self.ahist[-2]
        prev_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[prev_action[0]]

        """
        Back if the previous action was a filter operation that didn't improve readability is NON-HUMANE
        Exception: if the filter action resulted in a single row / group, then it is NON-HUMANE in spite
        of the readability gain
        """
        if prev_action_type_str == 'filter':
            prev_df = self.get_previous_df(past_steps=2)
            prev_prev_df = self.get_previous_df(past_steps=3)

            num_of_grouped_columns = max(1, len(state["grouping"]))
            normalized_readability_gain = compute_normalized_readability_gain(
                prev_df,
                prev_prev_df,
                num_of_grouped_columns)
            return trigger_rule(NetHumanRule.back_after_filter_readability_gain,
                                normalized_readability_gain)

        """Back when the current state consists of x non-back operations is MORE-HUMANE
        then when it consists of y non-back operations for x > y.
        Exception: if the previous action was back then it is MOSTLY-HUMANE"""
        if prev_action_type_str == 'back':
            back_after_back_gain = 0.2
            return trigger_rule(NetHumanRule.back_after_back,
                                back_after_back_gain)
        return 0

    # filter
    elif last_action_type_str == 'filter':
        cur_df = self.get_previous_df()
        cur_fdf = dfs[0]
        filtered_column = self.env_dataset_prop.GROUP_COLS[last_action[1]]

        prev_df = self.get_previous_df(past_steps=2)

        """Filter on a dataset containing small amount of rows in the data layer is NON-HUMANE"""
        prev_fdf = self.get_previous_fdf(past_steps=2)
        if len(prev_fdf) < 40:
            too_low_rows_to_filter_punishment = -1.0
            return trigger_rule(NetHumanRule.filter_small_number_of_rows, too_low_rows_to_filter_punishment)

        """Filter that results in a single row is NON-HUMANE"""
        if len(cur_fdf) == 1:
            filter_results_in_single_row_punishment = -1.0
            return trigger_rule(NetHumanRule.filter_results_in_single_row, filter_results_in_single_row_punishment)

        """
        It is NON-HUMANE to filter using the 'equals' or 'contains' operator more the once on the same column
        in the same subsession or with a combination of 'not equal' and then 'equals' or 'contains'
        """
        filtered_column_operators = [
            self.env_prop.INT_OPERATOR_MAP_ATENA[filtering_tuple.condition] for filtering_tuple
            in state.filtering if filtering_tuple.field == filtered_column
        ]
        if len(filtered_column_operators) > 1 and len(filtered_column_operators) != filtered_column_operators.count(
                operator.ne):
            filter_on_the_same_column_in_subsession_punishment = -1.0
            return trigger_rule(NetHumanRule.filter_on_the_same_column_in_subsession,
                                filter_on_the_same_column_in_subsession_punishment)

        """Stacking more than 2 Filters is NON-HUMANE"""
        filtering_length = len(state.filtering)
        if filtering_length > 2:
            too_many_filters_punishment = -1.0
            return trigger_rule(NetHumanRule.stacking_more_than_two_filters, too_many_filters_punishment)

        """
        Filter from a column that is not currently displayed is NON-HUMANE
        Exception: since info_line grouping is inhumane, we allow filtering from this column without
        displaying it
        """
        if state.grouping and filtered_column not in state.grouping and filtered_column not in {'info_line'}:
            filter_from_undisplayed_column_punishment = -1.0
            return trigger_rule(NetHumanRule.filter_from_undisplayed_column,
                                filter_from_undisplayed_column_punishment)
        else:
            filter_from_displayed_column_gain = 0.7
            trigger_rule(NetHumanRule.filter_from_displayed_column,
                         filter_from_displayed_column_gain)

        """
        Filter operation as last action that didn't improve readability is NON-HUMANE
        Exception: if the filter action resulted in a single row / group, then it is NON-HUMANE inspite
        of the readability gain
        """
        if done:
            num_of_grouped_columns = max(1, len(state["grouping"]))
            normalized_readability_gain = compute_normalized_readability_gain(
                self.get_previous_df(past_steps=1),
                self.get_previous_df(past_steps=2),
                num_of_grouped_columns
            )
            trigger_rule(NetHumanRule.end_episode_filter_readability_gain,
                         normalized_readability_gain)

        """Group as first action is MORE-HUMANE then Filter"""
        if self.step_num == 1:
            filter_as_first_action_punishment = -1.0
            trigger_rule(NetHumanRule.filter_as_first_action, filter_as_first_action_punishment)

        """Filter on 'highest_layer', 'ip_dst', 'ip_src', 'info_line',
        'tcp_dstport', 'tcp_srcport', 'tcp_stream' is HUMANE"""
        most_humane_filter_columns = {'highest_layer', 'info_line'}
        humane_filter_columns = {'ip_dst', 'ip_src',
                                 'tcp_dstport', 'tcp_srcport'}

        """Filter on 'eth_dst', 'eth_src', 'length' is MOSTLY-NON-HUMANE (note that no human actually did that)"""
        neutral_filter_columns = {'length', 'eth_dst', 'eth_src', 'tcp_stream'}

        """Filter on 'packet_number', 'sniff_timestamp' is NON-HUMANE"""
        non_humane_filter_columns = {'packet_number', 'sniff_timestamp'}

        if filtered_column in most_humane_filter_columns:
            most_humane_columns_filter_gain = 0.8
            trigger_rule(NetHumanRule.humane_columns_filter, most_humane_columns_filter_gain)
        elif filtered_column in humane_filter_columns:
            humane_columns_filter_gain = 0.1
            trigger_rule(NetHumanRule.humane_columns_filter, humane_columns_filter_gain)
        elif filtered_column in neutral_filter_columns:
            neutral_columns_filter_punishment = -0.5
            trigger_rule(NetHumanRule.neutral_columns_filter, neutral_columns_filter_punishment)
        elif filtered_column in non_humane_filter_columns:
            inhumane_columns_filter_punishment = -1.0
            return trigger_rule(NetHumanRule.inhumane_columns_filter, inhumane_columns_filter_punishment,
                                remove_all=True)

        """Filter using  '<built-in function ne>' or  '<built-in function eq>' on
        the 'info-line' column is NON-HUMANE"""
        from gym_atena.lib.helpers import INT_OPERATOR_MAP_ATENA_STR
        filter_operator = INT_OPERATOR_MAP_ATENA_STR[last_action[2]]
        if filtered_column == 'info_line' and filter_operator in {"eq", "ne"}:
            info_line_bad_filter_operators = -1.0
            return trigger_rule(NetHumanRule.info_line_bad_filter_operators, info_line_bad_filter_operators,
                                remove_all=True)
        elif filtered_column == 'info_line' and filter_operator in {"contains"}:
            info_line_good_filter_operators = 0.5
            trigger_rule(NetHumanRule.info_line_good_filter_operators, info_line_good_filter_operators)
        elif filter_operator == "ne":
            """
            Using the filter operator NOT EQUAL should be discouraged
            """
            not_equal_filter_operator_punishment = -0.2
            trigger_rule(NetHumanRule.using_not_equal_filter_operator, not_equal_filter_operator_punishment)

        """Filter that does not change the number of rows / groups is MOSTLY-NON-HUMANE"""
        if len(prev_df) == len(cur_df):
            filter_num_of_rows_unchanged = -1.0
            trigger_rule(NetHumanRule.filter_num_of_rows_unchanged, filter_num_of_rows_unchanged)
        else:  # this is mainly relevant for using contains in multi-token cells (info_line in our case)
            filter_num_of_rows_changed = 0.1
            trigger_rule(NetHumanRule.filter_num_of_rows_changed, filter_num_of_rows_changed)

        filter_term = self.states_hisotry[-1]["filtering"][-1].term
        if filter_term in set(self.env_dataset_prop.FILTER_BY_FIELD_DICT[filtered_column]):
            """Filter on a token that do exist in the human sessions is HUMANE"""
            filter_term_appears_in_human_session_gain = 0.9
            trigger_rule(NetHumanRule.filter_term_appears_in_human_session,
                         filter_term_appears_in_human_session_gain)
        else:
            """Filter on a token that do not exist in the human sessions is MOSTLY-NON-HUMANE"""
            filter_term_does_not_appear_in_human_session_punishment = -0.4
            trigger_rule(NetHumanRule.filter_term_doesnt_appear_in_human_session,
                         filter_term_does_not_appear_in_human_session_punishment)

    # group
    elif last_action_type_str == 'group':
        cur_df = dfs[1]
        grouped_column = self.env_dataset_prop.GROUP_COLS[last_action[1]]

        """Group that results in one group is NON-HUMANE"""
        if len(cur_df) <= 1:
            group_result_in_single_group_punishment = -1.0
            return trigger_rule(NetHumanRule.group_results_in_single_group,
                                group_result_in_single_group_punishment)

        """Group as first action is MORE-HUMANE then Filter"""
        if self.step_num == 1:
            group_as_first_action_gain = 1.0
            trigger_rule(NetHumanRule.group_as_first_action, group_as_first_action_gain)

        """
        Using the filter operator EQUAL or CONTAINS on some column and then GROUP on the same column
        in the same subsession is INHUMANE
        """
        from gym_atena.lib.helpers import INT_OPERATOR_MAP_ATENA_STR
        grouped_column_is_filtered_using_equal_or_contains = any(
            [True for filtering_tuple
             in state.filtering if filtering_tuple.field == grouped_column and (
                     INT_OPERATOR_MAP_ATENA_STR[filtering_tuple.condition] in {"eq", "contains"}
             )
             ]
        )
        if grouped_column_is_filtered_using_equal_or_contains:
            group_on_filtered_column_in_subsession_punishment = -1.0
            return trigger_rule(NetHumanRule.group_on_filtered_column_in_subsession,
                                group_on_filtered_column_in_subsession_punishment)

        """Group on 'eth_src', 'highest_layer', 'ip_dst', 'ip_src',
        'tcp_dstport', 'tcp_srcport', 'tcp_stream' is HUMANE"""
        most_humane_grouped_columns = {'highest_layer', 'ip_dst', 'ip_src'}
        humane_grouped_columns2 = {'eth_src', }
        humane_grouped_columns = {'tcp_dstport', 'tcp_srcport', 'tcp_stream'}

        """Group on  'eth_dst', 'length' is MOSTLY-NON-HUMANE"""
        neutral_grouped_columns = {'length', 'eth_dst'}

        """Group on 'packet_number', 'info_line', 'sniff_timestamp' is NON-HUMANE"""
        non_humane_grouped_columns = {'packet_number', 'info_line', 'sniff_timestamp'}

        if grouped_column in most_humane_grouped_columns:
            most_humane_columns_group_gain = 0.4
            trigger_rule(NetHumanRule.humane_columns_group, most_humane_columns_group_gain)
        if grouped_column in humane_grouped_columns2:
            humane_columns_group_gain2 = 0.25
            trigger_rule(NetHumanRule.humane_columns_group, humane_columns_group_gain2)
        if grouped_column in humane_grouped_columns:
            humane_columns_group_gain = 0.1
            trigger_rule(NetHumanRule.humane_columns_group, humane_columns_group_gain)
        elif grouped_column in neutral_grouped_columns:
            neutral_columns_group_punishment = -0.5
            trigger_rule(NetHumanRule.neutral_columns_group, neutral_columns_group_punishment)
        elif grouped_column in non_humane_grouped_columns:
            inhumane_columns_group_punishment = -1.0
            return trigger_rule(NetHumanRule.inhumane_columns_group, inhumane_columns_group_punishment)

        """
        GROUP on column highest_layer on a filtered display is INHUMANE
        """
        if grouped_column == 'highest_layer':
            if state.filtering:
                highest_layer_group_on_filtered_display_punishment = -0.5
                trigger_rule(NetHumanRule.highest_layer_group_on_filtered_display,
                             highest_layer_group_on_filtered_display_punishment
                             )
            else:
                highest_layer_group_on_non_filtered_display_incentive = 0.5
                trigger_rule(NetHumanRule.highest_layer_group_on_filtered_display,
                             highest_layer_group_on_non_filtered_display_incentive
                             )
        """
        GROUP on eth_src that applied on a filtered display is INHUMANE
        """
        if grouped_column == 'eth_src':
            if state.filtering:
                eth_src_group_on_filtered_display_punishment = -0.5
                trigger_rule(NetHumanRule.group_eth_src_on_filtered_display,
                             eth_src_group_on_filtered_display_punishment
                             )
            else:
                eth_src_group_on_non_filtered_display_incentive = 0.5
                trigger_rule(NetHumanRule.group_eth_src_on_filtered_display,
                             eth_src_group_on_non_filtered_display_incentive
                             )

        """Group on a column that is already grouped is NON-HUMANE"""
        prev_state = self.states_hisotry[-2]
        if grouped_column in prev_state.grouping:
            column_already_grouped_punishment = -1.0
            return trigger_rule(NetHumanRule.column_already_grouped, column_already_grouped_punishment,
                                remove_all=True)

        """If the first GROUP in a display is applied on column length then it is INHUMANE"""
        if grouped_column == 'length' and not prev_state.grouping:
            first_group_length_punishment = -1.0
            return trigger_rule(NetHumanRule.first_group_length, first_group_length_punishment,
                                remove_all=True)

        """GROUP by ip_dst immediately after GROUP by ip_src should be encouraged if no filter is already applied"""
        if grouped_column == 'ip_dst' and self.step_num > 1 and not state.filtering:
            prev_action = self.ahist[-2]
            prev_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[prev_action[0]]
            prev_column = self.env_dataset_prop.GROUP_COLS[prev_action[1]]
            if prev_action_type_str == 'group' and prev_column == 'ip_src':
                group_ip_dst_after_group_ip_src_incentive = 1.4
                trigger_rule(NetHumanRule.group_ip_dst_after_group_ip_src,
                             group_ip_dst_after_group_ip_src_incentive
                             )

        """Group that results in no change in the number of groups is MOSTLY-NON-HUMANE"""
        prev_df = self.get_previous_df(past_steps=2)
        if len(prev_df) == len(cur_df):
            group_num_of_groups_unchanged = -0.45
            trigger_rule(NetHumanRule.group_num_of_groups_unchanged, group_num_of_groups_unchanged)
        else:
            group_num_of_groups_changed = 0.1
            trigger_rule(NetHumanRule.group_num_of_groups_changed, group_num_of_groups_changed)

        """Stacking more than 5 Group-by actions is MOSTLY-NON-HUMANE"""
        grouping_length = len(state.grouping)
        if grouping_length >= 5:
            if grouping_length == 5:
                stacking_five_groups_punishment = -0.6
                trigger_rule(NetHumanRule.stacking_five_groups, stacking_five_groups_punishment)
            else:  # if grouping_length > 5
                stacking_more_than_six_groups_punishment = -0.95
                trigger_rule(NetHumanRule.stacking_more_than_six_groups,
                             stacking_more_than_six_groups_punishment)

    return sum(humanity_scores_lst) / len(humanity_scores_lst)


class NetHumanRule(Enum):
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
