"""
A bunch of labeling functions for the Wide12 flight delays datasets
"""

import enum
import functools

import numpy as np
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.data_schemas.wide12_flights.columns_data import (
    FILTER_BY_FIELD_DICT
)
from gym_atena.envs.atena_snorkel.atena_snorkel_wide12_flights import (
    SnorkelWide12FlightsDataObj
)
from gym_atena.envs.atena_snorkel.atena_snorkel_data import ReadabilityGainType


def remove_digit_suffix_from_column(column):
    """
    Returns the column name without a digit suffix (e.g. '_2', '_3' etc.))
    """
    if column[-1].isdigit():
        return column[:-2]
    return column


def dec_filter_term_appears_once_in_session(func):
    """
    Decorator that prevents from the same filter term to be rewarded twice
    Args:
        func:

    Returns:

    """
    # We say that "wrapper", is wrapping "func"
    # and the magic begins
    @functools.wraps(func)
    def wrapper(snorkel_wide12_flights_data_obj):
        func_result = func(snorkel_wide12_flights_data_obj)
        if func_result == 1:
            filter_term_per_column_counts = snorkel_wide12_flights_data_obj.get_all_session_filtered_col_and_terms_counts()
            if filter_term_per_column_counts[(snorkel_wide12_flights_data_obj.last_action_obj.filtered_col,
                                              snorkel_wide12_flights_data_obj.last_action_obj.filter_term)] > 1:
                func_result = -1
        return func_result

    return wrapper


def LF_empty_display(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.num_of_rows_lst[-1] == 0:
        return -1
    return 0


def LF_most_humane_columns_group(snorkel_wide12_flights_data_obj):
    most_humane_grouped_columns = {
        'airline', 'departure_delay', 'delay_reason',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col
        if remove_digit_suffix_from_column(grouped_col) in most_humane_grouped_columns:
            return 1

    return 0


def LF_humane_columns_group2(snorkel_wide12_flights_data_obj):
    humane_grouped_columns2 = {
        'origin_airport',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col
        if remove_digit_suffix_from_column(grouped_col) in humane_grouped_columns2:
            return 1

    return 0


def LF_humane_columns_group(snorkel_wide12_flights_data_obj):
    humane_grouped_columns = {
        'scheduled_trip_time', 'scheduled_departure', 'day_of_week',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col
        if remove_digit_suffix_from_column(grouped_col) in humane_grouped_columns:
            return 1
    return 0


def LF_neutral_columns_group(snorkel_wide12_flights_data_obj):
    neutral_grouped_columns = {
        'day_of_year', 'destination_airport', 'flight_number',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col
        if remove_digit_suffix_from_column(grouped_col) in neutral_grouped_columns:
            return -1
    return 0


def LF_inhumane_columns_group(snorkel_wide12_flights_data_obj):
    non_humane_grouped_columns = {'flight_id', 'scheduled_arrival',
                                  'taxi_out', 'taxi_in', 'wheels_off',
                                  'wheels_on', 'year', 'month', 'elapsed_time', 'air_time', 'distance',
                                  'air_system_delay', 'security_delay', 'airline_delay',
                                  'late_aircraft_delay', 'weather_delay'
                                  }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col
        if remove_digit_suffix_from_column(grouped_col) in non_humane_grouped_columns:
            return -1
    return 0


def LF_column_already_grouped(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        session_action_objs = snorkel_wide12_flights_data_obj.subsession_action_objs_lst
        last_action = session_action_objs[-1]
        other_session_actions = session_action_objs[:-1]
        for other_session_action in other_session_actions:
            if other_session_action.act_type == 'group' and other_session_action.grouped_col == last_action.grouped_col:
                return -1
    return 0


def LF_group_num_of_groups_unchanged(snorkel_wide12_flights_data_obj):
    """
    Returns -1 if the number of groups is not changed
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        if snorkel_wide12_flights_data_obj.num_of_rows_lst[-1] == snorkel_wide12_flights_data_obj.num_of_rows_lst[-2]:
            return -1
    return 0


def LF_group_num_of_groups_changed(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        if not (snorkel_wide12_flights_data_obj.num_of_rows_lst[-1] == snorkel_wide12_flights_data_obj.num_of_rows_lst[-2]):
            return 1
    return 0


def LF_stacking_five_groups(snorkel_wide12_flights_data_obj):
    # if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
    grouped_cols = snorkel_wide12_flights_data_obj.get_subsession_grouped_cols()

    if len(grouped_cols) == 5:
        return -1
    return 0


def LF_stacking_more_than_five_groups(snorkel_wide12_flights_data_obj):
    # if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
    grouped_cols = snorkel_wide12_flights_data_obj.get_subsession_grouped_cols()

    if len(grouped_cols) >= 6:
        return -1
    return 0


def LF_filter_term_appears_in_human_session(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if (snorkel_wide12_flights_data_obj.last_action_obj.filter_term in
                FILTER_BY_FIELD_DICT[remove_digit_suffix_from_column(snorkel_wide12_flights_data_obj.last_action_obj.filtered_col)]):
            return 1
    return 0


def LF_filter_term_not_appear_in_human_session(snorkel_wide12_flights_data_obj):
    """
    This function returns -1 if the filter term does not appear in the human sessions
    if filtered column is NOT 'ip_src', 'ip_dst' or 'eth_src'
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if snorkel_wide12_flights_data_obj.last_action_obj.filtered_col not in {'ip_src', 'ip_dst', 'eth_src'} and not (
                snorkel_wide12_flights_data_obj.last_action_obj.filter_term in
                FILTER_BY_FIELD_DICT[
                    remove_digit_suffix_from_column(snorkel_wide12_flights_data_obj.last_action_obj.filtered_col)]):
            return -1
    return 0


def LF_most_humane_columns_filter(snorkel_wide12_flights_data_obj):
    most_humane_filter_columns = {
        'departure_delay',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        if remove_digit_suffix_from_column(filtered_col) in most_humane_filter_columns:
            return 1
    return 0


def LF_most_humane_columns_filter2(snorkel_wide12_flights_data_obj):
    most_humane_filter_columns = {}

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        if remove_digit_suffix_from_column(filtered_col) in most_humane_filter_columns:
            return 1
    return 0


def LF_humane_columns_filter(snorkel_wide12_flights_data_obj):
    humane_filter_columns = {
        'airline', 'origin_airport', 'delay_reason', 'destination_airport',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        if remove_digit_suffix_from_column(filtered_col) in humane_filter_columns:
            return 1
    return 0


def LF_neutral_columns_filter(snorkel_wide12_flights_data_obj):
    neutral_filter_columns = {
        'scheduled_trip_time',
    }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        if remove_digit_suffix_from_column(filtered_col) in neutral_filter_columns:
            return -1
    return 0


def LF_inhumane_columns_filter(snorkel_wide12_flights_data_obj):
    non_humane_filter_columns = {'flight_id', 'flight_number', 'day_of_year', 'scheduled_arrival', 'day_of_week',
                                 'scheduled_departure',
                                 'taxi_out', 'taxi_in', 'wheels_off',
                                 'wheels_on', 'year', 'month', 'elapsed_time', 'air_time', 'distance',
                                 'air_system_delay', 'security_delay', 'airline_delay',
                                 'late_aircraft_delay', 'weather_delay'
                                 }

    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        if remove_digit_suffix_from_column(filtered_col) in non_humane_filter_columns:
            return -1
    return 0


def LF_filter_num_of_groups_unchanged(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if snorkel_wide12_flights_data_obj.num_of_rows_lst[-1] == snorkel_wide12_flights_data_obj.num_of_rows_lst[-2]:
            return -1
        # else:
        #     return 1
    return 0


def LF_filter_num_of_rows_unchanged(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[-1] == snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[-2]:
            return -1
    return 0


def LF_back_with_no_history(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'back':
        if len(snorkel_wide12_flights_data_obj.prev_subsession_action_objs_lst) == 0:
            return -1
    return 0


def LF_back_after_back(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'back':
        if snorkel_wide12_flights_data_obj.prev_action_obj and snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'back':
            return 1
    return 0


def LF_group_as_first_action(snorkel_wide12_flights_data_obj):
    if len(snorkel_wide12_flights_data_obj) == 1:
        if snorkel_wide12_flights_data_obj.last_action_obj.act_type != 'group':
            return -1
    return 0


def LF_stacking_more_than_two_filters(snorkel_wide12_flights_data_obj):
    """
    Note: This will return -1 even if the current action is group and there
    are currently more than 2 filters applied
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    session_action_objs = snorkel_wide12_flights_data_obj.subsession_action_objs_lst
    filter_actions_num = 0

    for action_obj in session_action_objs:
        if action_obj.act_type == 'filter':
            filter_actions_num += 1

    if filter_actions_num >= 3:
        return -1
    return 0


def LF_filter_from_undisplayed_column(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        displayd_cols = snorkel_wide12_flights_data_obj.get_displayed_cols()
        if snorkel_wide12_flights_data_obj.last_action_obj.filtered_col != 'info_line' and snorkel_wide12_flights_data_obj.last_action_obj.filtered_col not in displayd_cols:
            return -1
    return 0


def LF_filter_from_displayed_column(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        displayd_cols = snorkel_wide12_flights_data_obj.get_displayed_cols()
        if snorkel_wide12_flights_data_obj.last_action_obj.filtered_col == 'info_line' or snorkel_wide12_flights_data_obj.last_action_obj.filtered_col in displayd_cols:
            return 1
    return 0


def LF_back_after_good_filter_readability_gain(snorkel_wide12_flights_data_obj):
    """
    Note: the threshold for 'ip_src', 'ip_dst', 'eth_src', 'eth_dst' is higher
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    # FIX: Check if prev_action_obj exists before accessing its attributes
    if (snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'back' and 
        snorkel_wide12_flights_data_obj.prev_action_obj is not None and 
        snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'filter'):
        # Filter term appears in human sessions
        if LF_filter_term_appears_in_human_session(snorkel_wide12_flights_data_obj) == 1:
            nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(ReadabilityGainType.BACK_AFTER_FILTER)
            if snorkel_wide12_flights_data_obj.prev_action_obj.filtered_col in {'ip_src', 'ip_dst', 'eth_src', 'eth_dst'}:
                if nrg > 0.35:
                    return 1
            else:
                if nrg > 0.2:
                    return 1
    return 0


def LF_back_after_bad_filter_readability_gain(snorkel_wide12_flights_data_obj):
    # FIX: Check if prev_action_obj exists before accessing its attributes
    if (snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'back' and 
        snorkel_wide12_flights_data_obj.prev_action_obj is not None and 
        snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'filter'):
        nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(ReadabilityGainType.BACK_AFTER_FILTER)
        if snorkel_wide12_flights_data_obj.prev_action_obj.filtered_col in {'ip_src', 'ip_dst', 'eth_src', 'eth_dst'}:
            if nrg <= -0.15:
                return -1
        else:
            if nrg <= -0.15:
                return -1
    return 0


def LF_group_results_in_single_group(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        if snorkel_wide12_flights_data_obj.num_of_rows_lst[-1] <= 1:
            return -1
    return 0


def LF_filter_small_number_of_rows(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[-2] < 40:
            return -1
    return 0


def LF_filter_results_in_single_row(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[-1] <= 1:
            return -1
    return 0


def LF_filter_on_the_same_column_in_subsession(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        last_filtered_column = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col

        filtered_column_ops = [snorkel_wide12_flights_data_obj.last_action_obj.filter_op]
        for action_obj in snorkel_wide12_flights_data_obj.prev_subsession_action_objs_lst:
            if action_obj.act_type == 'filter' and action_obj.filtered_col == last_filtered_column:
                filtered_column_ops.append(action_obj.filter_op)
        if len(filtered_column_ops) > 1 and len(filtered_column_ops) != filtered_column_ops.count("ne"):
            return -1
    return 0


def LF_end_episode_good_filter_readability_gain(snorkel_wide12_flights_data_obj):
    """
    Note: the threshold for 'ip_src', 'ip_dst', 'eth_src', 'eth_dst' is higher
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter' and len(
            snorkel_wide12_flights_data_obj) == snorkel_wide12_flights_data_obj.episode_length:
        # filter term appears in human sessions
        if LF_filter_term_not_appear_in_human_session(snorkel_wide12_flights_data_obj) != -1:
            nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(
                ReadabilityGainType.END_OF_EPISODE_FILTER)
            if snorkel_wide12_flights_data_obj.last_action_obj.filtered_col in {'ip_src', 'ip_dst', 'eth_src', 'eth_dst'}:
                if nrg > 0.44:
                    return 1
            else:
                if nrg > 0.2:
                    return 1
    return 0


def LF_end_episode_bad_filter_readability_gain(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter' and len(
            snorkel_wide12_flights_data_obj) == snorkel_wide12_flights_data_obj.episode_length:
        nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(ReadabilityGainType.END_OF_EPISODE_FILTER)
        if snorkel_wide12_flights_data_obj.prev_action_obj.filtered_col in {'ip_src', 'ip_dst', 'eth_src', 'eth_dst'}:
            if nrg <= -0.15:
                return -1
        else:
            if nrg <= -0.15:
                return -1
    return 0


def LF_using_not_equal_filter_operator(snorkel_wide12_flights_data_obj):
    """
    This function should reduce the incentive to use the "NOT EQUAL" filter operator,
    unless the numbers of rows filtered is more than half of the rows and the column filtered is 'departure_delay'
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        if (snorkel_wide12_flights_data_obj.last_action_obj.filter_op == 'ne' and
                2 * snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[-1] > snorkel_wide12_flights_data_obj.num_of_fdf_rows_lst[
                    -2] and
                remove_digit_suffix_from_column(snorkel_wide12_flights_data_obj.last_action_obj.filtered_col) not in ['departure_delay',
                                                                                   ]):
            return -1
    return 0


def LF_group_on_filtered_column_in_subsession(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group':
        prev_subsession = snorkel_wide12_flights_data_obj.prev_subsession_action_objs_lst
        grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col

        for action_obj in prev_subsession:
            if action_obj.act_type == 'filter' and action_obj.filter_op in ['eq',
                                                                            'contains'] and action_obj.filtered_col == grouped_col:
                return -1
    return 0


def helper_LF_group_after_group(snorkel_wide12_flights_data_obj, group_col1, group_col2):
    """
    Returns True if the last action in snorkel_wide12_flights_data_obj is a group action on 'group_col2' column and
    the previous actions was a group on 'group_col1' column and the display was not filtered. Return False otherise
    Args:
        snorkel_wide12_flights_data_obj:
        group_col1:
        group_col2:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and snorkel_wide12_flights_data_obj.last_action_obj.grouped_col == group_col2:
        if len(snorkel_wide12_flights_data_obj) > 1 and len(snorkel_wide12_flights_data_obj.get_subsession_filtered_cols()) == 0:
            if snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'group' and snorkel_wide12_flights_data_obj.prev_action_obj.grouped_col == group_col1:
                return True
    return False


def LF_first_group_scheduled_trip_time(snorkel_wide12_flights_data_obj):
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and remove_digit_suffix_from_column(snorkel_wide12_flights_data_obj.last_action_obj.grouped_col) in ['scheduled_trip_time']:
        if len(snorkel_wide12_flights_data_obj.subsession_action_objs_lst) == 1:
            return -1
    return 0


def LF_group_after_bad_filter_readability_gain(snorkel_wide12_flights_data_obj):
    """
    If we have a GROUP action after a FILTER action, but removing the FILTER action and taking the GROUP action
    immediately (instead of the FILTER action) is not humane if the result with the FILTER action is not more
    readable than without it. In such a case the function returns -1.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if (len(snorkel_wide12_flights_data_obj) > 1 and
            snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and
            snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'filter'
    ):
        nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(ReadabilityGainType.GROUP_AFTER_FILTER)
        # The display was already grouped
        if len(snorkel_wide12_flights_data_obj.get_grouped_cols_cur_state_with_offset_from_end(1)) >= 1:
            if nrg <= -0.49:
                return -1
        if nrg <= -0.3:
            return -1
    return 0


def LF_group_after_recursive_bad_filter_readability_gain(snorkel_wide12_flights_data_obj):
    """
    This function return -1 in the following case: The last action is a GROUP action that became after two consecutive
    FILTER actions. The first filter action didn't improved the readability and the second also didn't improve the
    readability regarding the GROUP action (.i.e. if we made the group action immediately without taking the FILTER
    action before it, the resulted display is as readable as with the filter action). The rational behind this
    kind of punishment is that it reasonable that the agent will "sacrifice" a bed FILTER action (the first one) to
    get a good readable grouping at the end, but if also the second filter was useless, we understand that this
    sacrifice was useless, so we further punish the agent.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if (len(snorkel_wide12_flights_data_obj) > 2 and snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and
            snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'filter' and
            snorkel_wide12_flights_data_obj.get_ith_action_obj_from_end(3).act_type == 'filter'):

        if LF_group_after_bad_filter_readability_gain(snorkel_wide12_flights_data_obj) == -1:
            nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(
                ReadabilityGainType.GROUP_AFTER_RECURSIVE_FILTER)
            if nrg <= -0.3:
                return -1
    return 0


def is_first_group_non_filtered(snorkel_wide12_flights_data_obj):
    """
    Return False if the last action action in `snorkel_wide12_flights_data_obj` (which is assumed to be a group action
    on a non-filtered display) was have been taken before (i.e. a group action on the same column). Returns True
    otherwise.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    filter_terms_stack = []
    grouped_col = snorkel_wide12_flights_data_obj.last_action_obj.grouped_col

    prev_snorkel_wide12_flights_data_obj = SnorkelWide12FlightsDataObj.copy_with_offset_from_end(snorkel_wide12_flights_data_obj, 1)
    if len(prev_snorkel_wide12_flights_data_obj.actions_lst) > 0 and (
            prev_snorkel_wide12_flights_data_obj.get_ith_action_obj_from_end(
                len(prev_snorkel_wide12_flights_data_obj)).act_type != 'back'
    ):
        filter_terms_stack.append(prev_snorkel_wide12_flights_data_obj.filter_terms_lst[0])

    for i in range(1, len(prev_snorkel_wide12_flights_data_obj.actions_lst)):
        cur_act = prev_snorkel_wide12_flights_data_obj.get_ith_action_obj_from_end(len(prev_snorkel_wide12_flights_data_obj) - i)

        if filter_terms_stack and cur_act.act_type == 'back':
            filter_terms_stack.pop()
        else:
            filter_terms_stack.append(prev_snorkel_wide12_flights_data_obj.filter_terms_lst[i])

        if cur_act.act_type == 'group' and cur_act.grouped_col == grouped_col:
            if all([filter_term is None for filter_term in filter_terms_stack]):
                return False  # not first event

    return True  # first event


def LF_same_group_non_filtered(snorkel_wide12_flights_data_obj):
    """
    A punishment is given for a GROUP action on the same column on an unfiltered display if this action has been
    already taken before unless the grouped column is in {'origin_airport', 'departure_delay', 'airline'}.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if (snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and
            remove_digit_suffix_from_column(snorkel_wide12_flights_data_obj.last_action_obj.grouped_col) not in {
                'origin_airport', 'departure_delay', 'airline',
            }
            and len(snorkel_wide12_flights_data_obj.get_subsession_filtered_cols()) == 0
    ):
        if not is_first_group_non_filtered(snorkel_wide12_flights_data_obj):
            return -1

    return 0


def LF_filter_after_bad_filter_readability_gain(snorkel_wide12_flights_data_obj):
    """
    If we have a FILTER action after a FILTER action, but removing the first FILTER action and taking the second FILTER
    action immediately (instead of the first FILTER action) is not humane if the result with the first FILTER action
    is not more readable than without it. In such a case the function returns -1.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if (len(snorkel_wide12_flights_data_obj) > 1 and
            snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter' and
            snorkel_wide12_flights_data_obj.prev_action_obj.act_type == 'filter'
    ):
        nrg = snorkel_wide12_flights_data_obj.compute_normalized_readability_gain(ReadabilityGainType.FILTER_AFTER_FILTER)
        if nrg <= -0.3:
            return -1
    return 0


# def LF_grouped_col_once_in_a_session(snorkel_wide12_flights_data_obj):
#     """
#     GROUP on the column 'tcp_srcport' or 'tcp_dstport' more than once in the session is not humane (no matter
#     if the display was filter or not)
#     """
#     if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'group' and snorkel_wide12_flights_data_obj.last_action_obj.grouped_col in {
#         'tcp_srcport', 'tcp_dstport'}:
#         if snorkel_wide12_flights_data_obj.get_all_session_grouped_col_counts()[
#             snorkel_wide12_flights_data_obj.last_action_obj.grouped_col] > 1:
#             return -1
#     return 0


def LF_filter_term_appears_once_in_session(snorkel_wide12_flights_data_obj):
    """
    Using the same filter term on the same column throughout the session is not humane.
    Args:
        snorkel_wide12_flights_data_obj:

    Returns:

    """
    if snorkel_wide12_flights_data_obj.last_action_obj.act_type == 'filter':
        filter_term = snorkel_wide12_flights_data_obj.last_action_obj.filter_term
        filtered_col = snorkel_wide12_flights_data_obj.last_action_obj.filtered_col
        filter_term_per_column_counts = snorkel_wide12_flights_data_obj.get_all_session_filtered_col_and_terms_counts()

        base_column = remove_digit_suffix_from_column(filtered_col)

        if filter_term_per_column_counts[(filtered_col,
                                          filter_term)] > 1:
            return -1
        elif filtered_col == 'URG' and (
                (filtered_col, '[FIN, PSH, URG]')
                in filter_term_per_column_counts
        ):
            return -1
        elif filtered_col == '[FIN, PSH, URG]' and (
                (snorkel_wide12_flights_data_obj.last_action_obj.filtered_col, 'URG')
                in filter_term_per_column_counts
        ):
            return -1
        elif filtered_col == 'PSH' and (
                (filtered_col, '[PSH, ACK]')
                in filter_term_per_column_counts
        ):
            return -1
        elif filtered_col == '[PSH, ACK]' and (
                (snorkel_wide12_flights_data_obj.last_action_obj.filtered_col, 'PSH')
                in filter_term_per_column_counts
        ):
            return -1
        elif filtered_col in {'[SYN, ACK]', '[SYN]'} and (
                (filtered_col, 'SYN')
                in filter_term_per_column_counts
        ):
            return -1
        elif filtered_col == 'SYN' and (
                (filtered_col, '[SYN, ACK]') in filter_term_per_column_counts or
                (filtered_col, '[SYN]') in filter_term_per_column_counts
        ):
            return -1
    return 0


priority_LF_lst = [
    LF_empty_display,
    LF_inhumane_columns_group,
    LF_column_already_grouped,
    LF_inhumane_columns_filter,
    LF_filter_num_of_rows_unchanged,
    LF_back_with_no_history,
    LF_group_as_first_action,
    LF_stacking_more_than_two_filters,
    LF_filter_from_undisplayed_column,
    LF_group_results_in_single_group,
    LF_filter_small_number_of_rows,
    LF_filter_results_in_single_row,
    LF_filter_on_the_same_column_in_subsession,
    LF_group_on_filtered_column_in_subsession,
    LF_first_group_scheduled_trip_time,
    LF_same_group_non_filtered,
    # LF_filter_term_not_appear_in_human_session,
    LF_filter_after_bad_filter_readability_gain,
    LF_filter_term_appears_once_in_session,
    LF_group_num_of_groups_unchanged

]


data_driven_LF_lst = [
    LF_empty_display,
    LF_group_num_of_groups_unchanged,
    LF_filter_num_of_groups_unchanged,
    LF_filter_num_of_rows_unchanged,
    LF_back_after_good_filter_readability_gain,
    LF_back_after_bad_filter_readability_gain,
    LF_group_results_in_single_group,
    LF_filter_small_number_of_rows,
    LF_filter_results_in_single_row,
    LF_end_episode_good_filter_readability_gain,
    LF_end_episode_bad_filter_readability_gain,
    LF_group_after_bad_filter_readability_gain,
    LF_group_after_recursive_bad_filter_readability_gain,
    LF_filter_after_bad_filter_readability_gain,
]

L_fns_priors_pairs = {LF_empty_display: 1.0,
                      LF_most_humane_columns_group: 0.35,
                      LF_humane_columns_group2: 0.15,
                      LF_humane_columns_group: 0.05,
                      LF_neutral_columns_group: 0.1,
                      LF_inhumane_columns_group: 1.0,
                      LF_column_already_grouped: 1.0,
                      LF_group_num_of_groups_unchanged: 1.0,
                      # LF_group_num_of_groups_changed: 0.05,
                      LF_stacking_five_groups: 0.8,
                      LF_stacking_more_than_five_groups: 1.0,
                      LF_filter_term_appears_in_human_session: 0.48,
                      # LF_filter_term_not_appear_in_human_session: 1.0,
                      LF_most_humane_columns_filter: 0.35,
                      LF_most_humane_columns_filter2: 0.15,
                      LF_humane_columns_filter: 0.05,
                      LF_neutral_columns_filter: 0.1,
                      LF_inhumane_columns_filter: 1.0,
                      LF_filter_num_of_groups_unchanged: 0.4,
                      LF_filter_num_of_rows_unchanged: 1.0,
                      LF_back_with_no_history: 1.0,
                      LF_back_after_back: 0.1,
                      LF_group_as_first_action: 1.0,
                      LF_stacking_more_than_two_filters: 1.0,
                      LF_filter_from_undisplayed_column: 1.0,
                      # LF_filter_from_displayed_column: 0.1,
                      LF_back_after_good_filter_readability_gain: 0.7,
                      LF_back_after_bad_filter_readability_gain: 0.5,
                      LF_group_results_in_single_group: 1.0,
                      LF_filter_small_number_of_rows: 1.0,
                      LF_filter_results_in_single_row: 1.0,
                      LF_filter_on_the_same_column_in_subsession: 1.0,
                      LF_end_episode_good_filter_readability_gain: 0.7,
                      LF_end_episode_bad_filter_readability_gain: 0.7,
                      LF_using_not_equal_filter_operator: 0.6,
                      LF_group_on_filtered_column_in_subsession: 1.0,
                      LF_first_group_scheduled_trip_time: 1.0,
                      LF_group_after_bad_filter_readability_gain: 0.8,
                      LF_group_after_recursive_bad_filter_readability_gain: 0.8,
                      LF_same_group_non_filtered: 1.0,
                      # LF_group_ip_src_after_group_eth_src: 0.45,
                      LF_filter_after_bad_filter_readability_gain: 1.0,
                      # LF_grouped_col_once_in_a_session: 0.8,
                      LF_filter_term_appears_once_in_session: 1.0
                      }

L_fns = list(L_fns_priors_pairs.keys())

# Define an Enum for the labeling functions dinamically
# See https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python
SnorkelWide12FlightsRule = enum.Enum('SnorkelWide12FlightsRule', {fn.__name__: idx for idx, fn in enumerate(L_fns)})
