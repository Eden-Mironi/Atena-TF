import math
from collections import OrderedDict, defaultdict
from enum import Enum

from gym_atena.lib import helpers as ATENAUtils
from gym_atena.lib.helpers import INT_OPERATOR_MAP_ATENA_STR


class ReadabilityGainType(Enum):
    BACK_AFTER_FILTER = "back_after_filter"
    GROUP_AFTER_FILTER = "group_after_filter"
    GROUP_AFTER_RECURSIVE_FILTER = "group_after_recursive_filter"
    FILTER_AFTER_FILTER = "filter_after_filter"
    END_OF_EPISODE_FILTER = "end_of_episode_filter"


class ActionObj(object):
    """
    Helper class to hold data w.r.t. an analysis action
    """
    def __init__(self, act_type, grouped_col=None, filtered_col=None, aggregated_col=None, filter_op=None,
                 filter_term=None):
        self.act_type = act_type
        self.grouped_col = grouped_col
        self.filtered_col = filtered_col
        self.aggregated_col = aggregated_col
        self.filter_op = filter_op
        self.filter_term = filter_term


class SnorkelDataObj(object):
    KEYS = None
    AGG_KEYS = None
    FILTER_COLS = None
    GROUP_COLS = None

    """
    Holds data w.r.t to a single sample (analysis action) for Snorkel
    """
    def __init__(self, actions_lst, filter_terms_lst, num_of_rows_lst, num_of_fdf_rows_lst,
                 num_of_immediate_action_rows_lst,
                 dataset_number,
                 episode_length=12):
        self.actions_lst = actions_lst
        self.filter_terms_lst = filter_terms_lst
        self.num_of_rows_lst = num_of_rows_lst
        self.num_of_fdf_rows_lst = num_of_fdf_rows_lst
        self.num_of_immediate_action_rows_lst = num_of_immediate_action_rows_lst
        self.dataset_number = dataset_number
        self.episode_length = episode_length

    @classmethod
    def construct_from_dict(cls, dic, episode_length=12):
        return cls(dic['actions_lst'], dic['filter_terms_lst'], dic['num_of_rows_lst'], dic['num_of_fdf_rows_lst'],
                   dic.get('num_of_immediate_action_rows_lst', None),
                   dic.get('dataset_num', None), episode_length)

    @classmethod
    def copy_with_offset_from_end(cls, snorkel_net_data_obj, offset):
        assert offset >= 0
        return cls(snorkel_net_data_obj.actions_lst[: len(snorkel_net_data_obj.actions_lst)-offset],
                   snorkel_net_data_obj.filter_terms_lst[: len(snorkel_net_data_obj.filter_terms_lst) - offset],
                   snorkel_net_data_obj.num_of_rows_lst[: len(snorkel_net_data_obj.num_of_rows_lst) - offset],
                   snorkel_net_data_obj.num_of_fdf_rows_lst[: len(snorkel_net_data_obj.num_of_fdf_rows_lst) - offset],
                   snorkel_net_data_obj.num_of_immediate_action_rows_lst[: len(snorkel_net_data_obj.num_of_immediate_action_rows_lst) - offset],
                   snorkel_net_data_obj.dataset_number,
                   snorkel_net_data_obj.episode_length,
                   )

    def get_last_action_vec(self):
        return self.actions_lst[-1]

    def get_prev_action_vec(self):
        if len(self.actions_lst) > 1:
            return self.actions_lst[-2]
        return None

    def get_ith_action_vec_from_end(self, i):
        if len(self.actions_lst) >= i:
            return self.actions_lst[-1 * i]
        return None

    def get_action_obj(self, act_vec, filter_term):
        act_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[act_vec[0]]
        action_obj = ActionObj(act_type)

        if act_type == 'filter':
            action_obj.filtered_col = self.FILTER_COLS[act_vec[1]]
            action_obj.filter_op = INT_OPERATOR_MAP_ATENA_STR[act_vec[2]]
            action_obj.filter_term = filter_term
        elif act_type == 'group':
            action_obj.grouped_col = self.GROUP_COLS[act_vec[1]]
            action_obj.aggregated_col = self.AGG_KEYS[0]

        return action_obj

    @property
    def last_action_obj(self):
        return self.get_action_obj(self.get_last_action_vec(), self.filter_terms_lst[-1])

    @property
    def prev_action_obj(self):
        if len(self.actions_lst) > 1:
            return self.get_action_obj(self.get_prev_action_vec(), self.filter_terms_lst[-2])
        return None

    def get_ith_action_obj_from_end(self, i):
        assert i >= 1
        if len(self.actions_lst) >= i:
            return self.get_action_obj(self.get_ith_action_vec_from_end(i), self.filter_terms_lst[-1 * i])
        return None

    def get_all_session_action_objs(self):
        action_objs = []

        for act_vec, filter_term in zip(self.actions_lst, self.filter_terms_lst):
            action_objs.append(self.get_action_obj(act_vec, filter_term))

        return action_objs

    @property
    def subsession_action_objs_lst(self):
        stack = []

        for act_vec, filter_term in zip(self.actions_lst, self.filter_terms_lst):
            action_obj = self.get_action_obj(act_vec, filter_term)
            if stack and action_obj.act_type == 'back':
                stack.pop()
            else:
                stack.append(action_obj)

        return stack

    @property
    def prev_subsession_action_objs_lst(self):
        stack = []

        for act_vec, filter_term in zip(self.actions_lst[:-1], self.filter_terms_lst[:-1]):
            action_obj = self.get_action_obj(act_vec, filter_term)
            if stack and action_obj.act_type == 'back':
                stack.pop()
            else:
                stack.append(action_obj)

        return stack

    def __len__(self):
        return len(self.actions_lst)

    def __repr__(self):
        return f'''actions_lst :
            {self.actions_lst} 

        filter_terms_lst:
            {self.filter_terms_lst}

        num_of_rows_lst:
            {self.num_of_rows_lst}

        num_of_fdf_rows_lst:
            {self.num_of_fdf_rows_lst}
        '''

    def get_displayed_cols(self):
        is_grouped = False
        displayed_cols = set()
        subsession_action_objs_lst = self.subsession_action_objs_lst
        for action_obj in subsession_action_objs_lst:
            if action_obj.act_type == 'group':
                is_grouped = True
                displayed_cols.add(action_obj.grouped_col)

        if not is_grouped:
            return self.KEYS
        else:
            return list(displayed_cols)

    def get_subsession_filtered_cols(self):
        """
        Return filtered columns in the order in which they are filtered. If the same column is filtered more
        than once, the first time determines the order.
        Returns:

        """
        filtered_cols = OrderedDict()
        subsession_action_objs_lst = self.subsession_action_objs_lst

        for action_obj in subsession_action_objs_lst:
            if action_obj.act_type == 'filter' and action_obj.filtered_col not in filtered_cols:
                filtered_cols[action_obj.filtered_col] = None  # None is a dummy value

        return list(filtered_cols.keys())

    def get_grouped_cols_cur_state_with_offset_from_end(self, offset):
        """
        Return grouped columns in the order in which they are grouped until `offest` actions from the end.
        If the same column is grouped more than once, the first time determines the order.
        Returns:

        """
        assert offset >= 0

        grouped_cols_markers_stack = []
        grouped_cols = OrderedDict()

        for i, (act_vec, filter_term) in enumerate(zip(self.actions_lst[: len(self.actions_lst)-offset], self.filter_terms_lst[:len(self.filter_terms_lst)-offset])):
            action_obj = self.get_action_obj(act_vec, filter_term)
            if grouped_cols_markers_stack and action_obj.act_type == 'back':
                groped_col_marker = grouped_cols_markers_stack.pop()
                if groped_col_marker:
                    grouped_cols.popitem(last=True)
            else:
                if action_obj.act_type == 'group' and action_obj.grouped_col not in grouped_cols:
                    grouped_cols[i] = action_obj.grouped_col
                    grouped_cols_markers_stack.append(True)
                else:
                    grouped_cols_markers_stack.append(False)

        return list(grouped_cols.values())

    def get_subsession_grouped_cols(self):
        """
        Return grouped columns in the order in which they are grouped. If the same column is grouped more
        than once, the first time determines the order.
        Returns:

        """
        grouped_cols = OrderedDict()
        subsession_action_objs_lst = self.subsession_action_objs_lst

        for action_obj in subsession_action_objs_lst:
            if action_obj.act_type == 'group' and action_obj.grouped_col not in grouped_cols:
                grouped_cols[action_obj.grouped_col] = None  # None is a dummy value

        return list(grouped_cols.keys())

    def get_all_session_grouped_col_counts(self):
        """
        Return a dictionary with counts of all grouped columns during the current session

        """
        grouped_cols = defaultdict(int)

        for action_obj in self.get_all_session_action_objs():
            if action_obj.act_type == 'group':
                grouped_cols[action_obj.grouped_col] += 1

        return grouped_cols

    def get_all_session_filtered_col_and_terms_counts(self):
        """
        Return a dictionary where key is a (filtered_col, filter_term) pair and value is count during session

        """
        filter_counts = defaultdict(int)

        for action_obj in self.get_all_session_action_objs():
            if action_obj.act_type == 'filter':
                filter_counts[(action_obj.filtered_col, action_obj.filter_term)] += 1

        return filter_counts

    def compute_normalized_readability_gain(self, readability_gain_type):
        assert isinstance(readability_gain_type, ReadabilityGainType)
        # assert (self.last_action_obj.act_type == 'back' and self.prev_action_obj.act_type == 'filter' or
        #         self.last_action_obj.act_type == 'group' and self.prev_action_obj.act_type == 'filter' or
        #         self.last_action_obj.act_type == 'filter' and self.prev_action_obj.act_type == 'filter' or
        #         self.last_action_obj.act_type == 'filter' and len(self) == self.episode_length or
        #         (offset_from_end is not None and offset_from_end > 0)
        #         )

        if readability_gain_type is ReadabilityGainType.GROUP_AFTER_RECURSIVE_FILTER:
            offset_from_end = 3
            filter_action = self.get_ith_action_obj_from_end(offset_from_end)
            if filter_action.filtered_col == 'info_line':
                disp_rows_prev = self.num_of_fdf_rows_lst[-offset_from_end]
                disp_rows_prev_prev = self.num_of_fdf_rows_lst[-(offset_from_end + 1)]
            else:
                disp_rows_prev = self.num_of_rows_lst[-offset_from_end]
                disp_rows_prev_prev = self.num_of_rows_lst[-(offset_from_end + 1)]

        elif readability_gain_type is ReadabilityGainType.BACK_AFTER_FILTER:
            filter_action = self.prev_action_obj
            if filter_action.filtered_col == 'info_line':
                disp_rows_prev = self.num_of_fdf_rows_lst[-2]
                disp_rows_prev_prev = self.num_of_fdf_rows_lst[-3]
            else:
                disp_rows_prev = self.num_of_rows_lst[-2]
                disp_rows_prev_prev = self.num_of_rows_lst[-3]
        elif readability_gain_type is ReadabilityGainType.GROUP_AFTER_FILTER:
            disp_rows_prev = self.num_of_rows_lst[-1]
            disp_rows_prev_prev = self.num_of_immediate_action_rows_lst[-1]
        elif readability_gain_type is ReadabilityGainType.END_OF_EPISODE_FILTER:
            assert len(self) == self.episode_length
            filter_action = self.last_action_obj
            if filter_action.filtered_col == 'info_line':
                disp_rows_prev = self.num_of_fdf_rows_lst[-1]
                disp_rows_prev_prev = self.num_of_fdf_rows_lst[-2]
            else:
                disp_rows_prev = self.num_of_rows_lst[-1]
                disp_rows_prev_prev = self.num_of_rows_lst[-2]
        elif readability_gain_type is ReadabilityGainType.FILTER_AFTER_FILTER:
            disp_rows_prev = self.num_of_rows_lst[-1]
            disp_rows_prev_prev = self.num_of_immediate_action_rows_lst[-1]
        else:
            raise NotImplementedError

        num_of_grouped_cols = max(1, len(self.get_subsession_grouped_cols()))
        num_of_grouped_cols = 1

        denominator_epsilon = 0.00001

        # how compact is the resulted display
        compact_display_score = ATENAUtils.normalized_sigmoid_fkt(0.5, 17,
                                                                  1 - 1 / math.log(
                                                                      9 + disp_rows_prev * num_of_grouped_cols, 9))
        if disp_rows_prev == 1:
            normalized_readability_gain = -1
        else:
            prev_readability = ATENAUtils.normalized_sigmoid_fkt(
                0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev * num_of_grouped_cols + denominator_epsilon, 9))
            prev_prev_readability = ATENAUtils.normalized_sigmoid_fkt(
                0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev_prev * num_of_grouped_cols + denominator_epsilon, 9))

            assert prev_readability >= prev_prev_readability

            # how compact the resulted display of the filter action relative to the display before it.
            readability_gain = 1 - prev_prev_readability / prev_readability

            # transforming the readability gain to be in range [-0.5, 0.5]
            normalized_readability_gain = -0.5 + 1.0 * ATENAUtils.normalized_sigmoid_fkt(
                0.5, 11, 1 - readability_gain * compact_display_score)
            # # making negative normalized_readability_gain in the range [-2.0, 0) instead of
            # # of [-0.5, 0) to 'cancel' potential gain of the filter action
            # if normalized_readability_gain < 0:
            #     normalized_readability_gain *= 4
        return normalized_readability_gain