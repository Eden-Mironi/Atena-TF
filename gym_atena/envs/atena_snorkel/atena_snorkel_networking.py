from gym_atena.envs.atena_snorkel.atena_snorkel_data import SnorkelDataObj
import gym_atena.data_schemas.netowrking.columns_data as columns_data


class SnorkelNetDataObj(SnorkelDataObj):
    """
    Holds data w.r.t to a single sample (analysis action) for Snorkel for Cybersecurity datasets
    """
    KEYS = columns_data.KEYS
    AGG_KEYS = columns_data.AGG_KEYS
    FILTER_COLS = columns_data.FILTER_COLS
    GROUP_COLS = columns_data.GROUP_COLS
