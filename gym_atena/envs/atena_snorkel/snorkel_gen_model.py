from scipy import sparse
import numpy as np

# Use compatibility layer for modern Snorkel versions
try:
    from snorkel.learning import GenerativeModel
    print("Using original Snorkel API")
except ImportError:
    # Fallback to our compatibility adapter
    from .snorkel_compatibility import GenerativeModel
    print("Using Snorkel compatibility adapter")
import Configuration.config as cfg
from arguments import SchemaName

import gym_atena.envs.atena_snorkel.atena_snorkel_networking_lfs as net_lfs
import gym_atena.envs.atena_snorkel.atena_snorkel_flights_lfs as flights_lfs
import gym_atena.envs.atena_snorkel.atena_snorkel_big_flights_lfs as big_flights_lfs
import gym_atena.envs.atena_snorkel.atena_snorkel_wide_flights_lfs as wide_flights_lfs
import gym_atena.envs.atena_snorkel.atena_snorkel_wide12_flights_lfs as wide12_flights_lfs
from gym_atena.envs.atena_snorkel.utils import LfStatsObj, cross_entropy


class SnorkelModel(object):
    def __init__(self):
        schema_name = SchemaName(cfg.schema)

        # A Snorkel generative model for the cybersecurity datasets
        if schema_name is SchemaName.NETWORKING:
            self.gen_model_dir = 'snorkel_checkpoints'
            if cfg.analysis_mode:
                # Use relative path to ATENA-master checkpoints
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                master_path = os.path.join(current_dir, '../../../../../ATENA-master/snorkel_checkpoints')
                if os.path.isdir(master_path):
                    self.gen_model_dir = master_path
                else:
                    self.gen_model_dir = 'snorkel_checkpoints'  # fallback to local
            self.SnorkelRule = net_lfs.SnorkelNetRule
        elif schema_name is SchemaName.FLIGHTS:
            self.gen_model_dir = 'snorkel_flights_checkpoints'
            if cfg.analysis_mode:
                # Use relative path to ATENA-master checkpoints
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                master_path = os.path.join(current_dir, '../../../../../ATENA-master/snorkel_flights_checkpoints')
                if os.path.isdir(master_path):
                    self.gen_model_dir = master_path
                else:
                    self.gen_model_dir = 'snorkel_flights_checkpoints'  # fallback to local
            self.SnorkelRule = flights_lfs.SnorkelFlightsRule
        elif schema_name is SchemaName.BIG_FLIGHTS:
            self.gen_model_dir = 'snorkel_big_flights_checkpoints'
            if cfg.analysis_mode:
                # Use relative path to ATENA-master checkpoints
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                master_path = os.path.join(current_dir, '../../../../../ATENA-master/snorkel_big_flights_checkpoints')
                if os.path.isdir(master_path):
                    self.gen_model_dir = master_path
                else:
                    self.gen_model_dir = 'snorkel_big_flights_checkpoints'  # fallback to local
            self.SnorkelRule = big_flights_lfs.SnorkelBigFlightsRule
        elif schema_name is SchemaName.WIDE_FLIGHTS:
            self.gen_model_dir = 'snorkel_wide_flights_checkpoints'
            if cfg.analysis_mode:
                # Use relative path to ATENA-master checkpoints
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                master_path = os.path.join(current_dir, '../../../../../ATENA-master/snorkel_wide_flights_checkpoints')
                if os.path.isdir(master_path):
                    self.gen_model_dir = master_path
                else:
                    self.gen_model_dir = 'snorkel_wide_flights_checkpoints'  # fallback to local
            self.SnorkelRule = wide_flights_lfs.SnorkelWideFlightsRule
        elif schema_name is SchemaName.WIDE12_FLIGHTS:
            self.gen_model_dir = 'snorkel_wide12_flights_checkpoints'
            if cfg.analysis_mode:
                # Use relative path to ATENA-master checkpoints
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                master_path = os.path.join(current_dir, '../../../../../ATENA-master/snorkel_wide12_flights_checkpoints')
                if os.path.isdir(master_path):
                    self.gen_model_dir = master_path
                else:
                    self.gen_model_dir = 'snorkel_wide12_flights_checkpoints'  # fallback to local
            self.SnorkelRule = wide12_flights_lfs.SnorkelWide12FlightsRule
        else:
            raise NotImplementedError

    @property
    def schema_lfs_module(self):
        schema_name = SchemaName(cfg.schema)
        if schema_name is SchemaName.NETWORKING:
            return net_lfs
        elif schema_name is SchemaName.FLIGHTS:
            return flights_lfs
        elif schema_name is SchemaName.BIG_FLIGHTS:
            return big_flights_lfs
        elif schema_name is SchemaName.WIDE_FLIGHTS:
            return wide_flights_lfs
        elif schema_name is SchemaName.WIDE12_FLIGHTS:
            return wide12_flights_lfs
        else:
            raise NotImplementedError

    def get_non_abstain_functions(self, snorkel_data_obj):
        """
        Return all names of non-abstain (result != 0) labeling functions
        Args:
            snorkel_data_obj (SnorkelDataObj):

        Returns:

        """
        L_test, non_abstain_lfs_dict = self.get_labeling_functions_matrix_for_single_obj_and_non_abstain(
            snorkel_data_obj)
        return non_abstain_lfs_dict

    @property
    def data_driven_rules(self):
        return self.schema_lfs_module.data_driven_LF_lst

    def get_labeling_functions_matrix_for_single_obj_and_non_abstain(self, snorkel_data_obj):
        """
        Returns a tuple (L_test, non_abstain_lfs_dict) where L_test is a matrix with one column where each row represents
        another labeling function, and the entries are the score that this labeling function gives for the given
        `snorkel_data_obj`. non_abstain_lfs_dict is a dictionary containing the non-abstain (non-zero score) functions
         as key and the score as value.
        Args:
            snorkel_data_obj:

        Returns:

        """
        non_abstain_lfs_dict = {}
        L_test = np.zeros((len(self.schema_lfs_module.L_fns), 1)).astype(int)
        priority_tests_success = self.get_priority_tests_result(snorkel_data_obj)
        for i, L_fn in enumerate(self.schema_lfs_module.L_fns):
            try:
                L_fn_score = L_fn(snorkel_data_obj)

                # Add to non abstain functions if score != 0
                if L_fn_score != 0:
                    non_abstain_lfs_dict[self.SnorkelRule[L_fn.__name__]] = L_fn_score

                # Add score to matrix
                if priority_tests_success:
                    L_test[i, 0] = L_fn_score
                else:
                    L_test[i, 0] = -1
                    
            except (IndexError, AttributeError, KeyError) as e:
                # Handle cases where LF fails due to insufficient data
                print(f"LF {L_fn.__name__} failed: {e} - defaulting to abstain (0)")
                L_test[i, 0] = 0  # Abstain on error
        return L_test, non_abstain_lfs_dict

    def get_lfs_log_scale_priors(self):
        priors = []

        for lf in self.schema_lfs_module.L_fns:
            prior = self.schema_lfs_module.L_fns_priors_pairs[lf]
            priors.append(prior)

        return priors

    def print_failed_priority_tests(self, snorkel_data_obj):
        """Print failed priority tests with proper error handling"""
        for lf in self.schema_lfs_module.priority_LF_lst:
            try:
                if lf(snorkel_data_obj) == -1:
                    print(lf.__name__)
            except (IndexError, AttributeError) as e:
                print(f"Priority LF {lf.__name__} failed: {e}")

    def get_priority_tests_result(self, snorkel_data_obj):
        """Check priority labeling functions with proper error handling"""
        for lf in self.schema_lfs_module.priority_LF_lst:
            try:
                if lf(snorkel_data_obj) == -1:
                    return False
            except (IndexError, AttributeError) as e:
                # Handle cases where snorkel_data_obj doesn't have sufficient history
                print(f"Priority LF {lf.__name__} failed: {e} - assuming success")
                continue
        return True


class SnorkelGenModel(SnorkelModel):
    """
    A Snorkel generative model
    """
    def __init__(self):
        super().__init__()
        self.gen_model = GenerativeModel()
        self.gen_model.load(save_dir=self.gen_model_dir)

    def learned_lf_stats(self):
        """
        Returns statistics (TP, FP, ...) w.r.t to each labeling function
        Returns:

        """
        return self.gen_model.learned_lf_stats()

    def get_coherency_score_prediction(self, snorkel_data_obj):
        """
        Returns the coherency score of the generative model for the given action
        Args:
            snorkel_data_obj (SnorkelDataObj):

        Returns:

        """
        return self.get_coherency_score_prediction_and_non_abstain_funcs(snorkel_data_obj)[0]

    def get_coherency_score_prediction_and_non_abstain_funcs(self, snorkel_data_obj):
        """
        Returns the coherency score of the generative model for
        the given action and the non abstain functions (non-zero)
        Args:
            snorkel_data_obj (SnorkelDataObj):

        Returns:

        """
        L_test, non_abstain_lfs_dict = self.get_labeling_functions_matrix_for_single_obj_and_non_abstain(
            snorkel_data_obj)
        L_test_sparse = sparse.csr_matrix(L_test.T)
        return self.gen_model.marginals(L_test_sparse)[0], non_abstain_lfs_dict

    def apply(self, cands):
        """
        Applies the generative model to each sample in the list of candidates
        Args:
            cands (List(SnorkelDataObj)):

        Returns:

        """
        result = []

        for cand in cands:
            pred = self.get_coherency_score_prediction(cand)
            result.append(pred)

        return result

    def display_error_analysis(self, cands, gold_labels):
        """
        Display statistics (TP, FP, ...) w.r.t. the generative model
        Args:
            cands:
            gold_labels:

        Returns:

        """
        gen_labels = self.apply(cands)

        lf_stats_obj = LfStatsObj(self.get_coherency_score_prediction, cands, gold_labels,
                                  generative_model_prediction=True)
        lf_stats_obj.display_lf_stats()

        print('Cross-entropy loss:')
        print(cross_entropy(gen_labels, gold_labels))
