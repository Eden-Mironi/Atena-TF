import json
import os
from multiprocessing import Pool
import logging
import numpy as np
import Configuration.config as cfg
from arguments import SchemaName
from gym_atena.envs.atena_snorkel.atena_snorkel_networking import SnorkelNetDataObj
from gym_atena.envs.atena_snorkel.atena_snorkel_flights import SnorkelFlightDataObj
from gym_atena.envs.atena_snorkel.atena_snorkel_big_flights import SnorkelBigFlightsDataObj
from gym_atena.envs.atena_snorkel.atena_snorkel_wide_flights import SnorkelWideFlightsDataObj
from gym_atena.envs.atena_snorkel.atena_snorkel_wide12_flights import SnorkelWide12FlightsDataObj


logger = logging.getLogger(__name__)

# Use relative path for Snorkel dataset files
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# First priority: local data directory in atena-tf 2
SNORKEL_FILES_DIR_PATH = os.path.join(current_dir, '../../../../snorkel_data/')
if not os.path.isdir(SNORKEL_FILES_DIR_PATH):
    # Second priority: current working directory
    SNORKEL_FILES_DIR_PATH = os.getcwd()
if not os.path.isdir(SNORKEL_FILES_DIR_PATH):
    # Third priority: try ATENA-master as fallback
    SNORKEL_FILES_DIR_PATH = os.path.join(current_dir, '../../../../../ATENA-master/')
if not os.path.isdir(SNORKEL_FILES_DIR_PATH):
    # Final fallback: parent directory of atena-tf 2
    SNORKEL_FILES_DIR_PATH = os.path.join(current_dir, '../../../../../')
NET_SNORKEL_TRAINING_SET_FILE_NAME = 'snorkel_dataset.jsonl'
FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME = 'snorkel_dataset_flights.jsonl'
BIG_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME = 'snorkel_dataset_big_flights.jsonl'
WIDE_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME = 'snorkel_dataset_wide_flights.jsonl'
WIDE12_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME = 'snorkel_dataset_wide12_flights.jsonl'
NET_SNORKEL_TEST_SET_FILE_NAME = 'snorkel_test_dataset.jsonl'
NET_SNORKEL_GOLD_LABELS_TEST_SET_FILE_NAME = 'snorkel_gold_labels_test_dataset.txt'


def process_line(line):
    """
    Return a single snorkel data object from a single line in a json lines file
    Args:
        line:

    Returns:

    """
    schema_name = SchemaName(cfg.schema)
    if schema_name is SchemaName.NETWORKING:
        SnorkelDataObj = SnorkelNetDataObj
    elif schema_name is SchemaName.FLIGHTS:
        SnorkelDataObj = SnorkelFlightDataObj
    elif schema_name is SchemaName.BIG_FLIGHTS:
        SnorkelDataObj = SnorkelBigFlightsDataObj
    elif schema_name is SchemaName.WIDE_FLIGHTS:
        SnorkelDataObj = SnorkelWideFlightsDataObj
    elif schema_name is SchemaName.WIDE12_FLIGHTS:
        SnorkelDataObj = SnorkelWide12FlightsDataObj
    else:
        raise NotImplementedError
    return SnorkelDataObj.construct_from_dict(json.loads(line))


def reset_test_dataset():
    """
    Deletes content of Snorkel test dataset
    Returns:

    """
    with open(NET_SNORKEL_GOLD_LABELS_TEST_SET_FILE_NAME, mode='w'), open(NET_SNORKEL_TEST_SET_FILE_NAME, mode='w'):
        pass


def save_gold_labels_test_dataset(labels):
    """
    Adds a bunch of gold labels to the gold labels test dataset
    Args:
        labels (List(float)):

    Returns:

    """
    # Log candidate to training dataset
    with open(NET_SNORKEL_GOLD_LABELS_TEST_SET_FILE_NAME, mode='a') as f:
        for label in labels:
            f.write(str(label) + '\n')


def save_gold_label_test_dataset(label):
    """
    Adds a single gold label to the gold labels test dataset
    Args:
        label (float):

    Returns:

    """
    save_gold_labels_test_dataset([label])


class DataLoader(object):
    """
    Load snorkel objects from a json lines file
    """
    def __init__(self, data_path=SNORKEL_FILES_DIR_PATH, file_name=NET_SNORKEL_TRAINING_SET_FILE_NAME, max_examples_num=None):
        if file_name in {NET_SNORKEL_TRAINING_SET_FILE_NAME, NET_SNORKEL_TEST_SET_FILE_NAME}:
            logger.warning(f"Loading examples for NETWORKING schema!")
        elif file_name == FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME:
            logger.warning(f"Loading examples for FLIGHTS schema!")
        elif file_name == BIG_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME:
            logger.warning(f"Loading examples for BIG FLIGHTS schema!")
        elif file_name == WIDE_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME:
            logger.warning(f"Loading examples for WIDE FLIGHTS schema!")
        elif file_name == WIDE12_FLIGHTS_SNORKEL_TRAINING_SET_FILE_NAME:
            logger.warning(f"Loading examples for WIDE12 FLIGHTS schema!")
        else:
            raise NotImplementedError
        self.data_path = data_path

        self.num_of_data_elements = 0
        self.data = []
        if max_examples_num is None:
            max_examples_num = 1e8
        max_lines = max_examples_num

        # Create a list of training examples for Snorkel
        pool = Pool(4)
        lines = []
        with open(os.path.join(self.data_path, file_name), mode='r', encoding='utf-8') as f:
            for line in f:
                if max_lines > 0:
                    self.num_of_data_elements += 1
                    lines.append(line)
                else:
                    break
                max_lines -= 1

        # chunk the work into batches of 4 lines at a time
        self.data = pool.map(process_line, lines, 4)

    @classmethod
    def load_training_data(cls, max_examples_num=200000):
        return cls(max_examples_num=max_examples_num)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for idx in range(len(self.data)):
            yield self.data[idx]


class DataLoaderWithLabels(DataLoader):
    """
    Load snorkel objects from a json lines (.jsonl) file and labels from a .txt file
    """
    def __init__(self, data_path=SNORKEL_FILES_DIR_PATH,
                 file_name=NET_SNORKEL_TEST_SET_FILE_NAME,
                 labels_file_name=NET_SNORKEL_GOLD_LABELS_TEST_SET_FILE_NAME,
                 max_examples_num=None):
        super().__init__(data_path=data_path,
                         file_name=file_name,
                         max_examples_num=max_examples_num)

        self.labels = []
        if max_examples_num is None:
            max_examples_num = 1e8
        max_lines = max_examples_num

        # Create a list of labels
        with open(os.path.join(self.data_path, labels_file_name), mode='r', encoding='utf-8') as f:
            for line in f:
                if max_lines > 0:
                    self.labels.append(float(line))
                else:
                    break
                max_lines -= 1

        assert len(self.labels) == len(self.data)

    @classmethod
    def load_test_data(cls,
                       file_name=NET_SNORKEL_TEST_SET_FILE_NAME,
                       labels_file_name=NET_SNORKEL_GOLD_LABELS_TEST_SET_FILE_NAME,
                       max_examples_num=None):
        return cls(file_name=file_name,
                   labels_file_name=labels_file_name,
                   max_examples_num=max_examples_num)
