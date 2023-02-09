import csv
import os

from tqdm import tqdm
from src.utils import paths_config, general_config, bio_config


# def setup_config(datasets_dir, bwa_bin_file=None):
#     paths_config.init(datasets_dir, bwa_bin_file)
#     general_config.init()
#     bio_config.init()
def setup_config():
    paths_config.init()
    general_config.init()
    bio_config.init()


def get_inverted_class_labels_dict():
    return {v: k for k, v in general_config.CLASS_LABELS.items()}


def get_train_val_test_sizes():
    if not os.path.exists(paths_config.trainvaltest_splits_dir):
        raise NotADirectoryError(f'Error: Directory not found: {paths_config.trainvaltest_splits_dir}')
    elif not os.path.exists(paths_config.trainvaltest_sizes_file):
        raise FileNotFoundError(f'Error: File not found: {paths_config.trainvaltest_sizes_file}')

    sizes_info = {}
    with open(paths_config.trainvaltest_sizes_file, 'r') as trainvaltest_sizes_fp:
        csvreader = csv.reader(trainvaltest_sizes_fp, delimiter=',')
        for line in tqdm(csvreader):
            sizes_info[line[0]] = int(line[1])
    return sizes_info


