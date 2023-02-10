import csv
import os
from tqdm import tqdm
from src.utils.config_utils import Config


# def setup_config(datasets_dir, bwa_bin_file=None):
#     paths_config.init(datasets_dir, bwa_bin_file)
#     general_config.init()
#     bio_config.init()
def setup_config(datasets_dir, bwa_dir=None):
    return Config(datasets_dir, bwa_dir)


def get_inverted_class_labels_dict(config):
    return {v: k for k, v in config.general_config.CLASS_LABELS.items()}


def get_train_val_test_sizes(config):
    if not os.path.exists(config.paths_config.trainvaltest_splits_dir):
        raise NotADirectoryError(f'Error: Directory not found: {config.paths_config.trainvaltest_splits_dir}')
    elif not os.path.exists(config.paths_config.trainvaltest_sizes_file):
        raise FileNotFoundError(f'Error: File not found: {config.paths_config.trainvaltest_sizes_file}')

    sizes_info = {}
    with open(config.paths_config.trainvaltest_sizes_file, 'r') as trainvaltest_sizes_fp:
        csvreader = csv.reader(trainvaltest_sizes_fp, delimiter=',')
        for line in tqdm(csvreader):
            sizes_info[line[0]] = int(line[1])
    return sizes_info


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created")
    return dir_path
