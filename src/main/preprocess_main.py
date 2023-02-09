import argparse

from src.preprocessing.preprocess import reformat_input_data, align_spike_sequences, check_duplicates, write_test_size_info
from src.utils.general_utils import setup_config
import time


# def args_parse():
#     """
#        Description: Parse command-line arguments.
#        :returns: arguments parser.
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--datasets',
#                         help='Datasets directory',
#                         required=True,
#                         type=str)
#     parser.add_argument('--bwa',
#                         help='BWA bin directory',
#                         required=True,
#                         type=str)
#     args_list = parser.parse_args()
#     return args_list


if __name__ == '__main__':
    start = time.time()
    # args = args_parse()
    # setup_config(args.datasets)
    setup_config()
    reformat_input_data()
    align_spike_sequences()
    # check_duplicates()
    # split_train_val_test()
    write_test_size_info()
    end = time.time()
    print(end - start)
