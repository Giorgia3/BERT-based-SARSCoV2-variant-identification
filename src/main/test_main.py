import argparse
import os.path
import time
from src.utils.general_utils import setup_config
from src.classification.classifier import Tester

def args_parse():
    """
       Description: Parse command-line arguments.
       :returns: arguments parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetsdir',
                        help='Datasets directory',
                        required=True,
                        type=str)
    args_list = parser.parse_args()
    return args_list


if __name__ == '__main__':
    start = time.time()
    args = args_parse()
    config = setup_config(args.datasetsdir)
    print("Test procedure started")
    tester = Tester(config)
    tester.test(config)
    tester.report(config)
    end = time.time()
    print(end - start)
