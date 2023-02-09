import argparse
import os.path
import time
from src.utils.general_utils import setup_config
from src.classification.classifier import Tester

if __name__ == '__main__':
    start = time.time()
    setup_config()
    print("Test procedure started")
    tester = Tester()
    tester.test()
    tester.report()
    end = time.time()
    print(end - start)
