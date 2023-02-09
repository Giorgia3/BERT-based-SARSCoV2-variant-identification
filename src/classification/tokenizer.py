import csv
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
from src.utils import paths_config, general_config


def seq2kmer(seq, k, stride, string_format=False):
    """
    Convert original sequence to kmers with stride.
    Pad shorter sequences with 'N'.

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    list_seq = list(seq)
    not_divisible = (len(list_seq)%k != 0)
    while not_divisible:
        list_seq.append('N')
        not_divisible = (len(list_seq)%k != 0)
    kmers = [list_seq[x:x+k] for x in range(0, len(list_seq)-k+1, stride)]

    if string_format:
        return [''.join(kmer).lower() for kmer in kmers]

    return kmers


class Tokenizer:
    def __init__(self, *args):
        if len(args) > 0:
            tokenizer_dir = args[0]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
            self.token_count = None

    def __len__(self):
        return len(self.tokenizer)

    def __compute_token_count(self):
        if self.token_count is not None: # already computed
            return
        else:
            self.token_count = {}
            # compute the number of occurrencies of each token in the training dataset:
            # if token counts already computed load them from file
            if os.path.exists(paths_config.token_count_file):
                print("Token count file already exists.")
                with open(paths_config.token_count_file) as token_count_fp:
                    csvreader = csv.reader(token_count_fp, delimiter=',')
                    for line in tqdm(csvreader):
                        self.token_count[line[0]] = int(line[1])
            else:
                with open(paths_config.train_file) as train_fp, open(paths_config.token_count_file, 'w') as token_count_fp:
                    train_reader = csv.reader(train_fp, delimiter=',')
                    # loop on training data
                    for line in tqdm(train_reader):
                        label = line[0]
                        id = line[1]
                        pos = line[2]
                        seq = line[3]
                        # split sequence/chunk in kmers
                        tokens = seq2kmer(seq, general_config.K, general_config.STRIDE, string_format=True)
                        # update token count
                        for count, tok in enumerate(tokens):
                            # if count > general_config.MAX_LENGTH-2: #consider only tokens <MAX_LENGTH
                            #     break
                            if tok in self.token_count:
                                self.token_count[tok] += 1
                            else:
                                self.token_count[tok] = 1
                    # save token count on file
                    csvwriter = csv.writer(token_count_fp, delimiter=',')
                    for key, value in self.token_count.items():
                        csvwriter.writerow([key, value])

    def plot_kmers_histogram(self):
        self.__compute_token_count()

        # plot sorted token count histogram to find elbow, i.e. number of tokens which contribute to most of token occurrences
        token_count_ordered = {k: v for k, v in sorted(self.token_count.items(), key=lambda item: item[1], reverse=True)}
        fig = plt.figure(figsize=(20, 10))
        plt.step(range(len(token_count_ordered.keys())), list(token_count_ordered.values()), color='b')
        fig.suptitle(f"Ordered token occurrencies (k={general_config.K}, stride={general_config.STRIDE})", y=0.95,
                     fontsize=20)
        plt.xlabel(f'Tokens (tot num: {len(token_count_ordered)})', fontsize=16)
        plt.xticks(np.arange(0, len(token_count_ordered), 100), rotation=90)
        plt.ylabel('Num. occurrences', fontsize=16)
        plt.yticks(np.arange(0, max(list(token_count_ordered.values())), 1000))
        fig_path = Path(paths_config.tokens_histograms_dir) / f"k{general_config.K}_s{general_config.STRIDE}.jpg"
        fig.savefig(fig_path)
        plt.grid()
        plt.show()

    def add_tokens_to_bert_vocabulary(self, classifier):
        self.__compute_token_count()

        # select minimum n. occurrences at elbow to determine which are the most frequent tokens
        print(f"Min. n. occurrencies for kmers to be added as tokens to BERT vocabulary:")
        general_config.MIN_N_OCCUR_KMER = 0

        with open(paths_config.log_file, 'a') as log_fp:
            print(f"MIN_N_OCCUR_KMER = {general_config.MIN_N_OCCUR_KMER}")
            log_fp.write(f"Min. n. occurrencies for kmers to be added as tokens to BERT vocabulary:\n")
            log_fp.write(f"========================================================================\n")
            log_fp.write(f"MIN_N_OCCUR_KMER = {general_config.MIN_N_OCCUR_KMER}\n")

        if general_config.ADD_KMER_TOKENS_TO_VOCAB:
            # extract most frequent tokens based on min n. of occurrences
            frequent_tokens_in_train_data = [k for k, v in self.token_count.items() if v > general_config.MIN_N_OCCUR_KMER]
            print(
                f"Num. frequent (>{general_config.MIN_N_OCCUR_KMER} occurrences) tokens in train dataset: {len(frequent_tokens_in_train_data)}")

            # add most frequent tokens to BERT vocabulary
            print('Adding most frequent new tokens to Bert tokenizer...')
            num_added_toks = self.tokenizer.add_tokens(frequent_tokens_in_train_data)
            print(f'Num. of new tokens added: {num_added_toks}')
            with open(paths_config.log_file, 'a') as log_fp:
                log_fp.write(f"Num. of new frequent tokens added to BERT vocabulary: {num_added_toks}\n")
                log_fp.write('--------------------------------------------------------------\n')

            print('Done')

            return num_added_toks

