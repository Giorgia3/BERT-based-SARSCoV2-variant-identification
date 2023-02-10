import torch
from transformers import InputExample, InputFeatures
from torch.utils.data import Dataset
from src.classification.tokenizer import seq2kmer
from linecache import getline


def token_encode(config, seq, tokenizer):
    encoded_seq = {}

    #   (1) Tokenize the sentence.
    seq_tokens = seq2kmer(seq, config.general_config.K, config.general_config.STRIDE)

    #   (4) Map tokens to their IDs.
    seq_input_ids = []
    for token in seq_tokens:
        token_str = ''.join(token).lower()
        seq_input_ids.append(tokenizer.convert_tokens_to_ids(token_str))
        # print(f'{token_str} || {tokenizer.convert_tokens_to_ids(token_str)}')
    # print(seq_input_ids)

    #   (5) Truncate the sentence to `MAX_LENGTH`
    #       NB: -2 because we must consider also [CLS] and [SEP] tokens
    if len(seq_tokens) > config.general_config.MAX_LENGTH-2:
        seq_input_ids = seq_input_ids[:config.general_config.MAX_LENGTH - 2]

    #   (2) Prepend the `[CLS]` token to the start.
    seq_input_ids.insert(0, tokenizer.convert_tokens_to_ids('[CLS]'))

    #   (3) Append the `[SEP]` token to the end.
    seq_input_ids.append(tokenizer.convert_tokens_to_ids('[SEP]'))

    #   Pad if sentence is too short
    if len(seq_tokens) < config.general_config.MAX_LENGTH:
        length_seq = len(seq_input_ids)
        seq_input_ids.extend([tokenizer.convert_tokens_to_ids('[PAD]')] * (config.general_config.MAX_LENGTH - length_seq))
    # print(seq_tokens)
    # print(seq_input_ids)


    #   (6) Create attention masks for [PAD] tokens.
    seq_attention_mask = [int(id!=tokenizer.convert_tokens_to_ids('[PAD]')) for id in seq_input_ids] #map(lambda x: 0 if x==token_ids_dict['[PAD]'] else 1, seq_input_ids)
    # print(seq_attention_mask)
    # print()


    encoded_seq['input_ids'] = seq_input_ids
    encoded_seq['attention_mask'] = seq_attention_mask

    return encoded_seq


class DatasetGenerator(Dataset):

    def __init__(self, config, input_reader, input_fp, metadata, tokenizer):
        self.input_reader = input_reader
        self.metadata = metadata
        self.input_fp = input_fp
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.metadata['len']

    def __getitem__(self, idx):  # iterate lazily
        line = getline(self.input_fp.name, idx + 1).split(
            ',')  # idx+1 because getline considers 1 as index of first line in file, while idx starts from 0
        # line = next(itertools.islice(self.input_fp, idx, idx+1)).split(',')
        # print(line[:20])
        try:
            label = float(line[0])
        except Exception as e:
            print(f'line error: {line}')
            exit()
        seq_id = int(line[1])
        pos = int(line[2])
        seq = line[3]

        # Tokenize sequence and map the tokens to thier word IDs.
        example = InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=seq,
                               text_b=None,
                               label=label)

        encoded_seq = token_encode(self.config, example.text_a, self.tokenizer)

        features = InputFeatures(input_ids=encoded_seq['input_ids'],
                                 attention_mask=encoded_seq['attention_mask'],
                                 label=example.label)
        ids = features.input_ids
        mask = features.attention_mask

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(features.label, dtype=torch.float),
            'seq_ids': torch.tensor(seq_id, dtype=torch.int),
            'positions': torch.tensor(pos, dtype=torch.int)
        }


class DatasetGenerator_InputEmbeddings(Dataset):

    def __init__(self, config, input_fp, metadata, tokenizer):
        self.metadata = metadata
        self.input_fp = input_fp
        self.selected_variant = config.general_config.POSITIVE_CLASS_MLP
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.metadata['len']

    def __getitem__(self, idx):  # iterate lazily
        line = getline(self.input_fp.name, idx + 1).split(
            ',')  # idx+1 because getline considers 1 as index of first line in file, while idx starts from 0
        # line = next(itertools.islice(self.input_fp, idx, idx+1)).split(',')
        # print(line[:20])
        label_tmp = float(line[0])
        if label_tmp == self.config.general_config.CLASS_LABELS[self.selected_variant]:
            label = float(1)
        else:
            label = float(0)
        seq_id = int(line[1])
        pos = int(line[2])
        seq = line[3]

        # Tokenize sequence and map the tokens to thier word IDs.
        example = InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=seq,
                               text_b=None,
                               label=label)

        encoded_seq = token_encode(self.config, example.text_a, self.tokenizer)

        features = InputFeatures(input_ids=encoded_seq['input_ids'],
                                 attention_mask=encoded_seq['attention_mask'],
                                 label=example.label)
        ids = features.input_ids
        mask = features.attention_mask

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(features.label, dtype=torch.float),
            'seq_ids': torch.tensor(seq_id, dtype=torch.int),
            'positions': torch.tensor(pos, dtype=torch.int)
        }
