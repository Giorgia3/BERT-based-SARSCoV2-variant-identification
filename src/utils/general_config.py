import math
import yaml
from pathlib import Path

with open(Path(".") / "config" / "general_config.yaml") as general_config_fp:
    general_config_dict = yaml.safe_load(general_config_fp)

print(general_config_dict)

USE_GPU = general_config_dict['USE_GPU']
DO_TRAINING = general_config_dict['DO_TRAINING']
DO_TEST = general_config_dict['DO_TEST']
DO_FINETUNING = general_config_dict['DO_FINETUNING']
TASK_TYPE = general_config_dict['TASK_TYPE']
SPIKE_REGION_ANALYSIS = general_config_dict['SPIKE_REGION_ANALYSIS']

CLASS_LABELS = general_config_dict['CLASS_LABELS']

REDUCE_N_INPUT_SAMPLES = general_config_dict['REDUCE_N_INPUT_SAMPLES']
MAX_N_SAMPLES_PER_CLASS = general_config_dict['MAX_N_SAMPLES_PER_CLASS']

K = general_config_dict['K']
STRIDE = general_config_dict['STRIDE']
ADD_KMER_TOKENS_TO_VOCAB = general_config_dict['ADD_KMER_TOKENS_TO_VOCAB']
MIN_N_OCCUR_KMER = general_config_dict['MIN_N_OCCUR_KMER']

ALIGNMENT = general_config_dict['ALIGNMENT']
SPLIT_DATA_IN_CHUNKS = general_config_dict['SPLIT_DATA_IN_CHUNKS']
if SPLIT_DATA_IN_CHUNKS or ALIGNMENT:
    CHUNK_LEN = general_config_dict['CHUNK_LEN']
    CHUNK_STRIDE = general_config_dict['CHUNK_STRIDE']

# maximum n. of tokens to be considered for each sequence: (CHUNK_LEN-K)/STRIDE +1 (max value supported by Bert-Base: 512)
if SPLIT_DATA_IN_CHUNKS:
    MAX_LENGTH = math.ceil(('CHUNK_LEN' - K) / STRIDE + 1)
else:
    MAX_LENGTH = general_config_dict['MAX_LENGTH']

N_LAYERS = general_config_dict['N_LAYERS']
N_HEADS = general_config_dict['N_HEADS']
TRAIN_BATCH_SIZE = general_config_dict['TRAIN_BATCH_SIZE']
EVAL_BATCH_SIZE = general_config_dict['EVAL_BATCH_SIZE']
EPOCHS = general_config_dict['EPOCHS']
LR = general_config_dict['LR']

# attention threshold
THETA = general_config_dict['THETA']

# WEA
all_layers_heads = []
for head in range(0, N_HEADS):
    for layer in range(0, N_LAYERS):
        all_layers_heads.append([layer, head])
if TASK_TYPE == 'distance_cones_analysis':
    all_layers_heads1 = []
    for head in range(0, N_HEADS):
        for layer in range(0, N_LAYERS):
            all_layers_heads1.append([layer, head])
    dc_layer_heads = all_layers_heads1
    # dc_layer_heads = []
    # for head in range(0,12):
    #     dc_layer_heads.append([1-1, head])
    SELECTED_LAYER_HEAD_LIST = dc_layer_heads  # distance cones:[[1-1,5-1]]
else:
    SELECTED_LAYER_HEAD_LIST = all_layers_heads  # [[1-1,4-1], [12-1,9-1], [11-1, 5-1], [1-1, 5-1], [5-1, 10-1]] #all_layers_heads
SELECTED_CLASS = general_config_dict['SELECTED_CLASS']


def init():
    i = 0
    # for variant_name in variant_files:
    #     CLASS_LABELS[variant_name] = i
    #     i += 1


