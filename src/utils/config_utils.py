import os
from pathlib import Path
import yaml
import math

from src.utils.general_utils import make_dir


class GeneralConfig:
    def __init__(self):
        with open(Path(".") / "config" / "general_config.yaml") as general_config_fp:
            general_config_dict = yaml.safe_load(general_config_fp)

        self.USE_GPU = general_config_dict['USE_GPU']
        self.DO_TRAINING = general_config_dict['DO_TRAINING']
        self.DO_TEST = general_config_dict['DO_TEST']
        self.DO_FINETUNING = general_config_dict['DO_FINETUNING']
        self.TASK_TYPE = general_config_dict['TASK_TYPE']
        self.SPIKE_REGION_ANALYSIS = general_config_dict['SPIKE_REGION_ANALYSIS']

        self.CLASS_LABELS = general_config_dict['CLASS_LABELS']

        self.REDUCE_N_INPUT_SAMPLES = general_config_dict['REDUCE_N_INPUT_SAMPLES']
        self.MAX_N_SAMPLES_PER_CLASS = general_config_dict['MAX_N_SAMPLES_PER_CLASS']

        self.K = general_config_dict['K']
        self.STRIDE = general_config_dict['STRIDE']
        self.ADD_KMER_TOKENS_TO_VOCAB = general_config_dict['ADD_KMER_TOKENS_TO_VOCAB']
        self.MIN_N_OCCUR_KMER = general_config_dict['MIN_N_OCCUR_KMER']

        self.ALIGNMENT = general_config_dict['ALIGNMENT']
        self.SPLIT_DATA_IN_CHUNKS = general_config_dict['SPLIT_DATA_IN_CHUNKS']
        if self.SPLIT_DATA_IN_CHUNKS or self.ALIGNMENT:
            self.CHUNK_LEN = general_config_dict['CHUNK_LEN']
            self.CHUNK_STRIDE = general_config_dict['CHUNK_STRIDE']

        # maximum n. of tokens to be considered for each sequence: (CHUNK_LEN-K)/STRIDE +1 (max value supported by Bert-Base: 512)
        if self.SPLIT_DATA_IN_CHUNKS:
            self.MAX_LENGTH = math.ceil(('CHUNK_LEN' - self.K) / self.STRIDE + 1)
        else:
            self.MAX_LENGTH = general_config_dict['MAX_LENGTH']

        self.N_LAYERS = general_config_dict['N_LAYERS']
        self.N_HEADS = general_config_dict['N_HEADS']
        self.TRAIN_BATCH_SIZE = general_config_dict['TRAIN_BATCH_SIZE']
        self.EVAL_BATCH_SIZE = general_config_dict['EVAL_BATCH_SIZE']
        self.EPOCHS = general_config_dict['EPOCHS']
        self.LR = general_config_dict['LR']

        # attention threshold
        self.THETA = general_config_dict['THETA']

        # WEA
        all_layers_heads = []
        for head in range(0, self.N_HEADS):
            for layer in range(0, self.N_LAYERS):
                all_layers_heads.append([layer, head])
        if self.TASK_TYPE == 'distance_cones_analysis':
            all_layers_heads1 = []
            for head in range(0, self.N_HEADS):
                for layer in range(0, self.N_LAYERS):
                    all_layers_heads1.append([layer, head])
            dc_layer_heads = all_layers_heads1
            # dc_layer_heads = []
            # for head in range(0,12):
            #     dc_layer_heads.append([1-1, head])
            self.SELECTED_LAYER_HEAD_LIST = dc_layer_heads  # distance cones:[[1-1,5-1]]
        else:
            self.SELECTED_LAYER_HEAD_LIST = all_layers_heads  # [[1-1,4-1], [12-1,9-1], [11-1, 5-1], [1-1, 5-1], [5-1, 10-1]] #all_layers_heads
        self.SELECTED_CLASS = general_config_dict['SELECTED_CLASS']

class BioConfig:
    def __init__(self, k, stride):
        with open(Path(".") / "config" / "bio_config.yaml") as bio_config_fp:
            bio_config_dict = yaml.safe_load(bio_config_fp)

        self.spike_gene_start = bio_config_dict['spike_gene_start'] - 1
        self.spike_gene_end = bio_config_dict['spike_gene_end'] - 1

        self.domain_coordinates_1based = bio_config_dict['domain_coordinates_1based']

        self.n_tokens_per_seq = math.ceil(
            (((self.spike_gene_end - self.spike_gene_start) - k) / stride) + 1)

class PathsConfig:
    def __init__(self, datasets_dir, bwa_bin_dir=None):
        with open(Path(".") / "config" / "paths_config.yaml") as paths_config_fp:
            paths_config_dict = yaml.safe_load(paths_config_fp)

        self.main_dir = paths_config_dict['main_dir']

        self.bwa_bin_dir = bwa_bin_dir

        # Input data:
        self.datasets_dir = datasets_dir

        self.variant_files = {}
        for filename in os.listdir(Path(self.datasets_dir) / 'original'):
            self.variant_files[os.path.splitext(filename)[0]] = Path(self.datasets_dir) / 'original' / str(filename)
        self.reference_seq_file = Path(self.datasets_dir) / 'reference.fasta'
        if not os.path.exists(self.reference_seq_file):
            raise FileNotFoundError(f'File {self.reference_seq_file} not found in {self.datasets_dir} folder.')

        # Experiment dir:
        self.experiment_dir = Path(self.main_dir) / f"experiment"

        # Preprocessed data:
        self.preprocessed_data_dir = make_dir(Path(self.experiment_dir) / 'preprocessed_data')
        self.reformatted_seqs_file = Path(self.preprocessed_data_dir) / 'reformatted_seqs.fasta'
        self.duplicated_seqs_file = Path(self.preprocessed_data_dir) / 'duplicated_seqs.fasta'
        self.aligned_seqs_file = Path(self.preprocessed_data_dir) / 'aligned.sam'
        self.spike_seqs_file = Path(self.preprocessed_data_dir) / 'spike_seqs.csv'
        self.cigars_file = Path(self.preprocessed_data_dir) / 'cigars.csv'
        self.input_seqs_file = self.spike_seqs_file
        self.seqs_index_file_tmp = Path(self.preprocessed_data_dir) / 'seqs_index_tmp.csv'
        self.seqs_index_file = Path(self.preprocessed_data_dir) / 'seqs_index.csv'
        self.ids_dict_file = Path(self.preprocessed_data_dir) / 'ids_dict.csv'
        self.trainvaltest_splits_dir = make_dir(Path(self.preprocessed_data_dir) / 'trainvaltest_splits')
        self.test_file = self.input_seqs_file
        self.trainvaltest_sizes_file = Path(self.trainvaltest_splits_dir) / 'trainvaltest_sizes.csv'
        self.token_count_file = Path(self.preprocessed_data_dir) / 'token_count.csv'

        # Models dir
        self.model_file_finetuned = Path(self.main_dir) / 'models' / 'model_finetuned'
        self.models_dir = Path(self.main_dir) / 'models'

        # Outputs dir:
        self.outputs_dir = make_dir(Path(self.experiment_dir) / 'outputs')
        self.log_dir = make_dir(Path(self.outputs_dir) / 'log')
        self.tokens_histograms_dir = make_dir(Path(self.outputs_dir) / 'tokens_histograms')
        self.final_val_outputs_file = Path(self.outputs_dir) / 'final_val_outputs'
        self.final_test_outputs_file = Path(self.outputs_dir) / 'final_test_outputs'
        self.model_file = Path(self.outputs_dir) / 'model'
        self.training_stats_file = Path(self.outputs_dir) / 'training_stats'
        self.train_steps_loss_file = Path(self.outputs_dir) / 'train_steps_loss'
        self.test_accuracies_file = Path(self.outputs_dir) / 'test_accuracies'
        self.final_data_test_file = Path(self.outputs_dir) / 'final_data_test_file'

        # attention matrices
        self.attention_matrices_dir = make_dir(Path(self.outputs_dir) / 'attention_analysis')
        self.proportion_attn_domains_dir = make_dir(Path(self.attention_matrices_dir) / "proportion_attn_domains")

        # mathematical interpretation outputs
        self.math_interpret_dir = make_dir(Path(self.outputs_dir) / "mathematical_interpretation")
        self.CLS_embeddings_dir = make_dir(Path(self.math_interpret_dir) / "Y_outputs" / "CLS_embeddings")

        # biological interpretation output
        self.log_dir_bio = make_dir(Path(self.outputs_dir) / "biological_interpretation" / "log")

        # clustering output
        self.clustering_dir = make_dir(Path(self.outputs_dir) / "clustering")

        # log
        n_log_files = len(os.listdir(self.log_dir))
        if n_log_files > 0:
            self.log_file = Path(self.log_dir) / f'log({n_log_files}).txt'
        else:
            self.log_file = Path(self.log_dir) / f'log.txt'

class Config:
    def __init__(self, datasets_dir, bwa_bin_dir=None):
        self.paths_config = PathsConfig(datasets_dir, bwa_bin_dir)
        self.general_config = GeneralConfig()
        self.bio_config = BioConfig(self.general_config.K, self.general_config.STRIDE)