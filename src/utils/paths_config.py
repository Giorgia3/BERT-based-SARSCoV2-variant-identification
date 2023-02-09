import os
from pathlib import Path
import yaml

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created")
    return dir_path

with open(Path(".") / "config" / "paths_config.yaml") as paths_config_fp:
    paths_config_dict = yaml.safe_load(paths_config_fp)

main_dir = paths_config_dict['main_dir']

bwa_bin_dir = paths_config_dict['bwa_bin_dir']

# Input data:
datasets_dir = paths_config_dict['datasets_dir']

variant_files = {}
for filename in os.listdir(Path(datasets_dir) / 'original'):
    variant_files[os.path.splitext(filename)[0]] = Path(datasets_dir) / 'original' / str(filename)
reference_seq_file = Path(datasets_dir) / 'reference.fasta'
if not os.path.exists(reference_seq_file):
    raise FileNotFoundError(f'File {reference_seq_file} not found in {datasets_dir} folder.')

# Experiment dir:
experiment_dir = Path(main_dir) / f"experiment"

# Preprocessed data:
preprocessed_data_dir = make_dir(Path(experiment_dir) / 'preprocessed_data')
reformatted_seqs_file = Path(preprocessed_data_dir) / 'reformatted_seqs.fasta'
duplicated_seqs_file = Path(preprocessed_data_dir) / 'duplicated_seqs.fasta'
aligned_seqs_file = Path(preprocessed_data_dir) / 'aligned.sam'
spike_seqs_file = Path(preprocessed_data_dir) / 'spike_seqs.csv'
cigars_file = Path(preprocessed_data_dir) / 'cigars.csv'
input_seqs_file = spike_seqs_file
seqs_index_file_tmp = Path(preprocessed_data_dir) / 'seqs_index_tmp.csv'
seqs_index_file = Path(preprocessed_data_dir) / 'seqs_index.csv'
ids_dict_file = Path(preprocessed_data_dir) / 'ids_dict.csv'
trainvaltest_splits_dir = make_dir(Path(preprocessed_data_dir) / 'trainvaltest_splits')
# train_file = Path(trainvaltest_splits_dir) / 'train.csv'
# val_file = Path(trainvaltest_splits_dir) / 'val.csv'
# test_file = Path(trainvaltest_splits_dir) / 'test.csv'
test_file = input_seqs_file
trainvaltest_sizes_file = Path(trainvaltest_splits_dir) / 'trainvaltest_sizes.csv'
token_count_file = Path(preprocessed_data_dir) / 'token_count.csv'

# Models dir
model_file_finetuned = Path(main_dir) / 'models' / 'model_finetuned'
models_dir = Path(main_dir) / 'models'

# Outputs dir:
outputs_dir = make_dir(Path(experiment_dir) / 'outputs')
log_dir = make_dir(Path(outputs_dir) / 'log')
tokens_histograms_dir = make_dir(Path(outputs_dir) / 'tokens_histograms')
final_val_outputs_file = Path(outputs_dir) / 'final_val_outputs'
final_test_outputs_file = Path(outputs_dir) / 'final_test_outputs'
model_file = Path(outputs_dir) / 'model'
training_stats_file = Path(outputs_dir) / 'training_stats'
train_steps_loss_file = Path(outputs_dir) / 'train_steps_loss'
test_accuracies_file = Path(outputs_dir) / 'test_accuracies'
final_data_test_file = Path(outputs_dir) / 'final_data_test_file'

# attention matrices
attention_matrices_dir = make_dir(Path(outputs_dir) / 'attention_analysis')
proportion_attn_domains_dir = make_dir(Path(attention_matrices_dir) / "proportion_attn_domains")

# mathematical interpretation outputs
math_interpret_dir = make_dir(Path(outputs_dir) / "mathematical_interpretation")
CLS_embeddings_dir = make_dir(Path(math_interpret_dir) / "Y_outputs" / "CLS_embeddings")

# biological interpretation output
log_dir_bio = make_dir(Path(outputs_dir) / "biological_interpretation" / "log")

# clustering output
clustering_dir = make_dir(Path(outputs_dir) / "clustering")

# log
n_log_files = len(os.listdir(log_dir))
if n_log_files > 0:
    log_file = Path(log_dir) / f'log({n_log_files}).txt'
else:
    log_file = Path(log_dir) / f'log.txt'


def init():
    pass
