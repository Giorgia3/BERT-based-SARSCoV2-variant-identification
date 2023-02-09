import os
import pickle
from pathlib import Path

from src.utils import paths_config, general_config
from src.analysis.attention import attention_analysis
from src.utils.general_utils import setup_config

if __name__ == '__main__':
    setup_config()

    attention_matrices_file = Path(paths_config.attention_matrices_dir) / 'attention_matrices_np'
    attention_matrices_thresh_file = Path(paths_config.attention_matrices_dir) / 'attention_matrices_thresh_np'
    ticks_file = Path(paths_config.attention_matrices_dir) / 'ticks_np'
    attentions = {}

    if os.path.exists(attention_matrices_file):
        attentions['attentions_all_layers'] = pickle.load(open(attention_matrices_file, 'rb'))
    else:
        raise FileNotFoundError(f"Error: File not found: {attention_matrices_file}")

    if os.path.exists(attention_matrices_thresh_file):
        attentions['attentions_all_layers_thresh'] = pickle.load(open(attention_matrices_thresh_file, 'rb'))
    else:
        raise FileNotFoundError(f"Error: File not found: {attention_matrices_thresh_file}")

    if os.path.exists(ticks_file):
        attentions['repr_token_base_positions_axis'] = pickle.load(open(ticks_file, 'rb'))
    else:
        raise FileNotFoundError(f"Error: File not found: {ticks_file}")

    with open(paths_config.test_file) as test_fp, open(paths_config.log_file, 'a') as log_fp:
        selected_classes = range(len(general_config.CLASS_LABELS.keys()))  # [1]  #range(general_config.N_CLASSES)
        selected_layers = range(general_config.N_LAYERS)  # range(n_layers) #[10] #range(n_layers)
        selected_heads = range(general_config.N_HEADS)  # range(n_heads) #[4] #range(n_heads) #'avg'
        theta_plot = 0.01
        attention_analysis(attentions, log_fp, theta_plot, selected_classes, selected_layers, selected_heads)
