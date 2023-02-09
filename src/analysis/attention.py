import os
from pathlib import Path

from src.utils import paths_config, general_config
from src.utils.bio_config import domain_coordinates_1based
from src.utils.general_utils import get_inverted_class_labels_dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from mpl_toolkits.axes_grid1.colorbar import colorbar
import seaborn as sns


def convert_domain_coords_in_token_indices(domain):
    domain_coords_0based = np.asarray(domain_coordinates_1based[domain]) - 1
    domain_start_base_idx = domain_coords_0based[0] * 3
    domain_end_base_idx = domain_coords_0based[1] * 3 - 1

    domain_start_token_idx = None
    domain_end_token_idx = None

    bins = [[i * general_config.STRIDE - general_config.STRIDE, i * general_config.STRIDE - general_config.STRIDE + general_config.K - 1] for i in range(
        general_config.MAX_LENGTH)]
    bins[0][1] = -1 # cls

    for i,bin in enumerate(bins):
        if bin[0] <= domain_start_base_idx <= bin[1]:
            domain_start_token_idx = i
            break
    for i,bin in enumerate(bins):
        if bin[0] <= domain_end_base_idx <= bin[1]:
            domain_end_token_idx = i
    return domain_start_token_idx, domain_end_token_idx


def get_attentions(attentions, sample_idx_in_batch, layer=0, attention_head=0, sum_scores=False):
    '''
    get the particular output for a particular layer and attention head
    layer -> 0 to 11
    attention_head -> 0 to 11
    '''
    if sum_scores:
        # avg over all attention heads in a layer
        return attentions[layer][sample_idx_in_batch].sum(dim=0).cpu().detach().numpy()

    # return values for a particular attention head inside a specific layer
    return attentions[layer][sample_idx_in_batch][attention_head].cpu().detach().numpy()


def min_max_scale(X, range=(0, 1)):
    mi, ma = range
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled


def plt_attentions(mat, tick_labels, dir_path, filename="attention_matrix", theta=0, fig_size=(130, 100), annot=False,
                   cmap=sns.color_palette("viridis_r"), title=None):
    '''
    plot the NxN matrix passed as a heat map

    mat: square matrix to visualize
    tick_labels: labels for xticks and yticks (the tokens in our case)
    '''

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created")

    # percentile_threshold = np.percentile(mat, percentile)
    # mask_att_below_thresh = mat < mat.max() * threshold_ratio
    # mask_att_above_thresh = mat >= mat.max() * threshold_ratio
    mask_att_below_thresh = mat < theta
    mask_att_above_thresh = mat >= theta

    xs, ys = np.where(mask_att_above_thresh == True)
    high_attention_positions = []
    for x, y in zip(xs, ys):
        high_attention_positions.append([tick_labels[x], tick_labels[y], mat[x, y]])
    sorted_high_attention_positions = pd.DataFrame(high_attention_positions,
                                                   columns=['Row', 'Col', 'Attn. Score']).sort_values(
        by=['Attn. Score'], ascending=False)
    with open(f"{Path(dir_path) / filename}.txt", 'a') as fp:
        fp.write(title + '\n')
        fp.write(sorted_high_attention_positions.to_string(index=False))
        fp.write('\n')

    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(mat,
                     annot=annot,
                     yticklabels=tick_labels,
                     xticklabels=tick_labels,
                     cmap=cmap,
                     mask=mask_att_below_thresh)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    xtick_above_threshold = [np.any(mask_att_above_thresh[:, i]) for i in range(0, len(mask_att_above_thresh))]
    ytick_above_threshold = [np.any(mask_att_above_thresh[i, :]) for i in range(0, len(mask_att_above_thresh))]
    for (is_above_threshold, ticklbl) in zip(xtick_above_threshold, ax.xaxis.get_ticklabels()):
        if is_above_threshold:
            # ticklbl.set_weight("bold")
            ticklbl.set(color='black', backgroundcolor='yellow', weight='bold', alpha=0.5)
        # ticklbl.set_color('blue' if is_above_threshold else 'black')
        # ticklbl.set_backgroundcolor('blue' if is_above_threshold else '0')
        # ticklbl.set(color = 'white' if is_above_threshold else 'black', backgroundcolor='blue' if is_above_threshold else 'white')
    for (is_above_threshold, ticklbl) in zip(ytick_above_threshold, ax.yaxis.get_ticklabels()):
        if is_above_threshold:
            ticklbl.set(color='black', backgroundcolor='yellow', weight='bold', alpha=0.5)
            # ticklbl.set_weight("bold")
    if title:
        ax.set_title(title, fontsize=80)
    else:
        ax.set_title('Attention matrix', fontsize=80)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=80)

    # # padding_idx = n_tokens_per_seq + 2   # +2 to consider also 'CLS' and 'SEP' tokens
    # padding_idx = tick_labels.index(next(i for i in tick_labels if i.endswith('_[PAD]')))
    # ax.hlines([padding_idx], *ax.get_xlim(), linestyles='dashed')
    # ax.vlines([padding_idx], *ax.get_ylim(), linestyles='dashed')

    # plt.tight_layout()
    plt.show()

    # n_figures = len(os.listdir(dir_path))
    # if n_figures > 0:
    #     fig_path = Path(dir_path) / f'{filename}({n_figures}).jpg'
    # else:
    fig_path = Path(dir_path) / f'{filename}.jpg'
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)


def attention_analysis(attentions, log_fp_test, theta=0, selected_classes=None, selected_layers=None,
                       selected_heads=None):
    # token_base_positions_axis = attentions['token_base_positions_axis']
    # attentions_last_layer = attentions['attentions_last_layer']
    attentions_all_layers = attentions[
        'attentions_all_layers'] if 'attentions_all_layers' in attentions.keys() else None
    attentions_all_layers_thresh = attentions['attentions_all_layers_thresh']
    repr_token_base_positions_axis = attentions['repr_token_base_positions_axis']

    attn_last_layers_sum_dir = Path(paths_config.attention_matrices_dir) / "attention_maps"

    inv_class_labels_dict = get_inverted_class_labels_dict()

    for target_label in selected_classes:
        # percentile = None

        for layer in selected_layers:
            if selected_heads == 'avg':
                # calculate the avg of attention matrices of heads of current layer
                attn_mat = attentions_all_layers[target_label][layer].mean(dim=0).cpu().detach().numpy()
                plt_attentions(attn_mat,
                               repr_token_base_positions_axis[target_label],
                               attn_last_layers_sum_dir,
                               theta=theta,
                               filename=f"{inv_class_labels_dict[int(target_label)]}_{layer + 1}",
                               title=f"Average of attention matrices of heads of layer {layer + 1} for class '{inv_class_labels_dict[int(target_label)]}', theta = {theta:.3f}",
                               # cmap=colormaps_layers[layer]
                               )
            else:
                for head in selected_heads:
                    attn_mat = attentions_all_layers[target_label][layer][head].cpu().detach().numpy()
                    plt_attentions(attn_mat,
                                   repr_token_base_positions_axis[target_label],
                                   attn_last_layers_sum_dir,
                                   theta=theta,
                                   filename=f"{inv_class_labels_dict[int(target_label)]}_{layer + 1}_{head + 1}",
                                   title=f"Attention matrix of head {head + 1} of layer {layer + 1} for class '{inv_class_labels_dict[int(target_label)]}', theta = {theta:.3f}",
                                   # cmap=colormaps_layers[layer]
                                   )