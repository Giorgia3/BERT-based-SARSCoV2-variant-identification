import csv
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math

from matplotlib import pyplot as plt
from torch import nn

from src.utils import paths_config, general_config
from src.analysis.attention import get_attentions


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def loss_fn(outputs, targets):
    if len(general_config.CLASS_LABELS.keys()) > 2:
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs.logits, targets.to(torch.long))
    else:
        return outputs.loss


def mat_save_csv(mat, name, dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"Directory '{dirpath}' created")
    np.savetxt(Path(dirpath) / f'{name}.csv', mat, delimiter=",")


def compute_Von_Neumann_entropy_eigvals(mat, head_lay):
    eigvals = np.linalg.eigvals(mat)

    # print(f'{head_lay} {len(eigvals) - np.count_nonzero(eigvals)}/{len(eigvals)}')
    # print(f'{head_lay} {eigvals}')

    # dir_path = math_interpret_dir
    # fig = plt.figure(figsize=(8,8))
    # plt.hist(eigvals)
    # plt.title(f"Histogram of eigenvalues of the attention matrix for {head_lay}")
    # plt.grid(linestyle = '--')
    # plt.yticks(list(plt.yticks()[0]) + [1])
    # plt.show()
    # dit_path_head = Path(dir_path) / 'hist_sing_val' / head_lay
    # if not os.path.exists(dit_path_head):
    #     os.makedirs(dit_path_head)
    # fig_path = Path(dit_path_head) / f'head_lay.jpg'
    # fig.savefig(fig_path, bbox_inches='tight')
    # plt.close()

    root_sum_lambda_i = math.sqrt(sum([abs(lambda_i) ** 2 for lambda_i in eigvals]))
    # print(root_sum_lambda_i)
    norm_lambda = [abs(lambda_i) / root_sum_lambda_i for lambda_i in eigvals]
    # print(norm_lambda)
    return -sum(
        [abs(lambda_i_norm) * np.log(abs(lambda_i_norm)) for lambda_i_norm in norm_lambda if lambda_i_norm != float(0)])

    # root_sum_lambda_i = math.sqrt(sum([abs(lambda_i)**2 for lambda_i in eigvals]))
    # #print(root_sum_lambda_i)
    # norm_lambda = [lambda_i/root_sum_lambda_i for lambda_i in eigvals]
    # #print(norm_lambda)
    # return -sum([lambda_i_norm * np.log(lambda_i_norm) for lambda_i_norm in norm_lambda if lambda_i_norm!=float(0)])


def compute_Shannon_entropy(mat, head_lay):
    bins = pd.cut(mat.ravel(), bins=20)
    shannon_entropy = 0
    tot_count = len(mat.ravel())
    pjs = pd.value_counts(bins) / tot_count
    for pj in pjs:
        if pj != 0:
            shannon_entropy += pj * np.log(pj)

    return -shannon_entropy


def distance_cone(mat):
    # norm_per_row = np.linalg.norm(mat, axis=1)
    normalized_rows_mat = mat/np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    # norm_per_row = np.linalg.norm(norm, ord=2, axis=1, keepdims=True) #math.sqrt(sum([i**2 for i in mat[0]]))
    # print(f'NORMA RIGhe = {norm_per_row}')
    # norm = [i/norm_per_row for i in mat[0]]
    # print(norm)

    # #print(root_sum_lambda_i)
    # norm_lambda = [abs(lambda_i)/root_sum_lambda_i for lambda_i in eigvals]
    # # print(norm_per_row)
    # normalized_rows_mat = []
    # for row in range(mat.shape[0]):
    #     normalized_rows_mat.append(mat[row, :] / norm_per_row[row])
    summed_rows_array = np.sum(normalized_rows_mat, axis=0)
    norm_summed_rows_mat = np.linalg.norm(summed_rows_array)
    return norm_summed_rows_mat


def process_output_for_clustering(label_ids, logits, total_test_accuracy, seq_ids, positions, y_onehot, outputs, test_accuracies, final_data_test, count_class_samples):
    batch_accuracy = flat_accuracy(logits, label_ids)
    total_test_accuracy += batch_accuracy
    test_accuracies.append(batch_accuracy)

    final_data_test['seq_ids'].extend(seq_ids)
    final_data_test['positions'].extend(positions)
    final_data_test['targets'].extend(y_onehot.cpu().detach().numpy().tolist())
    final_data_test['logits'].extend(outputs.logits.cpu().detach().numpy().tolist())
    # final_data_test['outputs'].extend(torch.softmax(outputs.logits).cpu().detach().numpy().tolist())
    if len(general_config.CLASS_LABELS.keys()) > 2:
        final_data_test['outputs'].extend(
            torch.softmax(outputs.logits, dim=1).cpu().detach().numpy().tolist())
    else:
        final_data_test['outputs'].extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

    sel_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]
        count_class_samples[target_label] += 1
        if count_class_samples[target_label] <= 300:
            final_data_test['targets'].append(y_onehot[sample_idx].cpu().detach().numpy())
            # input_embeddings = outputs.hidden_states[0][sample_idx].cpu().detach().numpy()
            # output_embeddings = outputs.hidden_states[-1][sample_idx].cpu().detach().numpy()
            # final_data_test['input_embeddings'].append(input_embeddings)
            # final_data_test['output_embeddings'].append(output_embeddings)
            if len(general_config.CLASS_LABELS.keys()) > 2:
                final_data_test['outputs'].append(
                    torch.softmax(outputs.logits[sample_idx], dim=0).cpu().detach().numpy().tolist())
            else:
                final_data_test['outputs'].append(
                    torch.sigmoid(outputs.logits[sample_idx]).cpu().detach().numpy().tolist())

            CLS_embs_dict = {
                # "CLS_input_embeddings" : input_embeddings[0],
                # "CLS_Y_output_L12" : output_embeddings[0],
                "CLS_pre_softmax": logits[sample_idx],
                "CLS_final_output": torch.softmax(outputs.logits[sample_idx],
                                                  dim=0).cpu().detach().numpy().tolist()
            }
            for k, v in CLS_embs_dict.items():
                with open(Path(paths_config.CLS_embeddings_dir) / f"{k}.csv", 'a') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(v)

            for l in sel_layers:
                with open(Path(paths_config.CLS_embeddings_dir) / f"CLS_Y_output_L{l}.csv", 'a') as fp:
                    CLS_l_embeddings = outputs.hidden_states[l][sample_idx][0].cpu().detach().numpy()
                    writer = csv.writer(fp)
                    writer.writerow(CLS_l_embeddings)


def print_layer_output(seq_ids, label_ids, logits, outputs):
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]

        if target_label == general_config.CLASS_LABELS['omicron'] and target_label == np.argmax(
                logits[sample_idx]):  # correct prediction
            input_embs = outputs.hidden_states[0][
                sample_idx].cpu().detach().numpy()  # X = imput to current layer (output of previous layer)
            # NB: hidden_states of attn layers shifted by +1 because hidden_states[0] is embedding layer
            mat_save_csv(input_embs, f'input_embeddings', paths_config.math_interpret_dir)
            return


def process_output_for_attention_analysis(label_ids, logits, seq_ids, outputs, count_class_samples, ids, theta, attentions_all_layers, attentions_all_layers_thresh, repr_token_base_positions_axis, tokenizer):
    # get attention matrices
    # NB:   outputs.attentions is a tuple containing 12 elements, i.e. attentions of the 12 layers of Bert
    #       eg. if we consider first layer from the bottom (closest to the input):
    #           print(outputs.attentions[0].shape)
    #               -> torch.Size([4, 12, 512, 512]) -> [batch_size , num_heads, seq_len, seq_len]
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]

        if target_label == np.argmax(logits[sample_idx]):  # or not do_finetuning:  # correct prediction

            count_class_samples[target_label] += 1

            # get attention scores of sample from all heads in all layers
            sample_attentions = []
            sample_attentions_thresh = []
            for layer in range(general_config.N_LAYERS):
                layer_attentions = []
                masked_layer_attentions = []
                for head in range(general_config.N_HEADS):
                    # Source: "Bertology meets Biology...": We exclude attention to the [SEP] delimiter token, as it has been shown to
                    # be a “no-op” attention token (Clark et al., 2019), as well as attention to
                    # the [CLS] token, which is not explicitly used in language modeling.
                    # Additionally, exclude attention on [PAD] tokens.
                    head_attentions = get_attentions(outputs.attentions, sample_idx_in_batch=sample_idx,
                                                     layer=layer, attention_head=head, sum_scores=False)
                    sep_token_idx = ids[sample_idx].cpu().detach().numpy().tolist().index(next(
                        id for id in ids[sample_idx].cpu().detach().numpy().tolist() if
                        tokenizer.tokenizer.convert_ids_to_tokens(id) == "[SEP]"))
                    head_attentions[:, sep_token_idx] = 0  # SEP col
                    head_attentions[sep_token_idx, :] = 0  # SEP row
                    head_attentions[:, sep_token_idx:] = 0  # PAD cols
                    head_attentions[sep_token_idx:, :] = 0  # PAD rows
                    layer_attentions.append(head_attentions)
                    mask_att_below_thresh = head_attentions < theta
                    masked_head_attentions = np.ma.masked_array(head_attentions, mask=mask_att_below_thresh,
                                                                fill_value=0)
                    masked_layer_attentions.append(masked_head_attentions.filled())
                sample_attentions.append(layer_attentions)
                sample_attentions_thresh.append(masked_layer_attentions)

            # update sum of class attention scores of all heads of all layers
            if target_label not in attentions_all_layers:
                attentions_all_layers[target_label] = torch.zeros(np.asarray(sample_attentions).shape,
                                                                  dtype=torch.double)
                repr_token_base_positions_axis[target_label] = []
                for n, id in enumerate(ids[sample_idx].cpu().detach().numpy().tolist()):
                    if n == 0:
                        repr_token_base_positions_axis[target_label].append('[CLS]')
                    else:
                        start_p = int(((n * general_config.STRIDE - general_config.STRIDE) / 3) + 1)
                        end_p = int(start_p + (general_config.K / 3) - 1)
                        repr_token_base_positions_axis[target_label].append(f"{start_p}-{end_p}")
                        # repr_token_base_positions_axis [target_label] = [f"{n*general_config.STRIDE-general_config.STRIDE}_{tokenizer.convert_ids_to_tokens(id)}" for n,id in enumerate(ids[sample_idx].cpu().detach().numpy().tolist())]

            if target_label not in attentions_all_layers_thresh:
                attentions_all_layers_thresh[target_label] = torch.zeros(
                    np.asarray(sample_attentions_thresh).shape, dtype=torch.double)

            for layer in range(general_config.N_LAYERS):
                for head in range(general_config.N_HEADS):
                    attentions_all_layers[target_label][layer][head] = \
                        attentions_all_layers[target_label][layer][head] + np.asarray(
                            sample_attentions[layer][head])
                    attentions_all_layers_thresh[target_label][layer][head] = \
                        attentions_all_layers_thresh[target_label][layer][head] + np.asarray(
                            sample_attentions_thresh[layer][head])


def check_normality(seq_ids, label_ids, logits, outputs):
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]
        name_seq = seq_ids[sample_idx]
        if target_label == np.argmax(logits[sample_idx]):  # correct prediction
            X = outputs.hidden_states[0][
                sample_idx].cpu().detach().numpy()  # X = imput to current layer (output of previous layer)
            # NB: hidden_states of attn layers shifted by +1 because hidden_states[0] is embedding layer
            # token embeddings shape=(n_tokens_in_seq, emb_dim)=(512, 768)

            for i, row in enumerate(X):
                print(f'Mean row {i}: {np.mean(row)}')
        break


def process_output_for_distance_cones_analysis(label_ids, logits, seq_ids, outputs, model_params, distance_cones, model_config, count_layer, distance_cones_1_sample):
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]
        name_seq = seq_ids[sample_idx]
        if target_label == general_config.CLASS_LABELS['omicron'] and target_label == np.argmax(
                logits[sample_idx]):  # correct prediction

            for layer in set(np.asarray(general_config.SELECTED_LAYER_HEAD_LIST)[:, 0]):
                X = outputs.hidden_states[layer][
                    sample_idx].cpu().detach().numpy()  # X = imput to current layer (output of previous layer)
                # NB: hidden_states of attn layers shifted by +1 because hidden_states[0] is embedding layer
                # token embeddings shape=(n_tokens_in_seq, emb_dim)=(512, 768)
                # print(f'-------NORMA X[0]: {np.linalg.norm(X[0])}')
                rows_sum_X = np.sum(X, axis=0)
                # norm = np.linalg.norm(rows_sum_X, ord=2)
                # print(f'-------NORMA X: {norm}')
                # model_params = dict(model.named_parameters())
                # print(model_params["bert.encoder.layer.0.attention.self.value.weight"].data)
                W_v = model_params[
                    f"bert.encoder.layer.{layer}.attention.self.value.weight"].data.cpu().detach().numpy()
                W_q = model_params[
                    f"bert.encoder.layer.{layer}.attention.self.query.weight"].data.cpu().detach().numpy()
                W_k = model_params[
                    f"bert.encoder.layer.{layer}.attention.self.key.weight"].data.cpu().detach().numpy()
                emb_dim = X.shape[1]
                d_k = int(emb_dim / general_config.N_HEADS)
                multihead_output = None

                for head in np.sort([h for l, h in general_config.SELECTED_LAYER_HEAD_LIST if l == layer]):
                    W_v_i = W_v[:, head * d_k: head * d_k + d_k]
                    W_q_i = W_q[:, head * d_k: head * d_k + d_k]
                    W_k_i = W_k[:, head * d_k: head * d_k + d_k]
                    V_i = np.matmul(X, W_v_i)
                    Q_i = np.matmul(X, W_q_i)
                    K_i = np.matmul(X, W_k_i)

                    softmax_i = torch.softmax(torch.tensor(np.matmul(Q_i, K_i.T) / math.sqrt(d_k)),
                                              dim=1).cpu().detach().numpy()
                    output_head_i = np.matmul(softmax_i, V_i)

                    if head == 5 - 1:
                        output_head_5 = np.copy(output_head_i)

                    if multihead_output is None:
                        multihead_output = np.copy(output_head_i)
                    else:
                        multihead_output = np.hstack((multihead_output, output_head_i))

                layerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
                multihead_output = layerNorm(torch.from_numpy(X + multihead_output)).cpu().detach().numpy()

                layer_output = outputs.hidden_states[layer + 1][
                    sample_idx].cpu().detach().numpy()  # output of current layer

                #     print(f'Saving mat L{layer+1}')
                #     dirpath = Path(math_interpret_dir) / 'Y_outputs'
                #     mat_save_csv(multihead_output, f'Y_output_L{layer+1}', dirpath)

                # return final_data_test, test_accuracies, attentions, timings

                if layer not in distance_cones:
                    distance_cones[layer] = {}

                dc_X = distance_cone(X)
                # print(f'-------({name_seq}) DISTANCE CONE X: {dc_X}')
                if 'Layer input' not in distance_cones[layer]:
                    distance_cones[layer]['Layer input'] = []
                distance_cones[layer]['Layer input'].append(dc_X)

                dc_output_head_5 = distance_cone(output_head_5)
                # print(f'-------({name_seq}) DISTANCE CONE output_head_5: {dc_output_head_5}')
                if 'Head output' not in distance_cones[layer]:
                    distance_cones[layer]['Head output'] = []
                distance_cones[layer]['Head output'].append(dc_output_head_5)

                dc_multihead_output = distance_cone(multihead_output)
                # print(f'-------({name_seq}) DISTANCE CONE multihead_i: {dc_multihead_output}')
                if 'Multihead output' not in distance_cones[layer]:
                    distance_cones[layer]['Multihead output'] = []
                distance_cones[layer]['Multihead output'].append(dc_multihead_output)

                dc_layer_output = distance_cone(layer_output)
                # print(f'-------({name_seq}) DISTANCE CONE layer_output: {layer_output}')
                if 'Layer output' not in distance_cones[layer]:
                    distance_cones[layer]['Layer output'] = []
                distance_cones[layer]['Layer output'].append(dc_layer_output)

                if count_layer[layer] == 0:
                    if layer == 0:
                        distance_cones_1_sample[-1] = dc_X
                    distance_cones_1_sample[layer] = dc_layer_output
        break


def layer_output_analysis(seq_ids, logits, label_ids, outputs):
    output_layers_dir = Path(paths_config.math_interpret_dir) / "output_layer_histograms"
    if not os.path.exists(output_layers_dir):
        os.makedirs(output_layers_dir)
        print(f"Directory '{output_layers_dir}' created")

    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]
        name_seq = seq_ids[sample_idx]

        if target_label == general_config.CLASS_LABELS['omicron'] and target_label == np.argmax(
                logits[sample_idx]):  # correct prediction

            # output layer histogram:
            for layer in set(np.asarray(general_config.SELECTED_LAYER_HEAD_LIST)[:, 0]):
                layer_output = outputs.hidden_states[layer + 1][sample_idx].cpu().detach().numpy()
                layer_ouput_sum_rows = np.sum(layer_output, axis=0)
                fig = plt.figure(figsize=(8, 8))
                plt.hist(layer_ouput_sum_rows, bins=27)
                plt.title(f"L{layer + 1} output")
                plt.grid(linestyle='--')
                plt.yticks(list(plt.yticks()[0]) + [1])
                plt.show()
                fig_path = Path(output_layers_dir) / f'output_layer_{layer + 1}.jpg'
                fig.savefig(fig_path)

        return


def eigenvalues_analysis(seq_ids, label_ids, logits, outputs, model_params):
    layer_query_weight_names = [f'bert.encoder.layer.{layer}.attention.self.query.weight' for layer in
                                range(general_config.N_LAYERS)]
    layer_key_weight_names = [f'bert.encoder.layer.{layer}.attention.self.key.weight' for layer in
                              range(general_config.N_LAYERS)]
    layer_value_weight_names = [f'bert.encoder.layer.{layer}.attention.self.value.weight' for layer in
                                range(general_config.N_LAYERS)]
    for sample_idx in range(0, len(seq_ids)):
        target_label = label_ids[sample_idx]

        if target_label == general_config.CLASS_LABELS[general_config.SELECTED_CLASS] and target_label == np.argmax(
                logits[sample_idx]):  # correct prediction
            Qi_KiT_eigvals_file = Path(
                paths_config.math_interpret_dir) / "von_neumann_entropy_Qi_KiT.txt"
            symm_comp_eigvals_file = Path(
                paths_config.math_interpret_dir) / "von_neumann_entropy_symm_comp.txt"

            with open(Qi_KiT_eigvals_file, 'w') as Qi_KiT_eigvals_fp, open(symm_comp_eigvals_file,
                                                                           'w') as symm_comp_eigvals_fp:
                # get input embedding of sample for selected heads (shape=(general_config.N_LAYERS, batch_dim, n_tokens_in_seq, emb_dim))
                VN_entropy_Qi_KiT_layer_sum = np.zeros(general_config.N_LAYERS)
                VN_entropy_symm_comp_layer_sum = np.zeros(general_config.N_LAYERS)
                Sh_entropy_Qi_KiT_layer_sum = np.zeros(general_config.N_LAYERS)
                Sh_entropy_symm_comp_layer_sum = np.zeros(general_config.N_LAYERS)
                count_heads_layer = np.zeros(general_config.N_LAYERS)

                for layer, head in general_config.SELECTED_LAYER_HEAD_LIST:

                    X = outputs.hidden_states[layer + 1][
                        sample_idx].cpu().detach().numpy()  # token embeddings shape=(n_tokens_in_seq, emb_dim)=(512, 768)
                    emb_dim = X.shape[1]
                    d_k = int(emb_dim / general_config.N_HEADS)
                    W_q = model_params[layer_query_weight_names[layer]].data.cpu().detach().numpy()
                    W_k = model_params[layer_key_weight_names[layer]].data.cpu().detach().numpy()
                    W_v = model_params[layer_value_weight_names[layer]].data.cpu().detach().numpy()
                    W_q_i = W_q[:, head * d_k: head * d_k + d_k]
                    W_k_i = W_k[:, head * d_k: head * d_k + d_k]
                    W_v_i = W_v[:, head * d_k: head * d_k + d_k]

                    Q_i = np.matmul(X, W_q_i)
                    K_i = np.matmul(X, W_k_i)
                    V_i = np.matmul(X, W_v_i)

                    norm_Q_i = np.linalg.norm(Q_i, axis=1)
                    norm_K_i = np.linalg.norm(K_i, axis=1)
                    norm_V_i = np.linalg.norm(V_i, axis=1)

                    singular_values_W_q_i = abs(np.linalg.svd(W_q_i, compute_uv=False))
                    singular_values_W_k_i = abs(np.linalg.svd(W_k_i, compute_uv=False))
                    singular_values_W_v_i = abs(np.linalg.svd(W_v_i, compute_uv=False))
                    max_sv_W_q_i = max(singular_values_W_q_i)
                    max_sv_W_k_i = max(singular_values_W_k_i)
                    max_sv_W_v_i = max(singular_values_W_v_i)

                    softmax_i = torch.softmax(torch.tensor(np.matmul(Q_i, K_i.T) / math.sqrt(d_k)),
                                              dim=1).cpu().detach().numpy()
                    output_head_i = np.matmul(softmax_i, V_i)
                    norm_output_head_i = np.linalg.norm(output_head_i, axis=1)
                    singular_values_output_head_i = abs(np.linalg.svd(output_head_i, compute_uv=False))
                    max_sv_output_head_i = max(singular_values_output_head_i)

                    dit_path_head = Path(paths_config.math_interpret_dir) / 'norm_Qi_Ki_Vi' / f'{layer + 1}_{head + 1}'
                    if not os.path.exists(dit_path_head):
                        os.makedirs(dit_path_head)

                    # # histograms of singular values
                    # fig = plt.figure(figsize=(8,8))
                    # plt.hist(singular_values_W_k_i)
                    # plt.title(f"Histogram of singular values of W_k_i for head {head+1} in layer {layer+1}")
                    # plt.grid(linestyle = '--')
                    # plt.yticks(list(plt.yticks()[0]) + [1])
                    # plt.show()
                    # dit_path_head = Path(dir_path) / 'hist_sing_val' / f'{layer+1}_{head+1}'
                    # if not os.path.exists(dit_path_head):
                    #     os.makedirs(dit_path_head)
                    # fig_path = Path(dit_path_head) / f'{layer+1}_{head+1}.jpg'
                    # fig.savefig(fig_path, bbox_inches='tight')
                    # plt.close()

                    # # histograms of norms of Q_i
                    # fig,ax = plt.subplots(figsize=(8,8))
                    # plt.hist(norm_Q_i)
                    # plt.title(f"Histogram of the norms of Q_i for head {head+1} in layer {layer+1}\n(max SV of W_q_i: {max_sv_W_q_i})")
                    # plt.grid(linestyle = '--')
                    # ax.set_xlabel("Norm of Q_i")
                    # ax.set_ylabel("Frequency")
                    # plt.show()
                    # fig_path = Path(dit_path_head) / f'norm_Qi_{layer+1}_{head+1}.jpg'
                    # fig.savefig(fig_path, bbox_inches='tight')
                    # plt.close()

                    # # histograms of norms of K_i
                    # fig,ax = plt.subplots(figsize=(8,8))
                    # plt.hist(norm_K_i)
                    # plt.title(f"Histogram of the norms of K_i for head {head+1} in layer {layer+1}\n(max SV of W_k_i: {max_sv_W_k_i})")
                    # plt.grid(linestyle = '--')
                    # ax.set_xlabel("Norm of K_i")
                    # ax.set_ylabel("Frequency")
                    # plt.show()
                    # fig_path = Path(dit_path_head) / f'norm_Ki_{layer+1}_{head+1}.jpg'
                    # fig.savefig(fig_path, bbox_inches='tight')
                    # plt.close()

                    # # histograms of norms of V_i
                    # fig,ax = plt.subplots(figsize=(8,8))
                    # plt.hist(norm_V_i)
                    # plt.title(f"Histogram of the norms of V_i for head {head+1} in layer {layer+1}\n(max SV of W_v_i: {max_sv_W_v_i})")
                    # plt.grid(linestyle = '--')
                    # ax.set_xlabel("Norm of V_i")
                    # ax.set_ylabel("Frequency")
                    # plt.show()
                    # fig_path = Path(dit_path_head) / f'norm_Vi_{layer+1}_{head+1}.jpg'
                    # fig.savefig(fig_path, bbox_inches='tight')
                    # plt.close()

                    # # histograms of norms of output_head_i
                    # fig,ax = plt.subplots(figsize=(8,8))
                    # plt.hist(norm_output_head_i)
                    # plt.title(f"Histogram of the norms of Y_i output for head {head+1} in layer {layer+1}\n(max SV of Y_i: {max_sv_output_head_i})")
                    # plt.grid(linestyle = '--')
                    # ax.set_xlabel("Norm of Y_i")
                    # ax.set_ylabel("Frequency")
                    # plt.show()
                    # fig_path = Path(dit_path_head) / f'norm_Yi_{layer+1}_{head+1}.jpg'
                    # fig.savefig(fig_path, bbox_inches='tight')
                    # plt.close()

                    # histogram of singular values of output_head_i
                    fig, ax = plt.subplots(figsize=(8, 8))
                    plt.hist(singular_values_output_head_i)
                    plt.title(
                        f"Histogram of Singular Values of Y_i output for head {head + 1} in layer {layer + 1}\n(max SV of Y_i: {max_sv_output_head_i})")
                    plt.grid(linestyle='--')
                    ax.set_xlabel("Norm of Y_i")
                    ax.set_ylabel("Frequency")
                    plt.show()
                    fig_path = Path(dit_path_head) / f'sing_val_Yi_{layer + 1}_{head + 1}.jpg'
                    fig.savefig(fig_path, bbox_inches='tight')
                    plt.close()

                #     # Von Neumann norms
                #     count_heads_layer[layer] += 1
                #     # Eigenvalues of QK^T
                #     Qi_KiT = np.matmul(Q_i, K_i.T)
                #     eigvals_Qi_KiT = np.linalg.eigvals(Qi_KiT)
                #     n_eigvals_zero_Qi_KiT = f"{len(eigvals_Qi_KiT) - np.count_nonzero(eigvals_Qi_KiT)}/{len(eigvals_Qi_KiT)}"
                #     # von neumann entropy and shannon entropy
                #     VN_entropy_Qi_KiT = compute_Von_Neumann_entropy_eigvals(Qi_KiT, f"head{head+1}_layer_{layer+1}")
                #     Sh_entropy_Qi_KiT = compute_Shannon_entropy(Qi_KiT, f"head{head+1}_layer_{layer+1}")
                #     Qi_KiT_eigvals_fp.write(f"\thead_{head+1}_layer_{layer+1} | n_eigv_zero={n_eigvals_zero_Qi_KiT} | VN_entropy_={VN_entropy_Qi_KiT}\n")
                #     VN_entropy_Qi_KiT_layer_sum[layer] += VN_entropy_Qi_KiT
                #     Sh_entropy_Qi_KiT_layer_sum[layer] += Sh_entropy_Qi_KiT

                #     # Eigenvalues of symmetric component
                #     symm_comp_mat = (Qi_KiT + Qi_KiT.T) / 2
                #     eigvals_symm_comp_mat = np.linalg.eigvals(symm_comp_mat)
                #     n_eigvals_zero_symm_comp_mat = f"{len(eigvals_symm_comp_mat) - np.count_nonzero(eigvals_symm_comp_mat)}/{len(eigvals_symm_comp_mat)}"
                #     # von neumann entropy
                #     VN_entropy_symm_comp_mat = compute_Von_Neumann_entropy_eigvals(symm_comp_mat, f"head{head+1}_layer_{layer+1}")
                #     Sh_entropy_symm_comp_mat = compute_Shannon_entropy(symm_comp_mat, f"head{head+1}_layer_{layer+1}")
                #     symm_comp_eigvals_fp.write(f"\thead_{head+1}_layer_{layer+1} | n_eigv_zero={n_eigvals_zero_symm_comp_mat} | VN_entropy_head_{head+1}layer_{layer+1} = {VN_entropy_symm_comp_mat}\n")
                #     VN_entropy_symm_comp_layer_sum[layer] += VN_entropy_symm_comp_mat
                #     Sh_entropy_symm_comp_layer_sum[layer] += Sh_entropy_symm_comp_mat

                # VN_entropy_Qi_KiT_layer_mean = VN_entropy_Qi_KiT_layer_sum / count_heads_layer
                # VN_entropy_symm_comp_layer_mean = VN_entropy_symm_comp_layer_sum / count_heads_layer
                # Sh_entropy_Qi_KiT_layer_mean = Sh_entropy_Qi_KiT_layer_sum / count_heads_layer
                # Sh_entropy_symm_comp_layer_mean = Sh_entropy_symm_comp_layer_sum / count_heads_layer
                # fig = plt.figure(figsize=(10,10))
                # x = [i+1 for i in range(general_config.N_LAYERS)]
                # fig, ax = plt.subplots()
                # ax.plot(x, VN_entropy_Qi_KiT_layer_mean, 'b', label='VN entropy of QiKi^T')
                # ax.plot(x, VN_entropy_symm_comp_layer_mean, 'r', label='VN entropy of QiKi^T symm. comp.')
                # ax.axis('equal')
                # leg = ax.legend()
                # plt.title(f"Average Von Neumann entropy for each layer")
                # plt.grid(linestyle = '--')
                # ax.set_xticks(x)
                # #fig.yticks(list(plt.yticks()[0]) + [1])
                # ax.set_xlabel("Layer")
                # ax.set_ylabel("VN entropy")
                # plt.show()
                # fig_path = Path(dir_path) / f'VN_entropy_plot.jpg'
                # fig.savefig(fig_path, bbox_inches='tight')
                # plt.close()

                # fig = plt.figure(figsize=(10,10))
                # x = [i+1 for i in range(general_config.N_LAYERS)]
                # fig, ax = plt.subplots()
                # ax.plot(x, Sh_entropy_Qi_KiT_layer_mean, 'b', label='Shannon entropy of QiKi^T')
                # ax.plot(x, Sh_entropy_symm_comp_layer_mean, 'r', label='Shannon entropy of QiKi^T symm. comp.')
                # ax.axis('equal')
                # leg = ax.legend()
                # plt.title(f"Average Shannon entropy for each layer")
                # plt.grid(linestyle = '--')
                # ax.set_xticks(x)
                # #fig.yticks(list(plt.yticks()[0]) + [1])
                # ax.set_xlabel("Layer")
                # ax.set_ylabel("Shannon entropy")
                # plt.show()
                # fig_path = Path(dir_path) / f'Shannon_entropy_plot.jpg'
                # fig.savefig(fig_path, bbox_inches='tight')
                # plt.close()
            return


def distance_cones_analysis(distance_cones, distance_cones_1_sample):
    dist_cones_path = Path(paths_config.math_interpret_dir) / 'distance_cones'
    if not os.path.exists(dist_cones_path):
        os.makedirs(dist_cones_path)

    fig, ax = plt.subplots(figsize=((7, 4)))
    ax.plot(list(distance_cones_1_sample.keys()), list(distance_cones_1_sample.values()), '-o')
    ax.set_title('Cone index of one sequence of class Omicron')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Index")
    ax.set_xticks(list(distance_cones_1_sample.keys()))
    ax.set_xticklabels(np.asarray(list(distance_cones_1_sample.keys())) + 1)
    ax.grid()
    plt.show()
    fig_path = Path(dist_cones_path) / f'distance_cones_1_sample.jpg'
    fig.savefig(fig_path)
    fig.clear()

    for layer, layer_distance_cones in distance_cones.items():
        n_bins = 20
        distance_cones_1 = {k: v for k, v in layer_distance_cones.items() if
                            k in ['Layer input', 'Multihead output', 'Layer output']}
        fig, ax = plt.subplots(figsize=((7, 4)))
        ax.hist(distance_cones_1.values(), n_bins, histtype='step', stacked=True, fill=False,
                label=layer_distance_cones.keys())
        ax.set_title('L1 cone index')
        ax.legend(prop={'size': 10})
        ax.set_xlabel("Cone index")
        ax.set_ylabel("Frequency")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()
        fig_path = Path(dist_cones_path) / f'distance_cones_L{layer}_1.jpg'
        fig.savefig(fig_path)
        fig.clear()

        distance_cones_2 = {k: v for k, v in layer_distance_cones.items() if
                            k in ['Layer input', 'Head output']}
        fig, ax = plt.subplots(figsize=((7, 4)))
        ax.hist(distance_cones_2.values(), n_bins, histtype='step', stacked=True, fill=False,
                label=layer_distance_cones.keys())
        ax.set_title('L1 cone index')
        ax.legend(prop={'size': 10})
        ax.set_xlabel("Cone index")
        ax.set_ylabel("Frequency")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()
        fig_path = Path(dist_cones_path) / f'distance_cones_L{layer}_2.jpg'
        fig.savefig(fig_path)
        fig.clear()