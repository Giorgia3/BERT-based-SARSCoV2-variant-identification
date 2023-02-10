import json
from pathlib import Path

import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from src.utils.general_utils import get_inverted_class_labels_dict


def show_train_stats_and_plots(config, training_stats, train_steps_loss):
    print('Results:')
    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    fig = plt.figure(1, figsize=(12, 6))
    # plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Average Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()
    fig_path = Path(config.paths_config.outputs_dir) / 'trainval.jpg'
    fig.savefig(fig_path)

    fig2 = plt.figure(2, figsize=(30, 6))
    plt.plot(train_steps_loss, 'b', label="Training steps loss (every 40 steps)")

    # Label the plot.
    plt.title("Training steps loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    fig_path2 = Path(config.paths_config.outputs_dir) / 'train_steps.jpg'
    fig2.savefig(fig_path2)


def confusion_matrix_plot(config, targets_labels, outputs_labels, path, taskname=""):
    confusion_matr = metrics.confusion_matrix(targets_labels, outputs_labels)
    plt.figure()
    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matr, annot=True, fmt="d", cmap="viridis", annot_kws={"fontsize": 20})
    plt.title(f'{taskname} Confusion Matrix', fontsize=25)
    plt.ylabel('True label', fontsize=21)
    plt.xlabel('Predicted label', fontsize=21)
    curr_xticks, curr_xlabels = plt.xticks()
    inv_class_labels_dict = get_inverted_class_labels_dict(config)
    plt.xticks(curr_xticks, labels=[inv_class_labels_dict[int(t.get_text())] for t in curr_xlabels], rotation=90,
               fontsize=20)
    curr_yticks, curr_ylabels = plt.yticks()
    plt.yticks(curr_yticks, labels=[inv_class_labels_dict[int(t.get_text())] for t in curr_ylabels], rotation=0,
               fontsize=20)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    fig_path = Path(path) / f"{taskname.replace(' ', '_')}_confusion_matrix.jpg"
    fig = plt.gcf()
    fig.savefig(fig_path)
    plt.show()
    plt.figure()


def plot_PRC(config, y, y_score, y_pred, path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.set_title(f"Precision-Recall Curves")

    # metrics.PrecisionRecallDisplay.from_predictions(y, y_score, ax=ax, )
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in range(general_config.N_CLASSES):
    #     precision[i], recall[i], _ = metrics.precision_recall_curve(y[:],
    #                                                         y_score[:, i])
    #     average_precision[i] = metrics.average_precision_score(y[:], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    # precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y,
    #     y_score)
    # average_precision["micro"] = metrics.average_precision_score(y, y_score,
    #                                                     average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #     .format(average_precision["micro"]))
    # display = metrics.PrecisionRecallDisplay(
    #     recall=recall["micro"],
    #     precision=precision["micro"],
    #     average_precision=average_precision["micro"])
    # display.plot()
    # _ = display.ax_.set_title(f"Precision-Recall curve micro-averaged over all classes (AP={average_precision['micro']})")
    def multiclass_prc(config, y_test, y_score, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)

        for (idx, c_label) in enumerate(config.general_config.CLASS_LABELS.keys()):
            display = metrics.PrecisionRecallDisplay.from_predictions(y_test[:, idx].astype(int), y_pred[:, idx],
                                                                      name=c_label)
            display.plot(ax=ax)
        # ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
        return metrics.average_precision_score(y_test, y_pred, average=average)

    print('AP score:', multiclass_prc(config, y, y_score, y_pred))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # fig.show()
    fig.savefig(Path(path) / 'prc.png')


def ROC_curve_plot(config, targets_labels, outputs_labels, path, taskname="", figsize=(17, 6)):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels_dict = {class_label: [[], []] for class_label in config.general_config.CLASS_LABELS.values()}
    for i in range(len(targets_labels)):
        labels_dict[targets_labels[i]][0].append(targets_labels[i])
        labels_dict[targets_labels[i]][1].append(outputs_labels[i])

    for class_label, tgt_out_labels in labels_dict.items():
        print(class_label)
        print(tgt_out_labels[0])
        print(tgt_out_labels[1])
        fpr[class_label], tpr[class_label], _ = metrics.roc_curve(tgt_out_labels[0], tgt_out_labels[1],
                                                                  pos_label=class_label)
        roc_auc[class_label] = metrics.auc(fpr[class_label], tpr[class_label])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for class_label in labels_dict.keys():
        ax.plot(fpr[class_label], tpr[class_label],
                label='ROC curve (AUC = %0.2f) for class %i' % (roc_auc[class_label], class_label))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


def final_statistics(config, target_labels, output_labels, log_file, taskname, logits=None, target_names=None):
    accuracy = metrics.accuracy_score(target_labels, output_labels)
    f1_score_micro = metrics.f1_score(target_labels, output_labels, average='micro')
    f1_score_macro = metrics.f1_score(target_labels, output_labels, average='macro')
    if target_names == None:
        target_names = config.general_config.CLASS_LABELS
    specificity = {}
    for l_name, l in target_names.items():
        prec, recall, _, _ = metrics.precision_recall_fscore_support(target_labels == l,
                                                                     output_labels == l,
                                                                     pos_label=True, average=None)
        specificity[l_name] = recall[0]
    classification_report = metrics.classification_report(target_labels, output_labels,
                                                          target_names=target_names.keys())
    confusion_matr = metrics.confusion_matrix(target_labels, output_labels)
    print(f'{taskname}:')
    print(f"===============================")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"Specificity:\n{json.dumps(specificity, indent=4)}\n")
    print(f"Classification report:\n{str(classification_report)}")
    confusion_matrix_plot(config, target_labels, output_labels, config.paths_config.outputs_dir, taskname)
    # plot_PRC(target_labels, output_logits, output_labels, outputs_dir)
    with open(log_file, 'a') as log_fp:
        log_fp.write(f"\n{taskname}:\n")
        log_fp.write(f"===============================\n")
        log_fp.write(f"Accuracy Score = {accuracy}\n")
        log_fp.write(f"F1 Score (Micro) = {f1_score_micro}\n")
        log_fp.write(f"F1 Score (Macro) = {f1_score_macro}\n")
        log_fp.write(f"Classification report:\n{classification_report}\n")
        log_fp.write(f"Specificity:\n{json.dumps(specificity, indent=4)}\n")
        log_fp.write(f"Confusion matrix:\n{confusion_matr}\n")
    if logits:
        logits_pos_score = [x[target_labels[i]] for i, x in enumerate(logits)]
        ROC_curve_plot(config, target_labels, logits_pos_score, config.paths_config.outputs_dir, taskname)


def find_best_positions(config, final_data, min_score=0.8, threshold=0.5):
    best_positions = None

    outputs_labels = np.argmax(final_data['outputs'], axis=1)
    targets_labels = np.argmax(final_data['targets'], axis=1)

    # analisys according to position in sequence
    final_data['select_pred'] = []
    final_data['count_pred'] = []
    for target, pred, pred_score in zip(targets_labels, outputs_labels, final_data['outputs']):
        # select only predictions which are correct and with score > min_score (*)
        final_data['select_pred'].append(int(target == pred and pred_score[np.argmax(pred_score)] > min_score))
        final_data['count_pred'].append(1)
    final_data_df = pd.DataFrame(final_data).drop(columns=['outputs', 'targets', 'seq_ids'])
    grouped_df = final_data_df.groupby(['positions']).agg({'select_pred': 'sum', 'count_pred': 'count'})

    # for each position, calculate percentage of chunks which satisfy conditions (see (*))
    grouped_df['percents'] = grouped_df['select_pred'] / grouped_df['count_pred']
    # select only positions which have a percentage of chunks which satisfy conditions (see (*)) > threshold
    best_positions = [i for i, perc in enumerate(np.asarray(grouped_df['percents'])) if perc > threshold]

    print(f"Best positions: {best_positions}")
    with open(config.paths_config.log_file, 'a') as log_fp:
        log_fp.write(
            f"\nBest positions (i.e. those with >{threshold * 100}% of chunks with correct prediction and score>{min_score}):\n")
        log_fp.write(f"=======================================================================================\n")
        log_fp.write(f"{best_positions}\n")

    plot = grouped_df['percents'].plot.bar(figsize=(20, 10))
    fig = plot.get_figure()
    fig.suptitle(f"For each position, percentage of chunks with correct prediction and score>{min_score}", y=0.95,
                 fontsize=20)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid()
    plt.axhline(y=threshold, color='r', linestyle='-')
    fig_path = Path(config.paths_config.outputs_dir) / f"percent_pos.jpg"
    fig.savefig(fig_path)
    plt.show()

    return best_positions


def per_sample_result_computation(config, final_data, best_positions, min_score=0.8, taskname='', filter_positions=True,
                                  filter_score=True):
    targets_dict = {}
    preds_dict = {}
    for id, target in zip(final_data['seq_ids'], final_data['targets']):
        if id not in targets_dict:
            targets_dict[id] = np.argmax(target)
        if id not in preds_dict:
            preds_dict[id] = {'counts': [], 'outputs_label': [], 'prediction': []}

    final_data['select_pred'] = []
    final_data['outputs_label'] = []
    for pos, pred in zip(final_data['positions'], final_data['outputs']):
        # select prediction only if it is in best position and if prediction is certain (score>min_score) (*)
        if filter_positions:
            if filter_score:
                final_data['select_pred'].append(int(pos in best_positions and pred[np.argmax(pred)] > min_score))
            else:
                final_data['select_pred'].append(int(pos in best_positions))
        else:
            if filter_score:
                final_data['select_pred'].append(int(pred[np.argmax(pred)] > min_score))
            else:
                final_data['select_pred'].append(int(pred[np.argmax(pred)] > 0))
        final_data['outputs_label'].append(np.argmax(pred))

    final_data_df = pd.DataFrame(final_data)
    # filter only selected predictions according to conditions (see (*))
    filtered_data_df = final_data_df[final_data_df['select_pred'] > 0]
    filtered_data_df = filtered_data_df[['seq_ids', 'outputs_label']]
    grouped_sample_data_df = pd.DataFrame(filtered_data_df).groupby(['seq_ids', 'outputs_label']).size().reset_index(
        name='counts')
    grouped_sample_data_dict = grouped_sample_data_df.to_dict('list')

    for seq_id, output_l, count in zip(grouped_sample_data_dict['seq_ids'], grouped_sample_data_dict['outputs_label'],
                                       grouped_sample_data_dict['counts']):
        preds_dict[seq_id]['outputs_label'].append(output_l)
        preds_dict[seq_id]['counts'].append(count)

    for seq_id in preds_dict.keys():
        counts_sorted = preds_dict[seq_id]['counts'].copy()
        counts_sorted.sort(reverse=True)
        if len(preds_dict[seq_id]['counts']) == 0 or (
                len(counts_sorted) > 1 and all(element == counts_sorted[0] for element in counts_sorted)):
            preds_dict[seq_id]['prediction'] = ['uncertain']
        else:
            majority_class_index = np.argmax(preds_dict[seq_id]['counts'])
            preds_dict[seq_id]['prediction'] = preds_dict[seq_id]['outputs_label'][majority_class_index]

    correct_pred_count = 0
    uncertain_pred_count = 0
    tot_count = 0
    target_labels = []
    output_labels = []

    with open(config.paths_config.log_file, 'a') as log_fp:
        for seq_id in targets_dict.keys():
            tot_count += 1
            if preds_dict[seq_id]['prediction'] == ['uncertain']:
                uncertain_pred_count += 1
            else:
                target_labels.append(targets_dict[seq_id])
                output_labels.append(preds_dict[seq_id]['prediction'])
                if targets_dict[seq_id] == preds_dict[seq_id]['prediction']:
                    correct_pred_count += 1
            print(f"seq: {seq_id}\t target: {targets_dict[seq_id]}\t predicted: {preds_dict[seq_id]['prediction']}")
            log_fp.write(
                f"seq: {seq_id}\t target: {targets_dict[seq_id]}\t predicted: {preds_dict[seq_id]['prediction']}\n")

    final_statistics(config, target_labels, output_labels, config.paths_config.log_file,
                     f"{taskname} grouped by samples, filter_positions={filter_positions}, filter_score={filter_score}, min_score={min_score}")

    print(f"Correct predictions: {correct_pred_count}/{tot_count} -> {correct_pred_count / tot_count}")
    print(
        f"Wrong predictions: {tot_count - correct_pred_count - uncertain_pred_count}/{tot_count} -> {(tot_count - correct_pred_count - uncertain_pred_count) / tot_count}")
    print(f"Uncertain predictions: {uncertain_pred_count}/{tot_count} -> {uncertain_pred_count / tot_count}")

    with open(config.paths_config.log_file, 'a') as log_fp:
        log_fp.write(f"Correct predictions: {correct_pred_count}/{tot_count} -> {correct_pred_count / tot_count}\n")
        log_fp.write(
            f"Wrong predictions: {tot_count - correct_pred_count - uncertain_pred_count}/{tot_count} -> {(tot_count - correct_pred_count - uncertain_pred_count) / tot_count}\n")
        log_fp.write(
            f"Uncertain predictions: {uncertain_pred_count}/{tot_count} -> {uncertain_pred_count / tot_count}\n")


def show_test_plots(config, accuracies):
    # Create a barplot showing the accuracy score for each batch of test samples.
    fig = plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=list(range(len(accuracies))), y=accuracies, ci=None)

    plt.title('Accuracy per Batch')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch #')

    plt.show()
    fig_path = Path(config.paths_config.outputs_dir) / 'test.jpg'
    fig.savefig(fig_path)
