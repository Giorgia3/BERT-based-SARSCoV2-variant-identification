import csv
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import tensorflow as tf
from src.utils import paths_config, general_config
import gc
import time

from src.classification import report
from src.classification.report import show_test_plots, final_statistics
from src.classification.tokenizer import Tokenizer
from src.classification.utils import format_time, flat_accuracy, loss_fn, process_output_for_clustering, \
    print_layer_output, process_output_for_attention_analysis, check_normality, \
    process_output_for_distance_cones_analysis, layer_output_analysis, eigenvalues_analysis, \
    distance_cones_analysis
from src.preprocessing.data_generator import DatasetGenerator
from src.utils.general_utils import get_train_val_test_sizes


class Classifier:
    def __init__(self):
        # self.tokenizer = Tokenizer()
        # self.tokenizer.plot_kmers_histogram()
        # self.size_token_embeddings = self.tokenizer.add_tokens_to_bert_vocabulary()
        self.tokenizer = Tokenizer(paths_config.models_dir)
        self.size_token_embeddings = len(self.tokenizer)
        self.sizes_info = get_train_val_test_sizes()

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=len(general_config.CLASS_LABELS.keys()),
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )

        self.device = None
        if general_config.USE_GPU and (
                general_config.DO_TRAINING or general_config.DO_TEST or general_config.TASK_TYPE == 'one_vs_all_classification'):
            # Get the GPU device name.
            device_name = tf.test.gpu_device_name()

            # The device name should look like the following:
            if device_name == '/device:GPU:0':
                print('Found GPU at: {}'.format(device_name))
            # else:
                # raise SystemError('GPU device not found')

            # If there's a GPU available...
            if torch.cuda.is_available():
                # Tell PyTorch to use the GPU.
                self.device = torch.device("cuda")
                print('There are %d GPU(s) available.' % torch.cuda.device_count())
                print('We will use the GPU:', torch.cuda.get_device_name(0))
            # If not...
            else:
                print('No GPU available, using the CPU instead.')
                self.device = torch.device("cpu")
        else:
            print('Runtime type: None.')

        print('Resizing token embeddings of BERT...')
        self.model.resize_token_embeddings(self.size_token_embeddings)

        # Tell pytorch to run this model on the GPU.
        if self.device.type == 'cuda':
            self.model.cuda()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)


class Trainer(Classifier):
    def __init__(self):
        super().__init__()

    def __setup_training(self, train_data_size, log_fp):
        # Set up epochs and steps
        steps_per_epoch = int(train_data_size / general_config.TRAIN_BATCH_SIZE)
        num_train_steps = steps_per_epoch * general_config.EPOCHS
        warmup_steps = int(general_config.EPOCHS * train_data_size * 0.1 / general_config.TRAIN_BATCH_SIZE)

        # creates an optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=general_config.LR,
                                      # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_steps)
        # optimizer = nlp.optimization.create_optimizer(
        #     2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        log_fp.write("\nTraining setup:\n")
        log_fp.write("===============\n")
        log_fp.write(f"\tsteps_per_epoch = {steps_per_epoch}\n")
        log_fp.write(f"\tnum_train_steps = {num_train_steps}\n")
        log_fp.write(f"\twarmup_steps = {warmup_steps}\n")
        log_fp.write(f"\toptimizer = {optimizer}\n")
        log_fp.write(f"\tscheduler = {scheduler}\n")
        log_fp.write(f"\tseed_val = {seed_val}\n")

        return optimizer, scheduler

    def __train(self, train_dataloader, optimizer, scheduler, log_fp):
        # Perform one full pass over the training set.

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        train_steps_loss = []

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        self.model.train()

        for step, batch in enumerate(train_dataloader, 0):

            # Unpack this training batch from our dataloader.
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            ids = batch['ids'].to(self.device, dtype=torch.long)
            mask = batch['mask'].to(self.device, dtype=torch.long)
            targets = batch['targets'].to(self.device, dtype=torch.float)
            seq_ids = batch['seq_ids'].to(self.device, dtype=torch.int)
            y_onehot = torch.nn.functional.one_hot(targets.long(), num_classes=len(general_config.CLASS_LABELS.keys()))
            y_onehot = y_onehot.float()

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            self.model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            outputs = self.model(ids,
                                 token_type_ids=None,
                                 attention_mask=mask,
                                 labels=y_onehot,
                                 return_dict=True)

            loss = loss_fn(outputs, targets)

            # Save hidden states of last layer for clustering
            # hidden_states = Tuple of torch.FloatTensor (one for the output of the embeddings
            # + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            # last_hidden_states = outputs.hidden_states[12]

            # Progress update every 40 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5}  of  {:>5}.    Elapsed: {:}.     Loss:  {:}.'.format(step, len(train_dataloader),
                                                                                          elapsed, loss.item()))
                log_fp.write(('  Batch {:>5}  of  {:>5}.    Elapsed: {:}.     Loss:  {:}.\n'.format(step,
                                                                                                    len(train_dataloader),
                                                                                                    elapsed,
                                                                                                    loss.item())))
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += float(loss.item())
            if step % 50 == 0 and not step == 0:
                train_steps_loss.append(float(loss.item()))

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        log_fp.write("  Average training loss: {0:.2f}\n".format(avg_train_loss))
        log_fp.write("  Training epoch took: {:}\n".format(training_time))

        return avg_train_loss, training_time, train_steps_loss

    def __validate(self, val_dataloader, log_fp):
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        final_data = {'seq_ids': [],
                      'positions': [],
                      'targets': [],
                      'outputs': [],
                      'last_hidden_states_dict': {}
                      }

        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Unpack this training batch from our dataloader.
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            ids = batch['ids'].to(self.device, dtype=torch.long)
            mask = batch['mask'].to(self.device, dtype=torch.long)
            targets = batch['targets'].to(self.device, dtype=torch.float)
            seq_ids = batch['seq_ids'].to('cpu').numpy()
            positions = batch['positions'].to('cpu').numpy()

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                y_onehot = torch.nn.functional.one_hot(targets.long(), num_classes=len(
                    general_config.CLASS_LABELS.keys()))
                y_onehot = y_onehot.float()
                outputs = self.model(ids,
                                     attention_mask=mask,
                                     labels=y_onehot,
                                     return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
            loss = loss_fn(outputs, targets)
            logits = outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += float(loss.item())

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            final_data['targets'].extend(y_onehot.cpu().detach().numpy().tolist())
            if len(general_config.CLASS_LABELS.keys()) > 2:
                final_data['outputs'].extend(torch.softmax(outputs.logits, dim=1).cpu().detach().numpy().tolist())
            else:
                final_data['outputs'].extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
            final_data['seq_ids'].extend(seq_ids)
            final_data['positions'].extend(positions)

            # Save hidden states of last layer for clustering
            # hidden_states = Tuple of torch.FloatTensor (one for the output of the embeddings
            # + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            last_hidden_states = outputs.hidden_states[12].cpu().detach().numpy()
            for i, seq_id in enumerate(seq_ids):
                final_data['last_hidden_states_dict']['seq_id'] = last_hidden_states[i]

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
        log_fp.write("  Validation Accuracy: {0:.2f}\n".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        log_fp.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
        log_fp.write("  Validation took: {:}\n".format(validation_time))

        return final_data, avg_val_loss, avg_val_accuracy, validation_time

    def __train_epochs(self, train_dataloader, val_dataloader, train_data_size, log_fp):
        gc.collect()
        torch.cuda.empty_cache()
        optimizer, scheduler = self.__setup_training(train_data_size, log_fp)

        training_stats = []
        train_steps_loss = []
        total_t0 = time.time()
        log_fp.write("\nTraining and validation:\n")
        log_fp.write("==========================\n")

        for epoch in range(general_config.EPOCHS):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, general_config.EPOCHS))
            print('Training...')
            log_fp.write('\n======== Epoch {:} / {:} ========\n'.format(epoch + 1, general_config.EPOCHS))
            log_fp.write("\nTraining:\n")
            avg_train_loss, training_time, train_steps_loss_epoch = self.__train(train_dataloader, optimizer, scheduler,
                                                                                 log_fp)
            train_steps_loss.extend(train_steps_loss_epoch)

            print("")
            print("Validation...")
            log_fp.write("\nValidation:\n")
            final_data, avg_val_loss, avg_val_accuracy, validation_time = self.__validate(val_dataloader, log_fp)

            training_stats_epoch = {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'final_data': final_data
            }
            training_stats.append(training_stats_epoch)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
        log_fp.write("Total training took {:} (h:mm:ss)\n".format(format_time(time.time() - total_t0)))

        # return train hidden_states from last epoch
        return training_stats, train_steps_loss

    def finetune(self):
        with open(paths_config.train_file) as train_fp, open(paths_config.val_file) as val_fp, open(paths_config.log_file, 'a') as log_fp:
            train_reader = csv.reader(train_fp, delimiter=',')
            train_metadata = {'len': self.sizes_info['train_data_size_seqs']}
            X_train_generator = DatasetGenerator(train_reader, train_fp, train_metadata, self.tokenizer.tokenizer)
            train_dataloader = DataLoader(
                X_train_generator,  # The training samples.
                sampler=RandomSampler(X_train_generator),  # Select batches randomly
                batch_size=general_config.TRAIN_BATCH_SIZE,  # Trains with this batch size.
                num_workers=0
            )

            val_reader = csv.reader(val_fp, delimiter=',')
            val_metadata = {'len': self.sizes_info['val_data_size_seqs']}
            X_val_generator = DatasetGenerator(val_reader, val_fp, val_metadata, self.tokenizer)  # .batch(EVAL_BATCH_SIZE)

            validation_dataloader = DataLoader(
                X_val_generator,  # The validation samples.
                sampler=SequentialSampler(X_val_generator),  # Pull out batches sequentially.
                batch_size=general_config.EVAL_BATCH_SIZE,  # Evaluate with this batch size.
                num_workers=0
            )

            training_stats, train_steps_loss = self.__train_epochs(train_dataloader, validation_dataloader,
                                                                   self.sizes_info['train_data_size_seqs'],
                                                                   log_fp)

            self.save_model(paths_config.model_file_finetuned)

            # save last epoch validation data and statistics:
            final_data = training_stats[-1]['final_data']
            pickle.dump(final_data, open(paths_config.final_val_outputs_file, 'wb'))
            pickle.dump(train_steps_loss, open(paths_config.train_steps_loss_file, 'wb'))
            pickle.dump(training_stats, open(paths_config.training_stats_file, 'wb'))

    def report(self):
        if not os.path.exists(paths_config.final_val_outputs_file):
            raise FileNotFoundError(f'Error: file not found: {paths_config.final_val_outputs_file}')
        if not os.path.exists(paths_config.train_steps_loss_file):
            raise FileNotFoundError(f'Error: file not found: {paths_config.train_steps_loss_file}')
        if not os.path.exists(paths_config.training_stats_file):
            raise FileNotFoundError(f'Error: file not found: {paths_config.training_stats_file}')
        final_data = pickle.load(open(paths_config.final_val_outputs_file, 'rb'))
        train_steps_loss = pickle.load(open(paths_config.train_steps_loss_file, 'rb'))
        training_stats = pickle.load(open(paths_config.training_stats_file, 'rb'))
        report.show_train_stats_and_plots(training_stats, train_steps_loss)
        output_labels = np.argmax(final_data['outputs'], axis=1)
        target_labels = np.argmax(final_data['targets'], axis=1)
        report.final_statistics(target_labels, output_labels, final_data['outputs'], paths_config.log_file,
                                'Global Validation')


class Tester(Classifier):
    def __init__(self):
        super().__init__()

    def load_model(self, filepath):
        if self.device.type == 'cuda':
            self.model.load_state_dict(torch.load(filepath))
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))

    def test(self):
        if os.path.exists(paths_config.model_file_finetuned):
            self.load_model(paths_config.model_file_finetuned)
        else:
            raise FileNotFoundError(f'Error: model file not found: {paths_config.model_file_finetuned}')

        sizes_info = get_train_val_test_sizes()
        try:
            test_data_size = sizes_info['test_data_size_seqs'] # -1
        except NameError:
            test_data_size = sum(1 for line in open(paths_config.test_file))

        print(f'\nTEST DATA SIZE: {test_data_size}\n\n\n')

        with open(paths_config.test_file) as test_fp, open(paths_config.log_file, 'a') as log_fp:
            test_reader = csv.reader(test_fp, delimiter=',')
            test_metadata = {'len': test_data_size}
            X_test_generator = DatasetGenerator(test_reader, test_fp, test_metadata, self.tokenizer.tokenizer)
            test_dataloader = DataLoader(
                X_test_generator,  # The validation samples.
                sampler=SequentialSampler(X_test_generator),  # Pull out batches sequentially.
                batch_size=general_config.EVAL_BATCH_SIZE,  # Evaluate with this batch size.
                num_workers=0
            )

            final_data_test, test_accuracies, attentions, timings = self.__test(
                test_dataloader,
                log_fp,
                general_config.THETA
            )

            mean_syn = np.sum(timings) / len(timings)
            std_syn = np.std(timings)
            print(f"Inference time: {mean_syn} ({std_syn})")
            log_fp.write(f"Inference time: {mean_syn} ({std_syn})\n")

            # save output data on file for furure computation:
            pickle.dump(test_accuracies, open(paths_config.test_accuracies_file, 'wb'))
            pickle.dump(final_data_test, open(paths_config.final_data_test_file, 'wb'))

    def report(self):
        if not os.path.exists(paths_config.test_accuracies_file):
            raise FileNotFoundError(f'Error: file not found: {paths_config.test_accuracies_file}')
        if not os.path.exists(paths_config.final_data_test_file):
            raise FileNotFoundError(f'Error: file not found: {paths_config.final_data_test_file}')
        test_accuracies = pickle.load(open(paths_config.test_accuracies_file, 'rb'))
        final_data_test = pickle.load(open(paths_config.final_data_test_file, 'rb'))
        show_test_plots(test_accuracies)
        output_labels_test = np.argmax(final_data_test['outputs'], axis=1)
        target_labels_test = np.argmax(final_data_test['targets'], axis=1)
        output_logits_test = final_data_test['outputs']
        final_statistics(target_labels_test, output_labels_test, output_logits_test, paths_config.log_file, 'Test')  # , logits=final_data_test['logits'])

    def __test(self, test_dataloader, log_fp_test, theta=0):
        gc.collect()
        torch.cuda.empty_cache()

        print('Calculating predictions...')
        log_fp_test.write("\nTest:\n")
        log_fp_test.write("=====\n")
        # Put model in evaluation mode
        self.model.eval()

        # timing
        starter, ender, timings = None, None, None
        if self.device.type == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = []

            # GPU-warmup
            rand_idx = [np.random.randint(1, len(self.tokenizer.tokenizer.get_added_vocab())) for _ in range(512)]
            dummy_input = {
                'ids': torch.tensor([
                    np.take(np.asarray(list(self.tokenizer.tokenizer.get_added_vocab().values())), rand_idx, axis=0),
                    np.take(np.asarray(list(self.tokenizer.tokenizer.get_added_vocab().values())), rand_idx, axis=0)
                ],
                    dtype=torch.long),
                'mask': torch.ones(2, general_config.MAX_LENGTH, dtype=torch.long),
                'targets': torch.randn(2, general_config.MAX_LENGTH, dtype=torch.float),
                'seq_ids': torch.randint(2, general_config.MAX_LENGTH, (1, general_config.MAX_LENGTH)),
                'positions': torch.randint(2, general_config.MAX_LENGTH, (1, general_config.MAX_LENGTH))
            }
            for _ in range(10):
                _ = self.model(dummy_input['ids'].to(self.device, dtype=torch.long),
                          attention_mask=dummy_input['mask'].to(self.device, dtype=torch.long),
                          return_dict=False)

        test_accuracies = []
        total_test_accuracy = 0

        final_data_test = {'seq_ids': [],
                           'positions': [],
                           'targets': [],
                           'outputs': [],
                           'logits': [],
                           'output_embeddings': [],
                           'input_embeddings': [],
                           }
        attentions = {}
        attentions_last_layer = {}  # 1 heatmap per class, with attention scores of last layer of all samples
        attentions_all_layers = {}  # 1 list of attention matrices per class, with attention scores of all layers of 1 sample
        attentions_all_layers_thresh = {}
        count_class_samples = {}
        repr_token_base_positions_axis = {}

        distance_cones = {}
        distance_cones_1_sample = {}
        count_layer = {k: 0 for k in range(-1, 12)}

        model_params = dict(self.model.named_parameters())

        for _, label in general_config.CLASS_LABELS.items():
            count_class_samples[int(label)] = 0
        # attention_matrices_count = {}
        # for _, label in general_config.CLASS_LABELS.items():
        #     attention_matrices_count[int(label)] = 0
        # max_n_attention_matrices = 6

        # Predict
        for step, batch in enumerate(test_dataloader):

            # Progress update every 40 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5}  of  {:>5}.'.format(step, len(test_dataloader)))

            # Unpack the inputs from our dataloader
            ids = batch['ids'].to(self.device, dtype=torch.long)
            mask = batch['mask'].to(self.device, dtype=torch.long)
            targets = batch['targets'].to(self.device, dtype=torch.float)
            seq_ids = batch['seq_ids'].to('cpu').numpy()
            positions = batch['positions'].to('cpu').numpy()

            y_onehot = torch.nn.functional.one_hot(targets.long(), num_classes=len(
                general_config.CLASS_LABELS.keys())).float()

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                if self.device.type == 'cuda':
                    starter.record()
                outputs = self.model(ids,
                                     attention_mask=mask,
                                     return_dict=True,
                                     output_attentions=(general_config.TASK_TYPE == 'attention_analysis'),
                                     output_hidden_states=(general_config.TASK_TYPE == 'eigenvalues_analysis'
                                                           or general_config.TASK_TYPE == 'check_normality'
                                                           or general_config.TASK_TYPE == 'distance_cones_analysis'
                                                           or general_config.TASK_TYPE == 'layer_output_analysis'
                                                           or general_config.TASK_TYPE == 'print_layer_output'
                                                           or general_config.TASK_TYPE == 'clustering'))

                if self.device.type == 'cuda':
                    ender.record()
                    torch.cuda.synchronize()  # WAIT FOR GPU SYNC
                    curr_time = starter.elapsed_time(ender)
                    timings.append(curr_time)

            logits = outputs.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            # # Store predictions and true labels
            # predictions.append(logits)
            # true_labels.append(label_ids)

            if general_config.TASK_TYPE == 'print_layer_output':
                print_layer_output(seq_ids, label_ids, logits, outputs)
                return final_data_test, test_accuracies, attentions, timings

            elif general_config.TASK_TYPE == 'clustering':
                process_output_for_clustering(label_ids, logits, total_test_accuracy, seq_ids, positions, y_onehot, outputs, test_accuracies, final_data_test, count_class_samples)

            elif general_config.TASK_TYPE == 'attention_analysis':

                process_output_for_attention_analysis(label_ids, logits, seq_ids, outputs, count_class_samples, ids, theta, attentions_all_layers, attentions_all_layers_thresh, repr_token_base_positions_axis, self.tokenizer.tokenizer)

            elif general_config.TASK_TYPE == 'check_normality':
                check_normality(seq_ids, label_ids, logits, outputs)

            elif general_config.TASK_TYPE == 'distance_cones_analysis':
                process_output_for_distance_cones_analysis(label_ids, logits, seq_ids, outputs, model_params, distance_cones, self.model.config, count_layer, distance_cones_1_sample)
                break

            elif general_config.TASK_TYPE == 'layer_output_analysis':
                layer_output_analysis(seq_ids, logits, label_ids, outputs)
                return final_data_test, test_accuracies, attentions, timings

            elif general_config.TASK_TYPE == 'eigenvalues_analysis':
                eigenvalues_analysis(seq_ids, label_ids, logits, outputs, model_params)
                return final_data_test, test_accuracies, attentions, timings

        if general_config.TASK_TYPE == 'attention_analysis':
            # calculate mean of attention matrices:
            for target_label in general_config.CLASS_LABELS.values():
                for layer in range(len(attentions_all_layers[target_label])):
                    for head in range(len(attentions_all_layers[target_label][layer])):
                        attentions_all_layers[target_label][layer][head] = attentions_all_layers[target_label][layer][
                                                                               head] / count_class_samples[target_label]

        if general_config.TASK_TYPE == 'distance_cones_analysis':
            distance_cones_analysis(distance_cones, distance_cones_1_sample)

        # timing
        mean_syn = np.sum(timings) / len(timings)
        std_syn = np.std(timings)

        print('DONE.')

        # token_base_positions_axis = [f"{i*general_config.STRIDE-general_config.STRIDE+spike_gene_start}" for i in range(0,general_config.MAX_LENGTH)]

        attentions = {'attentions_all_layers': attentions_all_layers,
                      'attentions_all_layers_thresh': attentions_all_layers_thresh,
                      'repr_token_base_positions_axis': repr_token_base_positions_axis
                      }

        # print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
        # Report the final accuracy for this validation run.
        # avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        # print("  Global Test Accuracy: {0:.2f}".format(avg_test_accuracy))
        # log_fp_test.write("  Global Test Accuracy: {0:.2f}\n".format(avg_test_accuracy))

        return final_data_test, test_accuracies, attentions, timings

