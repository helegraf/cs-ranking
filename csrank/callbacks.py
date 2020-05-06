import logging
import math
import os
import warnings

import h5py
import numpy as np
from keras.callbacks import *
import tensorflow as tf

from csrank.metrics import tsp_loss_absolute_wrapper, tsp_distance_wrapper, tsp_loss_relative_wrapper
from csrank.metrics_np import kendalls_tau_for_scores_np, spearman_correlation_for_scores_scipy, \
    zero_one_rank_loss_for_scores_np, zero_one_rank_loss_for_scores_ties_np, zero_one_accuracy_for_scores_np, \
    categorical_accuracy_np, topk_categorical_accuracy_np, recall, precision, f1_measure, instance_informedness, \
    average_precision, hamming, subset_01_loss, auc_score
from csrank.numpy_util import scores_to_rankings
from csrank.tunable import Tunable
from csrank.util import print_dictionary, create_dir_recursively
from csrank.visualization.predictions import tsp_figure, figure_to_bytes, create_image_plotting_graph, \
    create_scalar_plotting_graph, create_attention_plotting_graph
from csrank.visualization.weights import visualize_attention_scores


class EarlyStoppingWithWeights(EarlyStopping, Tunable):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, **kwargs):
        super(EarlyStoppingWithWeights, self).__init__(**kwargs)
        self.logger = logging.getLogger(EarlyStoppingWithWeights.__name__)

    def on_train_begin(self, logs=None):
        super(EarlyStoppingWithWeights, self).on_train_begin(logs=logs)
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        current = logs.get(self.monitor)
        self.best_weights = self.model.get_weights()
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.logger.info("Setting best weights for final epoch {}".format(self.epoch))
            self.model.set_weights(self.best_weights)

    def set_tunable_parameters(self, patience=300, min_delta=2, **point):
        self.patience = patience
        self.min_delta = min_delta
        if len(point) > 0:
            self.logger.warning('This callback does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


class weightHistory(Callback):
    def on_train_begin(self, logs={}):
        self.zero_weights = []
        self.norm = []
        self.hidden_units_used = []

    def on_batch_end(self, batch, logs={}):
        hidden = [layer for layer in self.model.layers
                  if layer.name == 'hidden_1']

        y = np.array(hidden[0].get_weights()[0])
        close = np.isclose(y, 0, atol=1e-3)
        self.hidden_units_used.append(len(np.unique(np.where(np.logical_not(close))[1])))
        self.norm.append(np.abs(y).sum())
        self.zero_weights.append(close.sum())


class LRScheduler(LearningRateScheduler, Tunable):
    """Learning rate scheduler.

        # Arguments
            epochs_drop: unsigned int
            drop:
            verbose: int. 0: quiet, 1: update messages.
        """

    def __init__(self, epochs_drop=300, drop=0.1, **kwargs):
        super(LRScheduler, self).__init__(self.step_decay, **kwargs)

        self.epochs_drop = epochs_drop
        self.drop = drop

    def step_decay(self, epoch, lr):
        step = math.floor((1 + epoch) / self.epochs_drop)
        lrate = lr * math.pow(self.drop, step)
        return lrate

    def set_tunable_parameters(self, epochs_drop=300, drop=0.1, **point):
        self.epochs_drop = epochs_drop
        self.drop = drop
        if len(point) > 0:
            self.logger.warning('This callback does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    [this implementation is from github: https://github.com/bckenstler/CLR/blob/master/clr_callback.py]

    # License
    MIT License

    Copyright (c) 2017 Bradley Kenstler

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', **kwargs):
        super(CyclicLR, self).__init__(**kwargs)

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            # start with base learning rate
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            # if not at the start, use cyclical learning rate
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class WeightPrinterCallback(Callback):
    def __init__(self, learner):
        self.learner = learner

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.learner.model.layers:
            print(layer.get_weights())


class AdvancedTensorBoard(TensorBoard):
    def __init__(self, data_visualization_func=None, metric_for_visualization=None,
                 metric_for_visualization_requires_x=False,
                 num_visualizations_per_epoch=3, log_lr=True, log_gradient_norms=True, save_space=True,
                 visualization_frequency=None, save_visualization_data=False, metrics=[], metric_requires_x=[],
                 log_attention=False, **kwargs):
        super(AdvancedTensorBoard, self).__init__(**kwargs)

        # set up logging
        self.logger = logging.getLogger('Advanced_Tensorboard')

        # create plotting function if necessary
        func_dict = {
            'tsp_2d': tsp_figure
        }
        if data_visualization_func is not None and isinstance(data_visualization_func, str):
            self.data_visualizing_func = func_dict[data_visualization_func]
        else:
            self.data_visualizing_func = data_visualization_func

        # to avoid circular imports this dictionary converts give metrics
        all_metrics = {'KendallsTau': kendalls_tau_for_scores_np,
                       'SpearmanCorrelation': spearman_correlation_for_scores_scipy,
                       'ZeroOneRankLoss': zero_one_rank_loss_for_scores_np,
                       'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties_np,
                       'ZeroOneAccuracy': zero_one_accuracy_for_scores_np,
                       'TSPAbsoluteDifference_requiresX': tsp_loss_absolute_wrapper,
                       'TSPRelativeDifference_requiresX': tsp_loss_relative_wrapper,
                       'TSPDistance_requiresX': tsp_distance_wrapper,
                       'CategoricalAccuracy': categorical_accuracy_np,
                       'CategoricalTopK2': topk_categorical_accuracy_np(k=2),
                       'CategoricalTopK3': topk_categorical_accuracy_np(k=3),
                       'CategoricalTopK4': topk_categorical_accuracy_np(k=4),
                       'CategoricalTopK5': topk_categorical_accuracy_np(k=5),
                       'CategoricalTopK6': topk_categorical_accuracy_np(k=6),
                       'F1Score': f1_measure, 'Precision': precision, 'Recall': recall,
                       'Subset01loss': subset_01_loss, 'HammingLoss': hamming, 'Informedness': instance_informedness,
                       "AucScore": auc_score, "AveragePrecisionScore": average_precision
                       }

        # create metric for the plot
        if metric_for_visualization is not None:
            if isinstance(metric_for_visualization, str):
                self.metric_for_logging_func = all_metrics[metric_for_visualization]
            else:
                self.metric_for_logging_func = metric_for_visualization
        self.metric_for_logging_func_requires_x = metric_for_visualization_requires_x

        # create metrics that should be tracked
        self.metrics = [all_metrics[metric] for metric in metrics]
        self.metric_requires_x = metric_requires_x

        # check if data needs to be saved
        if data_visualization_func is not None or len(metrics) > 0 or log_attention or save_visualization_data:
            self.save_data = True

            self.inputs = []
            self.targets = []
            self.outputs = []

            self.tmp_inputs = tf.Variable(0., validate_shape=False)
            self.tmp_targets = tf.Variable(0., validate_shape=False)
            self.tmp_outputs = tf.Variable(0., validate_shape=False)

            self.batches_per_epoch = [0]
            self.current_epoch = 0
        else:
            self.save_data = False

        # check if attention data shall be saved
        if log_attention:
            self.tmp_attention_keys = {}
            self.tmp_attention_queries = {}
            self.tmp_attention_scores = {}
            # maps attention vis names to their variables
            self.attention_keys = {}
            self.attention_queries = {}
            self.attention_scores = {}
        self.log_attention = log_attention

        # set visualization frequency
        if visualization_frequency is not None:
            self.visualization_frequency = visualization_frequency
        else:
            self.visualization_frequency = self.histogram_freq

        # save all other parameters
        self.num_visualizations_per_epoch = num_visualizations_per_epoch
        self.log_lr = log_lr
        self.log_gradient_norms = log_gradient_norms
        self.save_space = save_space
        self.save_visualization_data = save_visualization_data

    def set_model(self, model):
        # create scalar and histograms for gradient norms
        if self.log_gradient_norms and self.histogram_freq and self.merged is None:
            all_gradients = []
            for layer in model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    if self.log_gradient_norms and weight in layer.trainable_weights:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'

                        grads = [tf.norm(grad.values) if is_indexed_slices(grad) else tf.norm(grad) for grad in grads]
                        for grad in grads:
                            all_gradients.append(grad)

                        tf.summary.histogram(name='{}_grad_norm'.format(mapped_weight_name), values=grads)

            global_norm = tf.linalg.global_norm(all_gradients)
            tf.summary.scalar(name='global_norm', tensor=global_norm)

        super(AdvancedTensorBoard, self).set_model(model)

    def on_batch_end(self, batch, logs=None):
        # if data needs to be saved, convert data from batch to numpy arrays and save
        if self.save_data and self.current_epoch % self.visualization_frequency == 0:
            # get data
            self.inputs.append(K.eval(self.tmp_inputs))
            self.targets.append(K.eval(self.tmp_targets))
            self.outputs.append(K.eval(self.tmp_outputs))

            if self.log_attention:
                for attention_layer, query in self.tmp_attention_queries.items():
                    self.attention[attention_layer].append(K.eval(query))
                for attention_layer, key in self.tmp_attention_keys.items():
                    self.attention[attention_layer].append(K.eval(query))
                for attention_layer, score in self.tmp_attention_scores.items():
                    self.attention[attention_layer].append(K.eval(score))

            # increase batch counter by 1
            self.batches_per_epoch[-1] += 1

        super(AdvancedTensorBoard, self).on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # if the learning rate is logged, note the value
        if self.log_lr:
            logs = logs or {}
            logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})

        # if in the next epoch input data is to be logged, reset batches per epoch
        if self.save_data:
            self.current_epoch += 1
            if self.current_epoch % self.visualization_frequency == 0:
                self.batches_per_epoch.append(0)

        super(AdvancedTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs):
        self.logger.info("Training has ended; starting to do visualizations and additional scalars")

        if self.save_data:
            # if any prediction visualizations are to be made, we choose the first k instances (k is however many
            # instances should be visualized) from the first batch in the first epoch, and identify them in all
            # following epochs, which needs to be done as instances are shuffled between epochs during training.

            # find the correct instances
            inputs, outputs, targets = self.extract_needed_instances()

            # prepare data structures for saving visualization data for later
            if self.save_visualization_data:
                save_data = self.prepare_save_data()

            with tf.Session() as sess:

                # plot images (or just save the data for it)
                if self.data_visualizing_func is not None or self.save_visualization_data or self.log_attention:
                    for j in range(self.num_visualizations_per_epoch):
                        # create prediction visualization graph
                        if self.data_visualizing_func is not None:
                            matplotlib_img_bytes, prediction_merged = create_image_plotting_graph(str(j))

                        # create attention activation visualization graph
                        if self.log_attention:
                            attention_img_bytes, attention_merged = [create_attention_plotting_graph(str(i), layer_name)
                                                   for layer_name in self.attention_keys.keys()]

                        previous_prediction = np.empty(shape=1)

                        for i in range(len(inputs)):
                            current_prediction = scores_to_rankings(np.asarray([outputs[i][j]]))

                            # if space is saved, images are only plotted if the actual ranking has changed from one
                            # epoch to another in order not to plot an unnecessary amount of images
                            if (self.save_space and not np.array_equal(previous_prediction, current_prediction)) \
                                    or not self.save_space:

                                if self.save_visualization_data:
                                    save_data['epoch'].append(i * self.visualization_frequency)
                                    save_data['instance_index'].append(j)

                                if self.metric_for_logging_func is not None and self.save_visualization_data:
                                    metric = self.create_metric(i, inputs, j, outputs, targets)
                                    save_data['metric'].append(metric)
                                else:
                                    metric = None

                                if self.data_visualizing_func is not None:
                                    self.plot_data_prediction(i, inputs, j, matplotlib_img_bytes, prediction_merged,
                                                              metric, outputs, sess, targets)
                                    if self.save_visualization_data:
                                        self.save_image_plotting_data(i, inputs, j, outputs, save_data, targets)

                                if self.log_attention:
                                    for layer_name in self.attention_keys.keys():
                                        self.plot_data_attention(i=i, queries=self.attention_queries[layer_name], j=j,
                                                                 attention_img_bytes=attention_img_bytes,
                                                                 merged=attention_merged,
                                                                 keys=self.attention_keys[layer_name],
                                                                 scores=self.attention_scores[layer_name],
                                                                 sess=sess)

                                previous_prediction = current_prediction

                self.logger.info("Done with visualizations, starting scalars")

                # plot the metric if any is given
                if len(self.metrics) > 0:
                    self.add_metric_plots(sess)

                self.logger.info("Done with scalars")

            self.writer.flush()

        # write data used to create the visualizations to a file
        if self.save_visualization_data:
            self.logger.info("Saving visualization raw origin data")
            self.write_visualizations_file(save_data)
            self.logger.info("Done saving data")

        self.logger.info("Custom train end done")

        super(AdvancedTensorBoard, self).on_train_end(logs)

    def plot_data_prediction(self, i, inputs, j, matplotlib_img_bytes, merged, metric, outputs, sess, targets):
        figure = figure_to_bytes(self.data_visualizing_func(inputs[i][j], targets[i][j],
                                                            outputs[i][j], metric,
                                                            i * self.visualization_frequency))
        run_summary = sess.run(fetches=merged, feed_dict={matplotlib_img_bytes: figure})
        self.writer.add_summary(run_summary, i * self.visualization_frequency)

    def plot_data_attention(self, i, queries, j, attention_img_bytes, merged, keys, sess, scores):
        figure = figure_to_bytes(visualize_attention_scores(queries[i][j], keys[i][j], scores[i][j]))

        run_summary = sess.run(fetches=merged, feed_dict={attention_img_bytes: figure})
        self.writer.add_summary(run_summary, i * self.visualization_frequency)

    def add_metric_plots(self, sess):
        metric_values_in_epoch = {}
        for metric in self.metrics:
            metric_values_in_epoch[metric] = []
        # go through the data epoch by epoch
        batches_so_far = 0
        for epoch in self.batches_per_epoch:
            metric_values = {}
            for metric in self.metrics:
                metric_values[metric] = []

            for batch in range(batches_so_far, batches_so_far + self.batches_per_epoch[epoch]):
                for metric in range(len(self.metrics)):
                    if self.metric_requires_x[metric]:
                        metric_values_in_batch = self.metrics[metric] \
                            (self.inputs[batch]) \
                            (self.targets[batch], self.outputs[batch])
                    else:
                        metric_values_in_batch = self.metrics[metric] \
                            (self.targets[batch], self.outputs[batch])

                    metric_value_in_batch_list = metric_values_in_batch.tolist()
                    if isinstance(metric_value_in_batch_list, list):
                        metric_values[self.metrics[metric]].extend(metric_values_in_batch)
                    else:
                        metric_values[self.metrics[metric]].append(metric_values_in_batch)

                for metric in self.metrics:
                    avg = np.average(metric_values[metric])
                    metric_values_in_epoch[metric].append(avg)
                    metric_values[metric] = []

            batches_so_far += self.batches_per_epoch[epoch]
        # plot data
        value_placeholders = []
        summary_scalars = []
        for metric in self.metrics:
            value_placeholder, summary_scalar = create_scalar_plotting_graph(metric.__name__)
            value_placeholders.append(value_placeholder)
            summary_scalars.append(summary_scalar)
        merged = tf.summary.merge(summary_scalars)
        for epoch in range(len(self.batches_per_epoch)):
            data = [metric_values_in_epoch[metric][epoch] for metric in self.metrics]
            run_summary = sess.run(fetches=merged, feed_dict={input_: data_entry for input_, data_entry in
                                                              zip(value_placeholders, data)})
            self.writer.add_summary(run_summary, epoch * self.visualization_frequency)

    def write_visualizations_file(self, save_data):
        pred_file = os.path.join(self.log_dir, "visualizations.h5")
        create_dir_recursively(pred_file, True)
        f = h5py.File(pred_file, 'w')
        f.create_dataset('epoch', data=np.asarray(save_data['epoch']))
        f.create_dataset('instance_index', data=np.asarray(save_data['instance_index']))
        f.create_dataset('inputs', data=np.asarray(save_data['inputs']))
        f.create_dataset('outputs', data=np.asarray(save_data['outputs']))
        f.create_dataset('targets', data=np.asarray(save_data['targets']))
        if self.metric_for_logging_func is not None:
            data_set = f.create_dataset('metric', data=np.asarray(save_data['metric']))
            data_set.attrs['metric_name'] = self.metric_for_logging_func.__name__
        f.close()

    def save_image_plotting_data(self, i, inputs, j, outputs, save_data, targets):
        save_data['inputs'].append(inputs[i][j])
        save_data['outputs'].append(outputs[i][j])
        save_data['targets'].append(targets[i][j])

    def prepare_save_data(self):
        save_data = {
            'epoch': [],
            'instance_index': [],
            'inputs': [],
            'outputs': [],
            'targets': []
        }
        if self.metric_for_logging_func is not None:
            save_data['metric'] = []
        return save_data

    def create_metric(self, i, inputs, j, outputs, targets):
        if self.metric_for_logging_func is not None:
            if self.metric_for_logging_func_requires_x:
                metric_func = self.metric_for_logging_func([inputs[i][j]])
                metric = metric_func([targets[i][j]], [outputs[i][j]])
            else:
                metric = self.metric_for_logging_func(targets[i][j], outputs[i][j])
            return metric
        else:
            return None

    def extract_needed_instances(self):
        inputs = [self.inputs[0][0:self.num_visualizations_per_epoch]]
        outputs = [self.outputs[0][0:self.num_visualizations_per_epoch]]
        targets = [self.targets[0][0:self.num_visualizations_per_epoch]]
        # for every epoch (except the first)
        last_element = self.batches_per_epoch[-1]
        if last_element == 0:
            del self.batches_per_epoch[-1]
        batches_so_far = self.batches_per_epoch[0]
        for epoch in range(1, len(self.batches_per_epoch)):
            inputs.append(np.empty(inputs[0].shape))
            outputs.append(np.empty(outputs[0].shape))
            targets.append(np.empty(targets[0].shape))

            # for every correctly ordered instance
            for correct_instance in range(len(inputs[0])):
                input_, output, target = self.find_correct_instance(
                    batches_so_far, self.batches_per_epoch[epoch], inputs[0][correct_instance])

                inputs[epoch][correct_instance] = input_
                outputs[epoch][correct_instance] = output
                targets[epoch][correct_instance] = target

            batches_so_far += self.batches_per_epoch[epoch]
        return inputs, outputs, targets

    def find_correct_instance(self, batches_so_far, num_batches_per_epoch, correct_instance):
        # for every batch in that epoch
        for batch in range(batches_so_far, batches_so_far + num_batches_per_epoch):
            # for every instance in that
            for instance in range(len(self.inputs[batch])):
                if np.array_equal(self.inputs[batch][instance], correct_instance):
                    return self.inputs[batch][instance], self.outputs[batch][instance], self.targets[batch][instance]

        raise ValueError("didnt find instance \n {} :(".format(correct_instance))


class DebugOutput(Callback):

    def __init__(self, delta=100, **kwargs):
        super(DebugOutput, self).__init__(**kwargs)
        self.delta = delta

    def on_train_end(self, logs=None):
        self.logger.debug('Total number of epochs: {}'.format(self.epoch))

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.logger = logging.getLogger('DebugOutput')

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if self.epoch % self.delta == 0:
            self.logger.debug('Epoch {} of the training finished.'.format(self.epoch))


def configure_callbacks(model, list_of_callbacks):
    # if tracking callback present, add to model fetches
    if list_of_callbacks is not None:
        for callback in list_of_callbacks:
            if isinstance(callback, AdvancedTensorBoard):
                if model._function_kwargs is not None and 'fetches' in model._function_kwargs.keys():
                    fetches = model._function_kwargs['fetches']
                else:
                    fetches = []

                if callback.save_data:
                    fetches.append(tf.assign(callback.tmp_inputs, model.inputs[0], validate_shape=False))
                    fetches.append(tf.assign(callback.tmp_targets, model.targets[0], validate_shape=False))
                    fetches.append(tf.assign(callback.tmp_outputs, model.outputs[0], validate_shape=False))

                if callback.log_attention:
                    fetches.append(tf.assign(callback.tmp_attention, model.layers[1].outputs[0], validate_shape=False))

                model._function_kwargs = {'fetches': fetches}