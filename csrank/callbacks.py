import logging
import math
import os
import warnings
from datetime import datetime

import h5py
import numpy as np
from keras import backend as K
from keras.callbacks import *
import tensorflow as tf
from keras.callbacks import TensorBoard

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


def configure_callbacks(list_of_callbacks, attention_outputs=None):
    # if tracking callback present, add to model fetches
    if list_of_callbacks is not None:
        for callback in list_of_callbacks:
            if isinstance(callback, AdvancedTensorBoard):
                if callback.log_attention:
                    if attention_outputs is None:
                        raise ValueError("No attention outputs for callback detected.")

                    for attention_output in attention_outputs:
                        name = attention_output["name"]
                        callback.attention_queries[name] = attention_output["query"]
                        callback.attention_keys[name] = attention_output["key"]
                        callback.attention_scores[name] = attention_output["scores"]


class AdvancedTensorBoard(TensorBoard):
    def __init__(self, inputs=None, targets=None, log_lr=False, log_gradient_norms=None,
                 prediction_visualization=None, metric_for_visualization=None,
                 metric_for_visualization_requires_x=False, log_attention=False, save_space=False,
                 **kwargs):
        super(AdvancedTensorBoard, self).__init__(**kwargs)

        self.logger = logging.getLogger(self.__class__.__name__)

        # check options
        if log_gradient_norms is not None and log_gradient_norms not in ["all", "global"]:
            raise ValueError("Allowed options for log_gradient_norms are: \"all\", \"global\", None\"")
        func_dict = {
            'tsp_2d': tsp_figure
        }
        if prediction_visualization is not None and isinstance(prediction_visualization, str):
            self.prediction_visualization = func_dict[prediction_visualization]
        else:
            self.prediction_visualization = prediction_visualization

        # create metric for the plot
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
        if metric_for_visualization is not None:
            if isinstance(metric_for_visualization, str):
                self.metric_for_visualization = all_metrics[metric_for_visualization]
            else:
                self.metric_for_visualization = metric_for_visualization
        else:
            self.metric_for_visualization = None
        self.metric_for_visualization_requires_x = metric_for_visualization_requires_x

        # inputs and targets
        self.x = inputs
        self.y = targets

        # options
        self.log_lr = log_lr
        self.log_gradient_norms = log_gradient_norms
        self.save_space = save_space
        self.log_attention = log_attention

        # other attributes
        self.symbolic_inputs = None
        self.prediction_plotting_graphs = None
        self.attention_plotting_graphs = {}
        self.previous_prediction = None
        self.attention_queries = {}
        self.attention_keys = {}
        self.attention_scores = {}

    def on_train_begin(self, logs=None):
        super(AdvancedTensorBoard, self).on_train_begin(logs)

        if self.prediction_visualization is not None:
            # create plotting graphs
            self.prediction_plotting_graphs = np.asarray([create_image_plotting_graph(num_visualization)
                                                          for num_visualization in range(len(self.x))])
            # initial previous
            self.previous_prediction = np.empty(shape=self.y.shape)

        # create attention activation visualization graph
        if self.log_attention:
            for layer_name in self.attention_keys.keys():
                self.attention_plotting_graphs[layer_name] = \
                    np.asarray([create_attention_plotting_graph(num_visualization, layer_name)
                                for num_visualization in range(len(self.x))])

    def on_epoch_end(self, epoch, logs=None):
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            # if the learning rate is logged, note the value
            if self.log_lr:
                logs = logs or {}
                logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})

            # prediction
            outputs = self.model.predict(self.x)
            predictions_as_rankings = scores_to_rankings(outputs)

            # metric set up
            if self.metric_for_visualization is not None:
                metrics = [self.metric_for_visualization(self.x)(self.y, predictions_as_rankings)
                           if self.metric_for_visualization_requires_x else
                           self.metric_for_visualization(self.y, predictions_as_rankings)]
            else:
                metrics = [None for _ in self.x]

            # attention set up
            if self.log_attention:
                queries = {}
                keys = {}
                scores = {}
                for layer_name in self.attention_keys.keys():
                    # check if equal to input; in that case dont fetch and just use input
                    print("feed", self.symbolic_inputs)
                    print("fetch", self.attention_queries[layer_name])
                    queries[layer_name] = self.eval_attention_tensor(self.attention_queries[layer_name])
                    keys[layer_name] = self.eval_attention_tensor(self.attention_keys[layer_name])
                    scores[layer_name] = self.eval_attention_tensor(self.attention_scores[layer_name])

            # visualize
            for num_visualization in range(len(self.x)):
                if not self.save_space or self.save_space and \
                        not np.array_equal(predictions_as_rankings[num_visualization],
                                           self.previous_prediction[num_visualization]):
                    # prediction
                    if self.prediction_visualization is not None:
                        prediction_img_bytes = self.prediction_plotting_graphs[num_visualization][0]
                        prediction_merged = self.prediction_plotting_graphs[num_visualization][1]
                        vis_data = self.prediction_visualization(self.x[num_visualization],
                                                                 self.y[num_visualization],
                                                                 predictions_as_rankings[num_visualization],
                                                                 metrics[num_visualization],
                                                                 epoch)
                        prediction_figure = figure_to_bytes(vis_data)
                        run_summary = self.sess.run(fetches=prediction_merged,
                                                    feed_dict={prediction_img_bytes: prediction_figure})
                        self.writer.add_summary(run_summary, epoch)

                    # attention
                    if self.log_attention:
                        for layer_name in self.attention_keys.keys():
                            attention_img_bytes = self.attention_plotting_graphs[layer_name][num_visualization][0]
                            attention_merged = self.attention_plotting_graphs[layer_name][num_visualization][1]
                            attention_figure = figure_to_bytes(
                                visualize_attention_scores(queries[layer_name][num_visualization],
                                                           keys[layer_name][num_visualization],
                                                           scores[layer_name][num_visualization]))
                            run_summary = self.sess.run(fetches=attention_merged,
                                                        feed_dict={attention_img_bytes: attention_figure})
                            self.writer.add_summary(run_summary, epoch)

                    self.previous_prediction[num_visualization] = predictions_as_rankings[num_visualization]

        super(AdvancedTensorBoard, self).on_epoch_end(epoch, logs)

    def eval_attention_tensor(self, symbolic_attention_tensor):
        if self.symbolic_inputs[0] == symbolic_attention_tensor:
            evaluated_attention_tensor = self.x
        else:
            evaluated_attention_tensor = K.function(self.symbolic_inputs, symbolic_attention_tensor)(self.x)
        return evaluated_attention_tensor

    def set_model(self, model):
        # create scalar and histograms for gradient norms
        if self.log_gradient_norms is not None and self.histogram_freq and self.merged is None:
            all_gradients = []
            for layer in model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    if self.log_gradient_norms and weight in layer.trainable_weights:
                        grads = model.optimizer.get_gradients(model.total_loss, weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'

                        grads = [tf.norm(grad.values) if is_indexed_slices(grad) else tf.norm(grad) for grad in grads]
                        for grad in grads:
                            all_gradients.append(grad)

                        if self.log_gradient_norms == "all":
                            tf.summary.histogram(name='{}_grad_norm'.format(mapped_weight_name), values=grads)

            global_norm = tf.linalg.global_norm(all_gradients)
            tf.summary.scalar(name='global_norm', tensor=global_norm)

        # save symbolic model inputs
        self.symbolic_inputs = model.inputs

        super(AdvancedTensorBoard, self).set_model(model)