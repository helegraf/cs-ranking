import logging

import numpy as np
from keras import Model, Input, optimizers
from keras.layers import TimeDistributed, Dense, Reshape
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from csrank.attention.set_transformer.modules import set_transformer_blocks
from csrank.callbacks import configure_callbacks
from csrank.choicefunction.choice_functions import ChoiceFunctions
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.learner import Learner
import tensorflow as tf

from csrank.losses import hinged_rank_loss
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.objectranking.set_transformer_object_ranker import SetTransformerObjectRanker


class SimpleAttentionNetwork(Learner):
    def __init__(self, loss_function, loss_function_requires_x_values=False,
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=[], layer_fun=None, batch_size=32, seed=42):
        self.loss_function = loss_function
        self.loss_function_requires_x_values = loss_function_requires_x_values
        self.optimizer = optimizers.get(optimizer)
        self.random_state = np.random.RandomState(seed=seed)

        self.layer_fun = layer_fun

        self.batch_size = batch_size
        self.model = None
        self.metrics = metrics
        self.logger = logging.getLogger(SimpleAttentionNetwork.__name__)
        super(SimpleAttentionNetwork, self).__init__()

    def predict_for_scores(self, scores, **kwargs):
        return super().predict_for_scores(scores, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        super(SimpleAttentionNetwork, self).set_tunable_parameters(**point)

    def fit(self, X, Y, generator=None, callbacks=None, epochs=36, validation_split=0.1, verbose=0,  **kwargs):

        n_inst, n_objects, n_features = X.shape
        self.model = self.construct_model(n_objects, n_features)
        configure_callbacks(self.model, callbacks)

        if generator is None:
            self.model.fit(x=X, y=Y, callbacks=callbacks, epochs=epochs, validation_split=validation_split,
                           batch_size=self.batch_size, verbose=verbose, **kwargs)
        else:
            self.model.fit_generator(generator=generator, callbacks=callbacks, epochs=epochs, verbose=verbose,
                                     **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return self.model.predict(X)

    def construct_model(self, n_objects, n_features):
        # input
        input_layer = Input(shape=(n_objects, n_features), name="input_node")

        output_layer = self.layer_fun(input_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        if self.loss_function_requires_x_values:
            self.loss_function = self.loss_function(input_layer)

        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)

        return model


class SimpleAttentionObjectRanker(SimpleAttentionNetwork, ObjectRanker):
    def __init__(self, loss_function=hinged_rank_loss, loss_function_requires_x_values=False, **kwargs):
        super(SetTransformerObjectRanker, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerObjectRanker.__name__)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)


class SetTransformerChoice(SimpleAttentionNetwork, ChoiceFunctions):

    def __init__(self, threshold=0.5, loss_function=binary_crossentropy, loss_function_requires_x_values=False,
                 **kwargs):
        super(SetTransformerChoice, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerChoice.__name__)
        self.threshold = threshold

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, verbose=0, **kwargs):
        if tune_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                super().fit(x_train, y_train, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(x_val, y_val, thin_thresholds=thin_thresholds, verbose=verbose)
        else:
            super().fit(X, Y, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)


class SetTransformerDiscreteChoice(SimpleAttentionNetwork, DiscreteObjectChooser):

    def __init__(self, loss_function='categorical_hinge', loss_function_requires_x_values=False, **kwargs):
        super(SetTransformerDiscreteChoice, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerDiscreteChoice.__name__)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)