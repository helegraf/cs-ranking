import logging

import numpy as np
from keras import Model, Input, optimizers
from keras.layers import TimeDistributed, Dense, Reshape
from keras.optimizers import SGD

from csrank.attention.set_transformer.modules import instantiate_attention_layer
from csrank.callbacks import configure_callbacks
from csrank.learner import Learner


class SetTransformer(Learner):
    def __init__(self, loss_function, loss_function_requires_x_values=False,
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=[], stacking_height=3,
                 attention_layer_config=None, batch_size=256, num_layers_dense=0, num_units_dense=8, seed=42,
                 n_objects=None, n_object_features=None, random_state=None, metrics_requiring_x=[]):
        if attention_layer_config is None:
            attention_layer_config = {"SAB": {"mab": {"MAB": {"multi_head": {
                "MultiHeadAttention": {"num_heads": 1, "attention": {"ScaledDotProductAttention": {}}}}}}}}
        elif not isinstance(attention_layer_config, dict):
            raise ValueError("Set Transformer attention layer needs to be given in the form of a dict as it is layered"
                             "and otherwise would lead to unintentional weight-sharing")

        self.loss_function = loss_function
        self.loss_function_requires_x_values = loss_function_requires_x_values
        self.optimizer = optimizers.get(optimizer)
        self.random_state = np.random.RandomState(seed=seed)

        if stacking_height < 1:
            raise ValueError("Stacking height needs to be at least 1")

        self.stacking_height = stacking_height
        self.num_layers_dense = num_layers_dense
        self.num_units_dense = num_units_dense

        self.attention_layer_config = attention_layer_config
        self.batch_size = batch_size
        self.model = None
        self.metrics = metrics
        self.metrics_requiring_x = metrics_requiring_x
        self.logger = logging.getLogger(SetTransformer.__name__)

        self.attention_layers = []

        super(SetTransformer, self).__init__()

    def predict_for_scores(self, scores, **kwargs):
        return super().predict_for_scores(scores, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        super(SetTransformer, self).set_tunable_parameters(**point)

    def fit(self, X, Y, generator=None, callbacks=None, epochs=36, validation_split=0.1, verbose=0,  **kwargs):
        _, n_objects, n_features = X.shape
        self.model = self.construct_model(n_objects, n_features)

        attention_outputs = []
        for attention_layer_num in range(len(self.attention_layers)):
            attention_layer = self.attention_layers[attention_layer_num]
            outputs = attention_layer.get_attention_layer_inputs_outputs()

            for output in outputs:
                output["name"] = "set_transformer_layer_{}_{}".format(str(attention_layer_num), output["name"])

            attention_outputs.extend(outputs)

        configure_callbacks(callbacks, attention_outputs=attention_outputs)

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

        # attention layers ("encoder")
        output_layer = input_layer
        for i in range(self.stacking_height):
            att_layer = instantiate_attention_layer(self.attention_layer_config)
            self.attention_layers.append(att_layer)
            output_layer = att_layer(output_layer)

        # dense layers rff
        for i in range(self.num_layers_dense):
            output_layer = TimeDistributed(Dense(units=self.num_units_dense, use_bias=True))(output_layer)

        # predict utility based on encoder ("decoder")
        output_layer_dec = TimeDistributed(Dense(units=1, use_bias=True))(output_layer)
        output_layer_dec_reshaped = Reshape(target_shape=(n_objects,))(output_layer_dec)

        model = Model(inputs=input_layer, outputs=output_layer_dec_reshaped)

        if self.loss_function_requires_x_values:
            self.loss_function = self.loss_function(input_layer)

        additional_metric = [metric(input_layer) for metric in self.metrics_requiring_x]

        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics.extend(additional_metric))

        return model



