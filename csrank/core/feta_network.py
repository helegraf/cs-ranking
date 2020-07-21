import logging
from itertools import permutations, combinations

import numpy as np
import tensorflow as tf
from keras import optimizers, Input, Model, backend as K
from keras.layers import Dense, concatenate, Lambda, add, Reshape, Permute, Concatenate
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.attention.set_transformer.modules import instantiate_attention_layer
from csrank.callbacks import configure_callbacks
from csrank.constants import allowed_dense_kwargs
from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.losses import hinged_rank_loss
from csrank.util import print_dictionary


class FETANetwork(Learner):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8, add_zeroth_order_model=False,
                 max_number_of_objects=5, num_subsample=5, loss_function=hinged_rank_loss,
                 loss_function_requires_x_values=False, batch_normalization=False,
                 kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal', activation='selu',
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=None, batch_size=256, random_state=None,
                 attention_pooling_config=None, attention_preselection_config=None,
                 num_attention_preselection_layers=0, attention_pooling_flavour=None, metrics_requiring_x=[], **kwargs):
        if attention_pooling_config is not None and not isinstance(attention_pooling_config, dict):
            raise ValueError("Attention pooling layer has to be given in dictionary-form because it needs to be "
                             "instantiated multiple times. Has to be able to be processed by "
                             "csrank.attention.set_transformer.modules.instantiate_attention_layer")
        if attention_preselection_config is not None and \
                not isinstance(attention_preselection_config, dict):
            raise ValueError("Attention preselection layer has to be given in dictionary-form because it needs to be "
                             "instantiated multiple times. Has to be able to be processed by "
                             "csrank.attention.set_transformer.modules.instantiate_attention_layer")
        allowed_attention_pooling_flavours = ["multi_pma", "single_pma", "mab"]
        if attention_pooling_flavour is not None \
                and attention_pooling_flavour not in allowed_attention_pooling_flavours:
            raise ValueError("{} is not an allowed attention flavour. The options are {}".format(
                attention_pooling_flavour, allowed_attention_pooling_flavours))
        if attention_pooling_flavour == "single_pma" and add_zeroth_order_model == False:
            raise ValueError("Using single_pma as attention pooling requires oth order model.")
        self.attention_pooling_flavour = attention_pooling_flavour
        self.loss_function_requires_x_values = loss_function_requires_x_values
        self.attention_preselection_config = attention_preselection_config
        self.attention_preselection_layers = None
        self.num_attention_preselection_layers = num_attention_preselection_layers
        self.attention_pooling_config = attention_pooling_config
        self.attention_pooling_layers = None
        self.logger = logging.getLogger(FETANetwork.__name__)
        self.random_state = check_random_state(random_state)
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.loss_function = loss_function
        self.metrics = metrics
        self.metrics_requiring_x = metrics_requiring_x
        self._n_objects = n_objects
        self.max_number_of_objects = max_number_of_objects
        self.num_subsample = num_subsample
        self.n_object_features = n_object_features
        self.batch_size = batch_size
        self.hash_file = None
        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()
        self._use_zeroth_model = add_zeroth_order_model
        self.n_hidden = n_hidden
        self.n_units = n_units
        keys = list(kwargs.keys())
        for key in keys:
            if key not in allowed_dense_kwargs:
                del kwargs[key]
        self.kwargs = kwargs
        self.set_up_input_layer()
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        self._pairwise_model = None
        self.model = None
        self._zero_order_model = None

    def set_up_input_layer(self):
        self.input_layer = Input(shape=(self.n_objects, self.n_object_features))
        if self.attention_preselection_config is not None:
            self.attention_preselection_layers = \
                [instantiate_attention_layer(self.attention_preselection_config)
                 for _ in range(self.num_attention_preselection_layers)]
            self.attention_preselected_input = self.input_layer
            for layer in self.attention_preselection_layers:
                self.attention_preselected_input = layer(self.attention_preselected_input)
        else:
            self.attention_preselected_input = None

    @property
    def n_objects(self):
        if self._n_objects > self.max_number_of_objects:
            return self.max_number_of_objects
        return self._n_objects

    def _construct_layers(self, **kwargs):
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        self.logger.info("n_hidden {}, n_units {}".format(self.n_hidden, self.n_units))
        if self.batch_normalization:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [NormalizedDense(self.n_units, name="hidden_zeroth_{}".format(x), **kwargs)
                                             for x in range(self.n_hidden)]
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        else:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [Dense(self.n_units, name="hidden_zeroth_{}".format(x), **kwargs) for x in
                                             range(self.n_hidden)]
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden
        self.output_node = Dense(1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer)
        if self._use_zeroth_model:
            self.output_node_zeroth = Dense(1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer)

    @property
    def zero_order_model(self):
        if self._zero_order_model is None and self._use_zeroth_model:
            self.logger.info('Creating zeroth model')
            inp = Input(shape=(self.n_object_features,))

            x = inp
            for hidden in self.hidden_layers_zeroth:
                x = hidden(x)
            zeroth_output = self.output_node_zeroth(x)

            self._zero_order_model = Model(inputs=[inp], outputs=zeroth_output)
            self.logger.info('Done creating zeroth model')
        return self._zero_order_model

    @property
    def pairwise_model(self):
        if self._pairwise_model is None:
            print("creating the stoopid pairwise model")
            self.logger.info('Creating pairwise model')
            x1 = Input(shape=(self.n_object_features,))
            x2 = Input(shape=(self.n_object_features,))

            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            merged_output = concatenate([n_g, n_l])
            self._pairwise_model = Model(inputs=[x1, x2], outputs=merged_output)
            self.logger.info('Done creating pairwise model')
        return self._pairwise_model

    def _predict_pair(self, a, b, only_pairwise=False, **kwargs):
        # TODO: Is this working correctly?
        pairwise = self.pairwise_model.predict([a, b], **kwargs)
        if not only_pairwise and self._use_zeroth_model:
            utility_a = self.zero_order_model.predict([a])
            utility_b = self.zero_order_model.predict([b])
            return pairwise + (utility_a, utility_b)
        return pairwise

    def _predict_scores_using_pairs(self, X, **kwd):
        n_instances, n_objects, n_features = X.shape
        n2 = n_objects * (n_objects - 1)
        pairs = np.empty((n2, 2, n_features))
        scores = np.zeros((n_instances, n_objects))
        for n in range(n_instances):
            for k, (i, j) in enumerate(permutations(range(n_objects), 2)):
                pairs[k] = (X[n, i], X[n, j])
            result = self._predict_pair(pairs[:, 0], pairs[:, 1], only_pairwise=True, **kwd)[:, 0]
            scores[n] += result.reshape(n_objects, n_objects - 1).mean(axis=1)
            del result
        del pairs
        if self._use_zeroth_model:
            scores_zero = self.zero_order_model.predict(X.reshape(-1, n_features))
            scores_zero = scores_zero.reshape(n_instances, n_objects)
            scores = scores + scores_zero
        return scores

    def construct_model(self):
        """
            Construct the :math:`1`-st order and :math:`0`-th order models, which are used to approximate the
            :math:`U_1(x, C(x))` and the :math:`U_0(x)` utilities respectively. For each pair of objects in
            :math:`x_i, x_j \\in Q` :math:`U_1(x, C(x))` we construct :class:`CmpNetCore` with weight sharing to
            approximate a pairwise-matrix. A pairwise matrix with index (i,j) corresponds to the :math:`U_1(x_i,x_j)`
            is a measure of how favorable it is to choose :math:`x_i` over :math:`x_j`. Using this matrix we calculate
            the borda score for each object to calculate :math:`U_1(x, C(x))`. For `0`-th order model we construct
            :math:`\\lvert Q \\lvert` sequential networks whose weights are shared to evaluate the :math:`U_0(x)` for
            each object in the query set :math:`Q`. The output mode is using linear activation.

            Returns
            -------
            model: keras :class:`Model`
                Neural network to learn the FETA utility score
        """

        def create_input_lambda(i):
            return Lambda(lambda x: x[:, i])

        if self.attention_preselected_input is not None:
            input_for_model = self.attention_preselected_input
        else:
            input_for_model = self.input_layer

        if self._use_zeroth_model:
            self.logger.debug('Create 0th order model')
            zeroth_order_outputs = []
            inputs = []
            for i in range(self.n_objects):
                x = create_input_lambda(i)(input_for_model)
                inputs.append(x)
                for hidden in self.hidden_layers_zeroth:
                    x = hidden(x)
                zeroth_order_outputs.append(self.output_node_zeroth(x))
            if self.attention_pooling_config is not None and self.attention_pooling_flavour=="single_pma":
                zeroth_order_scores = zeroth_order_outputs
            else:
                zeroth_order_scores = concatenate(zeroth_order_outputs)
            self.logger.debug('0th order model finished')
        self.logger.debug('Create 1st order model')
        outputs = [list() for _ in range(self.n_objects)]
        for i, j in combinations(range(self.n_objects), 2):
            if self._use_zeroth_model:
                x1 = inputs[i]
                x2 = inputs[j]
            else:
                x1 = create_input_lambda(i)(input_for_model)
                x2 = create_input_lambda(j)(input_for_model)
            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            outputs[i].append(n_g)
            outputs[j].append(n_l)

        # convert rows of pairwise matrix to keras layers:
        outputs = [concatenate(x) for x in outputs]

        # compute utility scores:
        if self.attention_pooling_flavour=="multi_pma" and self.attention_pooling_config is not None:
            add_dim = Reshape(target_shape=(self.n_objects - 1, 1))
            remove_dim = Reshape(target_shape=(1,))
            self.attention_pooling_layers = [instantiate_attention_layer(self.attention_pooling_config) for _ in outputs]
            scores = [remove_dim(self.attention_pooling_layers[x_obj](add_dim(outputs[x_obj]))) for x_obj in range(len(outputs))]
            scores = concatenate(scores)
        elif self.attention_pooling_flavour=="single_pma" and self.attention_pooling_config is not None:
            # build (n_objects x n_objects) matrix out of zeroth-order scores and pairwise scores
            # for each output, separate and concat right half of pairwise scores with 0th order scores
            # (upper right triangle of matrix)
            intermediate_outputs = []
            for i in range(len(outputs)-1):
                # separate right half
                right_half = Lambda(lambda x: x[:, i:])(outputs[i])

                # concat with zeroth order (from left)
                intermediate_outputs.append(concatenate([zeroth_order_scores[i], right_half]))

            # the last elem has no right, just add zeroth order score
            intermediate_outputs.append(zeroth_order_scores[-1])

            # for each output, separate and concat left half with zeroth order scores and right half
            for i in range(1, len(outputs)):
                left_half = Lambda(lambda x: x[:, :i])(outputs[i])

                intermediate_outputs[i] = concatenate([left_half, intermediate_outputs[i]])

            # put the new matrix entries in 1 matrix
            for i in range(len(outputs)):
                add_dim = Reshape(target_shape=(1, self.n_objects))
                intermediate_outputs[i] = add_dim(intermediate_outputs[i])
            outputs = concatenate(intermediate_outputs, axis=1)

            # put the finished matrix through attention (reducing first dimension to 1)
            self.attention_pooling_layers = [instantiate_attention_layer(self.attention_pooling_config)]
            out = self.attention_pooling_layers[0](outputs)

            # order the dimensions correctly
            permutation_layer = Permute((2, 1))
            scores = permutation_layer(out)
            remove_dim = Reshape(target_shape=(self.n_objects,))
            scores = remove_dim(scores)
        else:
            sum_func = lambda s: K.mean(s, axis=1, keepdims=True)
            scores = [Lambda(sum_func)(x) for x in outputs]
            scores = concatenate(scores)

        self.logger.debug('1st order model finished')
        if self._use_zeroth_model:
            if self.attention_pooling_flavour == "mab" and self.attention_pooling_config is not None:
                self.attention_pooling_layers = [instantiate_attention_layer(self.attention_pooling_config)]
                print("scores", scores)
                print("0th scpres", zeroth_order_scores)
                add_dim = Reshape(target_shape=(self.n_objects, 1))
                remove_dim = Reshape(target_shape=(self.n_objects,))
                scores = remove_dim(self.attention_pooling_layers[0]([add_dim(scores), add_dim(zeroth_order_scores)]))
            elif self.attention_pooling_flavour == "single_pma" and self.attention_pooling_config is not None:
                # the zeroth order scores are already accounted for
                pass
            else:
                scores = add([scores, zeroth_order_scores])
        model = Model(inputs=self.input_layer, outputs=scores)

        self.logger.debug('Compiling complete model...')
        if self.loss_function_requires_x_values:
            self.loss_function = self.loss_function(self.input_layer)

        additional_metric = [metric(self.input_layer) for metric in self.metrics_requiring_x]
        metrics = self.metrics.copy() if self.metrics is not None else []
        metrics.extend(additional_metric)

        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=metrics)
        return model

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        """
            Fit a generic preference learning model on a provided set of queries.
            The provided queries can be of a fixed size (numpy arrays).

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in form of rankings or choices for given objects
            epochs : int
                Number of epochs to run if training for a fixed query size
            callbacks : list
                List of callbacks to be called during optimization
            validation_split : float (range : [0,1])
                Percentage of instances to split off to validate on
            verbose : bool
                Print verbose information
            **kwd :
                Keyword arguments for the fit function
        """
        self.logger.debug('Enter fit function...')

        X, Y = self.sub_sampling(X, Y)
        self.model = self.construct_model()
        self.logger.debug('Starting gradient descent...')

        self.set_up_callbacks(callbacks)

        self.model.fit(x=X, y=Y, batch_size=self.batch_size, epochs=epochs, callbacks=callbacks,
                       validation_split=validation_split, verbose=verbose, **kwd)
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)

    def set_up_callbacks(self, callbacks):
        attention_outputs = []
        if self.attention_preselection_layers is not None:
            for preselection_layer in self.attention_preselection_layers:
                outputs = preselection_layer.get_attention_layer_inputs_outputs()

                for output_num in range(len(outputs)):
                    output = outputs[output_num]
                    output["name"] = "feta_preselection_{}_{}".format(output_num, output["name"])

                attention_outputs.extend(outputs)
        if self.attention_pooling_layers is not None:
            for pooling_layer in self.attention_pooling_layers:
                outputs = pooling_layer.get_attention_layer_inputs_outputs()

                for output_num in range(len(outputs)):
                    output = outputs[output_num]
                    output["name"] = "feta_pooling_{}_{}".format(output_num, output["name"])

                attention_outputs.extend(outputs)
        configure_callbacks(callbacks, attention_outputs)

    def sub_sampling(self, X, Y):
        if self._n_objects > self.max_number_of_objects:
            bucket_size = int(self._n_objects / self.max_number_of_objects)
            idx = self.random_state.randint(bucket_size,
                                            size=(len(X), self.n_objects))
            # TODO: subsampling multiple rankings
            idx += np.arange(start=0, stop=self._n_objects, step=bucket_size)[
                   :self.n_objects]
            X = X[np.arange(X.shape[0])[:, None], idx]
            Y = Y[np.arange(X.shape[0])[:, None], idx]
            tmp_sort = Y.argsort(axis=-1)
            Y = np.empty_like(Y)
            Y[np.arange(len(X))[:, None], tmp_sort] = np.arange(self.n_objects)
        return X, Y

    def _predict_scores_fixed(self, X, **kwargs):
        n_objects = X.shape[-2]
        self.logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        if self.n_objects != n_objects:
            scores = self._predict_scores_using_pairs(X, **kwargs)
        else:
            scores = self.model.predict(X, **kwargs)
        self.logger.info("Done predicting scores")
        return scores

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3,
                               batch_size=128, **point):
        """
            Set tunable parameters of the FETA-network to the values provided.

            Parameters
            ----------
            n_hidden: int
                Number of hidden layers used in the scoring network
            n_units: int
                Number of hidden units in each layer of the scoring network
            reg_strength: float
                Regularization strength of the regularizer function applied to the `kernel` weights matrix
            learning_rate: float
                Learning rate of the stochastic gradient descent algorithm used by the network
            batch_size: int
                Batch size to use during training
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._pairwise_model = None
        self._zero_order_model = None
        self.set_up_input_layer()
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))

    def clear_memory(self, **kwargs):
        """
            Clear the memory, restores the currently fitted model back to prevent memory leaks.

            Parameters
            ----------
            **kwargs :
                Keyword arguments for the function
        """
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)

            self._pairwise_model = None
            self._zero_order_model = None
            self.optimizer = self.optimizer.from_config(self._optimizer_config)
            self.set_up_input_layer()
            self._construct_layers(kernel_regularizer=self.kernel_regularizer,
                                   kernel_initializer=self.kernel_initializer,
                                   activation=self.activation, **self.kwargs)
            self.model = self.construct_model()
            self.model.load_weights(self.hash_file)
        else:
            self.logger.info("Cannot clear the memory")
