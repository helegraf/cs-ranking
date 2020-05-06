import copy
import logging
import math

from keras.engine import Layer
from keras import backend as K, Model, Input
from keras.layers import Dense, TimeDistributed
from keras_layer_normalization import LayerNormalization

from csrank.tensorflow_util import repeat_vector_along_new_axis


def validate_args(class_name, inputs_):
    if isinstance(inputs_, list) and len(inputs_) > 3:
        raise ValueError('{} layer input must be a list of length 1, 2 or 3, representing [query], [query, value] or'
                         '[query, value, key] or a single element representing query. The length of the given input is '
                         '{}'.format(class_name, len(inputs_)))


def extract_args(inputs_):
    if not isinstance(inputs_, list):
        # if only one element given, it is the query
        return inputs_, inputs_, inputs_
    else:
        # the first element is the query
        query = inputs_[0]

        # if no explicit key given, key and query are the same
        if len(inputs_) > 1:
            key = inputs_[1]
        else:
            key = query

        # if no explicit value given, value and key are the same
        if len(inputs_) > 2:
            value = inputs_[2]
        else:
            value = key
        return query, key, value


def extract_input_sizes(input_shape):
    if not isinstance(input_shape, list):
        n = m = input_shape[1]
        d_q = d_k = d_v = input_shape[2]
    else:
        n = input_shape[0][1]
        d_q = input_shape[0][2]

        if len(input_shape) > 1:
            m = input_shape[1][1]
            d_k = input_shape[1][2]
        else:
            m = n
            d_k = d_q

        if len(input_shape) > 2:
            d_v = input_shape[2][2]
        else:
            d_v = d_k
    return d_q, d_k, d_v, n, m


class AttentionLayer(Layer):
    def get_attention_layer_inputs_outputs(self):
        raise NotImplementedError()

    # only works if the layer itself is ONLY using layer operations, no pure tensor operations
    # def as_model(self, input_shape):
    #     if isinstance(input_shape, list):
    #         inputs = [Input(shape=shape) for shape in input_shape]
    #     else:
    #         inputs = Input(shape=input_shape)
    #
    #     # if not built, add batch size and build so call can be called
    #     if not self.built:
    #         if isinstance(input_shape, list):
    #             new_shape = [(None,) + elem for elem in input_shape]
    #         else:
    #             new_shape = (None,) + input_shape
    #         self.build(new_shape)
    #
    #     outputs = self.call(inputs)
    #
    #     return Model(inputs=inputs, outputs=outputs)


class BaseAttention(AttentionLayer):
    def __init__(self, activation=None, biased=False, **kwargs):
        super(BaseAttention, self).__init__(**kwargs)

        activations = {
            'elu': K.elu,
            'softmax': K.softmax,
            'softplus': K.softplus,
            'softsign': K.softsign,
            'relu': K.relu,
            'tanh': K.tanh,
            'sigmoid': K.sigmoid,
            'hard_sigmoid': K.hard_sigmoid
        }
        if activation is not None and activation not in activations.keys():
            raise ValueError("Given activation {} is invalid; allowed: {}".format(activation, activations.keys()))
        if activation is not None and isinstance(activation, str):
            self.activation = activations[activation]
        else:
            self.activation = activation

        self.biased = biased

        self.query = None
        self.key = None
        self.scores = None

    def call(self, inputs_, **kwargs):
        # handle args
        validate_args(self.__class__.__name__, inputs_)
        self.query, self.key, value = extract_args(inputs_)

        # calculate scores
        self.scores = K.softmax(self._compute_scores(self.query, self.key))

        # compute complete result
        return K.batch_dot(self.scores, value)

    def _compute_scores(self, query, key):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            # if one element is given, the result has the same size as it
            return input_shape
        else:
            # otherwise batch size and the number of rows coincide with the query
            batch_size = input_shape[0][0]
            row_dim = input_shape[0][1]

            # the last dimension (number of columns) is the number of columns of the value
            if len(input_shape) == 3:
                column_dim = input_shape[2][2]
            else:
                column_dim = input_shape[0][2]

            output_shape = (batch_size, row_dim, column_dim)
            return output_shape

    def get_attention_layer_inputs_outputs(self):
        return self.query, self.key, self.scores


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, scale=True, weighted=False, weights_initializer="glorot_uniform", factor=None,
                 sordoni_biased=False, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

        self.logger = logging.getLogger(ScaledDotProductAttention.__name__)

        # hyperparameters
        self.scale = scale
        self.weighted = weighted
        self.factor = factor if factor is not None else 1
        self.weights_initializer = weights_initializer
        self.sordoni_biased = sordoni_biased

        # weights
        self.w_a = None
        self.b = None
        self.s_b = None

    def build(self, input_shape):
        d_q, d_k, _, n, m = extract_input_sizes(input_shape)

        if self.scale:
            self.factor = 1 / math.sqrt(d_k)

        if self.weighted:
            self.w_a = self.add_weight(name='w_a',
                                       shape=(d_q, d_q),
                                       initializer=self.weights_initializer)
            self.logger.debug("Init weight with", self.w_a)

        if self.biased:
            self.b = self.add_weight(name="b",
                                     shape=(n, m),
                                     initializer=self.weights_initializer)

        if self.sordoni_biased:
            self.s_b = self.add_weight(name="s_b",
                                       shape=(m, d_k),
                                       initializer=self.weights_initializer)

        super(ScaledDotProductAttention, self).build(input_shape)

    def _compute_scores(self, query, key):
        if self.weighted:
            weighted_query = K.dot(query, self.w_a)
        else:
            weighted_query = query

        if self.sordoni_biased:
            biased_key = key + self.s_b
        else:
            biased_key = key

        product = self.factor * K.batch_dot(weighted_query, K.permute_dimensions(biased_key, pattern=(0, 2, 1)))

        if self.biased:
            biased_product = product + self.b
        else:
            biased_product = product

        if self.activation is not None:
            activated_product = self.activation(biased_product)
        else:
            activated_product = biased_product

        return activated_product


class AdditiveAttention(BaseAttention):

    def __init__(self, flavour='Luong', weights_initializer='glorot_uniform', **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        allowed_flavours = ['Luong', 'Bahdanau']
        if flavour not in allowed_flavours:
            raise ValueError("Given flavour {} is invalid; allowed: {}".format(flavour, allowed_flavours))

        self.weights_initializer = weights_initializer

        # non-linear activation is needed
        if self.activation is None:
            self.activation = K.tanh

        # hyperparameters
        self.flavour = flavour

        # weights
        self.w_a = None
        self.w_1 = None
        self.w_2 = None
        self.v_a = None
        self.b = None

    def build(self, input_shape):
        d_q, d_k, d_v, m, n = extract_input_sizes(input_shape)

        # check compatible
        if d_q != d_k:
            raise ValueError("Additive attention needs coinciding feature dimension for query and key."
                             " Given feature dimensions are: query: {} and key: {}".format(d_q, d_k))

        if self.flavour == 'Luong':
            self.w_a = self.add_weight(name="w_a",
                                       shape=(n, n),
                                       initializer=self.weights_initializer)
        else:
            # flavour = Bahdanau
            self.w_1 = self.add_weight(name="w_1",
                                       shape=(n, n),
                                       initializer=self.weights_initializer)
            self.w_2 = self.add_weight(name="w_2",
                                       shape=(m, m),
                                       initializer=self.weights_initializer)

        if self.biased:
            self.b = self.add_weight(name='b',
                                     shape=(m, m),
                                     initializer=self.weights_initializer)

        self.v_a = self.add_weight(name="v_a",
                                   shape=(n, n),
                                   initializer=self.weights_initializer)

        super(AdditiveAttention, self).build(input_shape)

    def _compute_scores(self, query, key):
        if self.flavour == 'Luong':
            dot = K.dot(K.permute_dimensions(K.sum(query[:, :, None] + key[:, None], axis=-1), (0, 2, 1)), self.w_a)
            concatenated = K.permute_dimensions(dot, (0, 2, 1))
        else:
            # flavour = Bahdanau
            dot_prod_query = K.permute_dimensions(K.dot(K.permute_dimensions(query, (0, 2, 1)), self.w_1), (0, 2, 1))
            dot_prod_key = K.permute_dimensions(K.dot(K.permute_dimensions(key, (0, 2, 1)), self.w_2), (0, 2, 1))
            concatenated = K.sum(dot_prod_query[:, :, None] + dot_prod_key[:, None], axis=-1)

        if self.biased:
            concatenated = concatenated + self.b

        return K.permute_dimensions(K.dot(K.permute_dimensions(self.activation(concatenated), (0, 2, 1)),
                                          self.v_a), (0, 2, 1))


class SimilarityAttention(BaseAttention):

    def __init__(self, **kwargs):
        super(SimilarityAttention, self).__init__(**kwargs)

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError("Similarity Attention cannot have any arguments, received arguments {}"
                             .format(kwargs.keys()))

    def _compute_scores(self, query, key):
        query_ = K.l2_normalize(query, axis=2)
        key_ = K.l2_normalize(key, axis=2)
        return K.batch_dot(query_, K.permute_dimensions(key_, (0, 2, 1)))


class MultiHeadAttention(AttentionLayer):
    def __init__(self, attention, num_heads=None, weights_initializer="glorot_uniform", **kwargs):
        # hyperparameters
        self.num_heads = num_heads
        self.attention = instantiate_attention_layer(attention)
        self.weights_initializer = weights_initializer

        # runtime shape information
        self.d_q_prime = None
        self.d_k_prime = None
        self.d_v_prime = None
        self.w_0 = None
        self.w_q = None
        self.w_k = None
        self.w_v = None

        # retain attention information
        self.attention_heads = None

        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        d_q, d_k, d_v, m, n = extract_input_sizes(input_shape)

        if self.num_heads is None:
            self.num_heads = d_q

        if not d_q % self.num_heads == 0 \
                or not d_k % self.num_heads == 0 \
                or not d_v % self.num_heads == 0:
            raise ValueError("The inputs to the multihead attention layer have feature lengths that cannot be divided "
                             "by the number of heads. Feature lengths d_q: {}, d_k: {}, d_v: {}, num heads: {}"
                             .format(d_q, d_k, d_v, self.num_heads))

        self.d_q_prime = int(d_q / self.num_heads)
        self.d_k_prime = int(d_k / self.num_heads)
        self.d_v_prime = int(d_v / self.num_heads)

        self.w_0 = self.add_weight(name='w_0',
                                           shape=(d_v, d_v),
                                           initializer=self.weights_initializer)

        self.w_q = self.add_weight(name='w_q',
                                   shape=(d_q, d_q),
                                   initializer=self.weights_initializer)

        self.w_k = self.add_weight(name='w_k',
                                   shape=(d_k, d_k),
                                   initializer=self.weights_initializer)

        self.w_v = self.add_weight(name='w_v',
                                   shape=(d_v, d_v),
                                   initializer=self.weights_initializer)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs_, **kwargs):
        # get inputs
        validate_args(self.__class__.__name__, inputs_)
        query, key, value = extract_args(inputs_)

        query_ = K.dot(query, self.w_q)
        key_ = K.dot(key, self.w_k)
        value_ = K.dot(value, self.w_v)

        self.attention_heads = [self.attention([query_[:, :, i * self.d_q_prime: (i + 1) * self.d_q_prime],
                                                key_[:, :, i * self.d_k_prime: (i + 1) * self.d_k_prime],
                                                value_[:, :, i * self.d_v_prime: (i + 1) * self.d_v_prime]])
                                for i in range(self.num_heads)]

        result = K.concatenate(self.attention_heads, axis=2)

        return K.dot(result, self.w_0)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            return input_shape
        else:
            batch_size = input_shape[0][0]
            row_dim = input_shape[0][1]

            if len(input_shape) == 3:
                column_dim = input_shape[2][2]
            else:
                column_dim = input_shape[0][2]

            return batch_size, row_dim, column_dim

    def get_attention_layer_inputs_outputs(self):
        return [head.get_attention_layer_inputs_outputs() for head in self.attention_heads]


class MAB(AttentionLayer):
    def __init__(self, multi_head, depth_rff=0, rff_config={"activation": "relu"}, **kwargs):
        """

        Parameters
        ----------
        multi_head : MultiHeadAttention
        depth_rff : int
        rff_config: dict
        kwargs :
        """
        super(MAB, self).__init__(**kwargs)

        # layers
        self.layer_norm_outer = LayerNormalization()
        self.layer_norm_inner = LayerNormalization()
        self.multi_head = instantiate_attention_layer(multi_head)

        # rff configuration
        self.depth_rff = depth_rff
        self.rff_config = rff_config
        self.rff = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("MAB layer needs two different inputs (list), to use the same input twice use SAB.")

        d_q, d_k, d_v, m, n = extract_input_sizes(input_shape)
        # self.rff = [TimeDistributed(Dense(units=d_v, activation="relu")) for _ in range(self.depth_rff)]

        if "units" not in self.rff_config.keys():
            self.rff_config = {"units": d_v, **self.rff_config}
        rff_config_final = copy.deepcopy(self.rff_config)
        rff_config_final["units"] = d_v

        self.rff = [TimeDistributed(Dense(**self.rff_config)) for _ in range(self.depth_rff)]
        self.rff.append(TimeDistributed(Dense(**rff_config_final)))
        self._layers.extend(self.rff)

        super(MAB, self).build(input_shape)

    def call(self, inputs_, **kwargs):
        x, y = inputs_

        h = self.layer_norm_inner(x + self.multi_head([x, y, y]))
        rff_h = h
        for layer in self.rff:
            rff_h = layer(rff_h)

        return self.layer_norm_outer(h + rff_h)

    def compute_output_shape(self, input_shape):
        return self.multi_head.compute_output_shape([input_shape[0], input_shape[1], input_shape[1]])

    def get_attention_layer_inputs_outputs(self):
        return self.multi_head.get_attention_layer_inputs_outputs()


class SAB(AttentionLayer):
    def __init__(self, mab, **kwargs):
        super(SAB, self).__init__(**kwargs)
        self.mab = instantiate_attention_layer(mab)

    def build(self, input_shape):
        super(SAB, self).build(input_shape)

    def call(self, inputs_, **kwargs):
        return self.mab([inputs_, inputs_])

    def compute_output_shape(self, input_shape):
        return self.mab.compute_output_shape([input_shape, input_shape])

    def get_attention_layer_inputs_outputs(self):
        return self.mab.get_attention_layer_inputs_outputs()


class ISAB(AttentionLayer):
    def __init__(self, num_inducing_points_m, mab_inner, mab_outer, **kwargs):
        super(ISAB, self).__init__(**kwargs)

        self.num_inducing_points_m = num_inducing_points_m
        self.mab_outer = instantiate_attention_layer(mab_outer)
        self.mab_inner = instantiate_attention_layer(mab_inner)

        self.i = None

    def build(self, input_shape):
        self.i = self.add_weight(name="I",
                                 shape=(self.num_inducing_points_m, input_shape[2]),
                                 initializer='glorot_uniform')
        super(ISAB, self).build(input_shape)

    def call(self, inputs_, **kwargs):
        repeated_i = repeat_vector_along_new_axis(self.i, K.shape(inputs_)[0])
        h = self.mab_inner([repeated_i, inputs_])
        return self.mab_outer([inputs_, h])

    def compute_output_shape(self, input_shape):
        inner_output_shape = self.mab_inner.compute_output_shape(
            [(input_shape[0], self.i.shape[0], self.i.shape[1]), input_shape])
        return self.mab_outer.compute_output_shape([input_shape, inner_output_shape])

    def get_attention_layer_inputs_outputs(self):
        return {"inner": self.mab_inner.get_attention_layer_inputs_outputs(),
                "outer": self.mab_outer.get_attention_layer_inputs_outputs()}


class PMA(AttentionLayer):
    def __init__(self, k, mab, depth_rff=1, rff_config={"activation": "relu"}, **kwargs):
        """

        Parameters
        ----------
        k : int
        mab : MAB
        depth_rff : int
        rff_config : dict
            Configuration for the rff-layers (Dense keras layer). The number of units does not have to be given. If
            given, it is ignored in the final rff layer.
        kwargs : Any
        """
        super(PMA, self).__init__(**kwargs)

        self.k = k
        self.mab = instantiate_attention_layer(mab)

        self.depth_rff = depth_rff
        self.rff_config = rff_config

        self.rff = None
        self.S = None

    def build(self, input_shape):
        self.S = self.add_weight(name='S',
                                 shape=(self.k, input_shape[2]),
                                 initializer='glorot_uniform')

        if "units" not in self.rff_config.keys():
            self.rff_config = {"units": input_shape[2], **self.rff_config}
        rff_config_final = copy.deepcopy(self.rff_config)
        rff_config_final["units"] = input_shape[2]

        self.rff = [TimeDistributed(Dense(**self.rff_config)) for _ in range(self.depth_rff)]
        self.rff.append(TimeDistributed(Dense(**rff_config_final)))
        self._layers.extend(self.rff)

        super(PMA, self).build(input_shape)

    def call(self, inputs_, **kwargs):
        rff_out = inputs_

        for dense_layer in self.rff:
            rff_out = dense_layer(rff_out)

        repeat_s = repeat_vector_along_new_axis(self.S, K.shape(inputs_)[0])
        return self.mab([repeat_s, rff_out])

    def compute_output_shape(self, input_shape):
        shape_1 = (input_shape[0], self.k, input_shape[2])
        mab_out = self.mab.compute_output_shape([shape_1, input_shape])
        return mab_out

    def get_attention_layer_inputs_outputs(self):
        return self.mab.get_attention_layer_inputs_outputs()


attention_blocks = {
    "ScaledDotProductAttention": ScaledDotProductAttention,
    "SimilarityAttention": SimilarityAttention,
    "AdditiveAttention": AdditiveAttention,
    "MultiHeadAttention": MultiHeadAttention,
}

set_transformer_blocks = {
    "MAB": MAB,
    "SAB": SAB,
    "ISAB": ISAB,
    "PMA": PMA
}


def instantiate_attention_layer(layer):
    """
    Instantiates a layer, which can be given as a layer-object or as a dictionary containing the layer name and options.
    If the given object already is a layer, just returns the layer.

    Parameters
    ----------
    layer : dict or Layer
            to be convert to a layer object

    Returns
    -------
    Layer
            an instantiation of the given layer, if it has not been instantiated before

    Examples
    ----------
    The layer can be a dict or object.

    >>> layer = ScaledDotProductAttention(weighted=False, biased=True)
    >>> created_layer = instantiate_attention_layer(layer)
    >>> layer == created_layer
    True

    >>> layer = {"ScaledDotProductAttention": {"weighted": True, "biased": True}}
    >>> created_layer = instantiate_attention_layer(layer)
    >>> created_layer.weighted
    True
    >>> created_layer.biased
    True
    """
    if isinstance(layer, dict):
        layer_name = list(layer.keys())[0]
        return {**attention_blocks, **set_transformer_blocks}[layer_name](**layer[layer_name])
    else:
        return layer
