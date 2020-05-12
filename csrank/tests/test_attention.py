import datetime
import multiprocessing

import numpy as np
from keras.callbacks import TensorBoard

np.random.seed(1)
import pytest
from keras import Input, Model
from keras.engine import Layer
from keras.layers import Dense, Concatenate
from keras.optimizers import SGD
from keras_layer_normalization import LayerNormalization
from numpy.testing import assert_almost_equal
import keras.backend as K
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

from csrank import DiscreteChoiceDatasetGenerator, ChoiceDatasetGenerator, ObjectRankingDatasetGenerator, \
    FATEObjectRanker, FETAObjectRanker
from csrank.attention.set_transformer.modules import ScaledDotProductAttention, MultiHeadAttention, MAB, SAB, ISAB, PMA, \
    SimilarityAttention, AdditiveAttention, instantiate_attention_layer
from csrank.attention.set_transformer.util import split_features_to_batch, combine_batch_to_features
from csrank.callbacks import WeightPrinterCallback, AdvancedTensorBoard
from csrank.choicefunction.set_transformer_choice import SetTransformerChoice
from csrank.discretechoice.set_transformer_discrete_choice import SetTransformerDiscreteChoice
from csrank.objectranking.set_transformer_object_ranker import SetTransformerObjectRanker


def soft_max(x):
    exp = np.exp(x)
    total_sum = np.sum(exp, axis=-1)[:, None]
    quotient = exp / total_sum
    return quotient


def attention_simple(x, y=None, z=None):
    y = x if y is None else y
    z = y if z is None else z
    return np.matmul(soft_max(np.matmul(x, np.transpose(y))), z)


def make_predictions_with_layer \
                (attention_layer, loss_function='mean_squared_error', data=None, obj_query=None, feat_query=None,
                 obj_key=None, feat_key=None, obj_value=None,
                 feat_value=None, result=[], epochs=1, learning_rate=.01):
    inputs = [Input(shape=(obj_query, feat_query))]

    if obj_key is not None and feat_key is not None:
        inputs.append(Input(shape=(obj_key, feat_key)))

    if obj_value is not None and feat_value is not None:
        inputs.append(Input(shape=(obj_value, feat_value)))

    output = attention_layer(inputs)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())

    model.compile(optimizer=SGD(learning_rate=learning_rate), loss=loss_function)

    # get model weights
    weights_before = np.asarray(model.get_weights())

    model.fit(data, result, epochs=epochs)

    weights_after = np.asarray(model.get_weights())

    predictions = model.predict(data)

    print("Model weights before:")
    print(weights_before)
    print("Model weights after training:")
    print(weights_after)
    print("Model weights difference:")
    difference = np.abs(weights_after - weights_before)
    print(difference)

    return predictions


@pytest.fixture(scope="module")
def seed_execution():
    seed(1)
    set_random_seed(1)
    np.random.seed(1)


@pytest.fixture(scope="module")
def set_tf_config():
    # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    #                         allow_soft_placement=True, log_device_placement=False,
    #                         device_count={'CPU': multiprocessing.cpu_count() - 2})
    # sess = tf.Session(config=config)
    # K.set_session(sess)
    pass


@pytest.fixture(scope="module",
                params=[("ones", 2, 1),
                        ("zeros", 2, 1),
                        ("zeros", 1, 2),
                        ("random", 3, 1),
                        ("random", 3, 2),
                        (2, 1, 5, 1),
                        (2, 1, 5, 1, 5, 2)],
                ids=["ones 2 objects 1 feature",
                     "zeros 2 objects 1 features",
                     "zeros 1 object 2 features",
                     "random 3 objects 1 feature",
                     "random 3 objects 2 features",
                     "query: 2 obj 1 feat, key: 5 obj 1 feat",
                     "query: 2 obj 1 feat, key: 5 obj 1 feat, value: 5 obj 2 feat"])
def input_data(request):
    if len(request.param) == 3:
        case, obj_query, feat_query = request.param

        if case == "zeros":
            data = np.asarray([[[0 for _ in range(feat_query)] for _ in range(obj_query)]])
            result = np.asarray([attention_simple(data[i]) for i in range(len(data))])

        elif case == "ones":
            data = np.asarray([[[1 for _ in range(feat_query)] for _ in range(obj_query)]])
            result = np.asarray([attention_simple(data[i]) for i in range(len(data))])
        else:
            random = np.random.RandomState(seed=feat_query * obj_query)
            data = random.randint(low=0, high=10, size=(100, obj_query, feat_query))
            result = np.asarray([attention_simple(data[i]) for i in range(len(data))])

        return {"obj_query": obj_query, "feat_query": feat_query, "data": data, "result": result}

    elif len(request.param) == 4:
        obj_query, feat_query, obj_key, feat_key = request.param

        random = np.random.RandomState(seed=obj_query * feat_query * obj_key * feat_key)

        data_1 = random.randint(low=0, high=10, size=(100, obj_query, feat_query))
        data_2 = random.randint(low=0, high=10, size=(100, obj_key, feat_key))
        data = [data_1, data_2]
        result = np.asarray([attention_simple(data_1[i], data_2[i]) for i in range(len(data_1))])

        return {"obj_query": obj_query, "feat_query": feat_query, "obj_key": obj_key, "feat_key": feat_key,
                "data": data, "result": result}

    else:
        obj_query, feat_query, obj_key, feat_key, obj_value, feat_value = request.param

        random = np.random.RandomState(seed=obj_query * feat_query * obj_key * feat_key * obj_value * feat_value)

        data_1 = random.randint(low=0, high=10, size=(100, obj_query, feat_query))
        data_2 = random.randint(low=0, high=10, size=(100, obj_key, feat_key))
        data_3 = random.randint(low=0, high=10, size=(100, obj_value, feat_value))
        data = [data_1, data_2, data_3]
        result = np.asarray([attention_simple(data_1[i], data_2[i], data_3[i]) for i in range(len(data_1))])

        return {"obj_query": obj_query, "feat_query": feat_query, "obj_key": obj_key, "feat_key": feat_key,
                "obj_value": obj_value, "feat_value": feat_value, "data": data, "result": result}


def test_topological_sort_error_with_layer(input_data):
    class SimpleAttention(Layer):
        def call(self, inputs_, **kwargs):
            return K.batch_dot(K.batch_dot(inputs_[0], K.permute_dimensions(inputs_[1], pattern=(0, 2, 1))), inputs_[2])

        def compute_output_shape(self, input_shape):
            output_shape = (input_shape[0][0], input_shape[0][1], input_shape[2][2])
            return output_shape

    # this test shows that even the most simple graph cannot be sorted topologically (even with hinged rank loss)
    # this at least is the case on CPU
    data = input_data["data"]
    n = input_data["obj_query"]
    d = input_data["feat_query"]

    if len(data) == 1:
        inputs = [Input(shape=(n, d)), Input(shape=(n, d)), Input(shape=(n, d))]
        outputs = SimpleAttention()(inputs)

        model = Model(inputs=inputs, outputs=outputs)

        # graph cannot be sorted with mean squared error OR hinged rank loss
        model.compile(optimizer="SGD", loss="mean_squared_error")
        model.fit(x=[data, data, data], y=input_data["result"], epochs=100)


def test_topological_sort_error(input_data):
    # apparently THIS can be sorted topologically - maybe the problem is ifs/ elses?
    data = input_data["data"]

    if len(data) == 1:
        query = K.placeholder(data.shape)
        key = K.placeholder(data.shape)
        value = K.placeholder(data.shape)

        scores = 1 * K.batch_dot(query, K.permute_dimensions(key, (0, 2, 1)))
        result = K.batch_dot(K.softmax(scores), value)

        with tf.Session() as sess:
            final = sess.run(fetches=[result], feed_dict={query: data, key: data, value: data})
            print(final)


def test_dot_product_attention_simple(input_data):
    attention_layer = ScaledDotProductAttention(scale=False, weighted=False, weights_initializer="identity")
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert_almost_equal(actual=predictions, desired=input_data["result"], decimal=3)


def test_dot_product_attention_scaled(input_data):
    attention_layer = ScaledDotProductAttention(scale=True, weighted=False, weights_initializer="identity", factor=1)
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert_almost_equal(actual=predictions, desired=input_data["result"], decimal=0)


def test_dot_product_attention_scaled_weighted(input_data):
    attention_layer = ScaledDotProductAttention(scale=True, weighted=True, weights_initializer="identity")
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert_almost_equal(actual=predictions, desired=input_data["result"], decimal=0)


def test_dot_product_attention_sordoni_bias(input_data):
    attention_layer = ScaledDotProductAttention(scale=False, weighted=False, weights_initializer="identity",
                                                sordoni_biased=True)
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_dot_product_attention_biased(input_data):
    attention_layer = ScaledDotProductAttention(scale=False, weighted=False, weights_initializer="identity",
                                                sordoni_biased=False, biased=True)
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_dot_product_attention_activated(input_data):
    attention_layer = ScaledDotProductAttention(scale=False, weighted=False, weights_initializer="identity",
                                                sordoni_biased=False, biased=False, activation='tanh')
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_similarity_attention(input_data):
    attention_layer = SimilarityAttention()
    predictions = make_predictions_with_layer(attention_layer, **input_data)

    print(predictions)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_concat_attention_luong(input_data):
    # additive attention can only be used for same-sized query, key and value
    if len(input_data) == 4:
        attention_layer = AdditiveAttention(flavour='Luong')
        predictions = make_predictions_with_layer(attention_layer, **input_data)

        print(predictions)

        assert not np.any(np.isnan(predictions))
        if not isinstance(input_data["data"], list):
            assert input_data["data"].shape == predictions.shape


def test_concat_attention_luong_biased(input_data):
    # additive attention can only be used for same-sized query, key and value
    if len(input_data) == 4:
        attention_layer = AdditiveAttention(flavour='Luong', biased=True)
        predictions = make_predictions_with_layer(attention_layer, **input_data)

        print(predictions)

        assert not np.any(np.isnan(predictions))
        if not isinstance(input_data["data"], list):
            assert input_data["data"].shape == predictions.shape


def test_concat_attention_bahdanau(input_data):
    # additive attention can only be used for same-sized query, key
    if len(input_data) == 4:
        attention_layer = AdditiveAttention(flavour='Bahdanau')
        predictions = make_predictions_with_layer(attention_layer, **input_data)

        print(predictions)

        assert not np.any(np.isnan(predictions))
        if not isinstance(input_data["data"], list):
            assert input_data["data"].shape == predictions.shape


def test_concat_attention_bahdanau_biased(input_data):
    # additive attention can only be used for same-sized query, key
    if len(input_data) == 4:
        attention_layer = AdditiveAttention(flavour='Bahdanau', biased=True)
        predictions = make_predictions_with_layer(attention_layer, **input_data)

        print(predictions)

        assert not np.any(np.isnan(predictions))
        if not isinstance(input_data["data"], list):
            assert input_data["data"].shape == predictions.shape


def test_multi_head_attention(input_data):
    attention_layer = MultiHeadAttention(attention_config=ScaledDotProductAttention(scale=False, weighted=False))
    predictions = make_predictions_with_layer(attention_layer, **input_data, epochs=10, learning_rate=1e-4)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_multi_head_attention_weighted_biased_sordoni(input_data):
    attention_layer = MultiHeadAttention(
        attention_config=ScaledDotProductAttention(scale=False, weighted=True, biased=True, sordoni_biased=True))
    predictions = make_predictions_with_layer(attention_layer, **input_data, epochs=10, learning_rate=1e-2)

    assert not np.any(np.isnan(predictions))
    if not isinstance(input_data["data"], list):
        assert input_data["data"].shape == predictions.shape


def test_mab(input_data):
    # this test doesn't check values just if operational and is only applicable for query or query/ key data
    if len(input_data) == 4 or len(input_data) == 6:
        data = input_data["data"]
        num_objects = input_data["obj_query"]
        num_features = input_data["feat_query"]
        result = input_data["result"]

        attention_layer = MAB(MultiHeadAttention(
            attention_config=ScaledDotProductAttention(scale=False, weighted=False)),
                              depth_rff=2, rff_config={"units": 4})

        if len(input_data) == 4:
            inputs = Input(shape=(num_objects, num_features))
            output = attention_layer([inputs, inputs])
        else:
            obj_key = input_data["obj_key"]
            feat_key = input_data["feat_key"]
            inputs = [Input(shape=(num_objects, num_features)), Input(shape=(obj_key, feat_key))]
            output = attention_layer(inputs)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        model.compile(optimizer=SGD(learning_rate=1), loss='mean_squared_error')

        weights_before = model.get_weights()

        model.fit(data, result, epochs=10)

        weights_after = model.get_weights()

        predictions = model.predict(data)

        print("predictions")
        print(predictions)
        print("true")
        print(result)

        print("Model weights before:")
        print(weights_before)
        print("Model weights after training:")
        print(weights_after)

        assert not np.any(np.isnan(predictions))
        if not isinstance(data, list):
            assert data.shape == predictions.shape


def test_sab(input_data):
    if len(input_data) == 4:
        data = input_data["data"]
        num_objects = input_data["obj_query"]
        num_features = input_data["feat_query"]
        result = input_data["result"]

        attention_layer = SAB(MAB(MultiHeadAttention(
            attention_config=ScaledDotProductAttention(scale=False, weighted=False))))
        inputs = Input(shape=(num_objects, num_features))
        output = attention_layer(inputs)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        model.compile(optimizer=SGD(learning_rate=1e-4), loss='mean_squared_error')

        weights_before = model.get_weights()

        model.fit(data, result, epochs=10)

        weights_after = model.get_weights()

        predictions = model.predict(data)

        print("Model weights before:")
        print(weights_before)
        print("Model weights after training:")
        print(weights_after)

        assert not np.any(np.isnan(predictions))
        assert data.shape == predictions.shape


def test_isab(input_data):
    if len(input_data) == 4:
        data = input_data["data"]
        num_objects = input_data["obj_query"]
        num_features = input_data["feat_query"]
        result = input_data["result"]

        attention_layer = \
            ISAB(5, MAB(MultiHeadAttention(attention_config=ScaledDotProductAttention(scale=True, weighted=False))),
                 MAB(MultiHeadAttention(attention_config=ScaledDotProductAttention(scale=True, weighted=False)))
                 )
        inputs = Input(shape=(num_objects, num_features))
        output = attention_layer(inputs)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        model.compile(optimizer=SGD(learning_rate=1e-4), loss='mean_squared_error')

        weights_before = np.asarray(model.get_weights())

        model.fit(data, result, epochs=10)

        weights_after = np.asarray(model.get_weights())

        predictions = model.predict(data)

        print("Model weights before:")
        print(weights_before)
        print("Model weights after training:")
        print(weights_after)
        print("Model weights difference:")
        difference = np.abs(weights_after - weights_before)
        print(difference)

        assert not np.any(np.isnan(predictions))
        assert data.shape == predictions.shape


def test_pma(input_data):
    if len(input_data) == 4:
        data = input_data["data"]
        num_objects = input_data["obj_query"]
        num_features = input_data["feat_query"]
        result = input_data["result"]

        # can only be tested with this number because otherwise the training data won't fit
        k = num_objects

        attention_layer = PMA(k,
                              mab=MAB(
                                  multi_head=MultiHeadAttention(attention_config=ScaledDotProductAttention(
                                      scale=False,
                                      weighted=False))),
                              depth_rff=2, rff_config={"units": 4})
        inputs = Input(shape=(num_objects, num_features))
        output = attention_layer(inputs)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        model.compile(optimizer=SGD(learning_rate=1e-4), loss='mean_squared_error')

        weights_before = np.asarray(model.get_weights())

        model.fit(data, result, epochs=10)

        weights_after = np.asarray(model.get_weights())

        predictions = model.predict(data)

        print("Model weights before:")
        print(weights_before)
        print("Model weights after training:")
        print(weights_after)
        print("Model weights difference:")
        difference = np.abs(weights_after - weights_before)
        print(difference)

        assert not np.any(np.isnan(predictions))
        assert predictions.shape == data.shape


def test_set_transformer_d_c():
    random_state = np.random.RandomState(seed=42)
    # gen = DiscreteChoiceDatasetGenerator(dataset_type="simple_max", n_objects=5, n_train_instances=10,
    #                                      n_test_instances=5,
    #                                      random_state=random_state)
    gen = DiscreteChoiceDatasetGenerator(dataset_type="min_max", n_objects=5, n_train_instances=10,
                                         n_test_instances=5, n_features=5,
                                         random_state=random_state)

    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    print(x_test)
    print(y_test)

    layer_options = {"SAB": {"mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}
    transformer = SetTransformerDiscreteChoice(stacking_height=1, attention_layer_config=layer_options,
                                               num_layers_dense=3, num_units_dense=6)

    callbacks = []
    transformer.fit(x_train, y_train, epochs=5, verbose=2, callbacks=callbacks)

    prediction = transformer.predict(x_test)
    scores_test = transformer.predict_scores(x_test)
    scores_train = transformer.predict_scores(x_train)

    print("prediction")
    print(prediction)

    print("prediction scores")
    print(scores_test)

    print("true")
    print(y_test)

    print("scores for train")
    print(scores_train)


def test_set_transformer_choice():
    random_state = np.random.RandomState(seed=42)
    # gen = DiscreteChoiceDatasetGenerator(dataset_type="simple_max", n_objects=5, n_train_instances=10,
    #                              n_test_instances=5, threshold=.5,
    #                              random_state=random_state)
    gen = ChoiceDatasetGenerator(dataset_type="min_max", n_objects=5, n_train_instances=10,
                                 n_test_instances=5,
                                 random_state=random_state, n_features=2)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    print(x_train)
    print(y_train)

    layer_options = {"SAB": {"mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}
    transformer = SetTransformerChoice(stacking_height=1, attention_layer_config=layer_options,
                                       num_layers_dense=3, num_units_dense=6, seed=10)

    # callbacks = [AdvancedTensorBoard(log_gradient_norms=True, visualization_frequency=1, log_attention=True)]
    callbacks = []
    transformer.fit(x_train, y_train, epochs=5, verbose=2, callbacks=callbacks)

    prediction = transformer.predict(x_test)
    print(prediction)


def test_set_transformer_ranking():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="simple_max", n_objects=5, n_train_instances=100,
                                        n_test_instances=5,
                                        random_state=random_state)
    # gen = ObjectRankingDatasetGenerator(dataset_type="min_max", n_objects=5, n_train_instances=10,
    #                                      n_test_instances=5,
    #                                      random_state=random_state, n_features=2)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    layer_options = {"SAB": {"mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}
    transformer = SetTransformerObjectRanker(stacking_height=3, attention_layer_config=layer_options,
                                             num_layers_dense=2, num_units_dense=8, seed=10,
                                             optimizer=SGD(lr=1e-2, nesterov=True, momentum=0.9))

    # callbacks = [AdvancedTensorBoard(log_gradient_norms=True, visualization_frequency=1, log_attention=True)]
    callbacks = [WeightPrinterCallback(transformer)]
    transformer.fit(x_train, y_train, epochs=10, verbose=2, callbacks=callbacks)

    print(transformer.model.summary())

    prediction = transformer.predict(x_test)
    print("prediction")
    print(prediction)
    scores = transformer.predict_scores(x_test)
    print("scores")
    print(scores)
    print("true")
    print(y_test)


def test_set_transformer_ranking_callback():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="simple_max", n_objects=5, n_train_instances=20,
                                        n_test_instances=5,
                                        random_state=random_state)
    # gen = ObjectRankingDatasetGenerator(dataset_type="min_max", n_objects=5, n_train_instances=10,
    #                                      n_test_instances=5,
    #                                      random_state=random_state, n_features=2)

    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    print(np.sum(x_train), np.sum(y_train), np.sum(x_test), np.sum(y_test))

    layer_options = {"SAB": {"mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention_config": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}

    transformer = SetTransformerObjectRanker(stacking_height=1, attention_layer_config=layer_options,
                                             num_layers_dense=2, num_units_dense=8, seed=10,
                                             optimizer=SGD(lr=1e-2, nesterov=True, momentum=0.9), batch_size=10)

    callbacks = [AdvancedTensorBoard(log_gradient_norms=True,
                                     visualization_frequency=1,
                                     log_attention=True,
                                     save_space=True,
                                     log_lr=True,
                                     save_visualization_data=True,
                                     num_visualizations_per_epoch=1,
                                     histogram_freq=1,
                                     # batch_size=256,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=False,
                                     embeddings_freq=0,
                                     update_freq="epoch",
                                     log_dir="./tensorboard_logs/set_transformer/" + datetime.datetime.now().strftime(
                                         "%Y%m%d-%H%M%S")
                                     )]
    transformer.fit(x_train, y_train, epochs=10, verbose=2, callbacks=callbacks)

    print(transformer.model.summary())

    K.clear_session()
    prediction = transformer.predict(x_test)
    print("prediction")
    print(prediction)
    scores = transformer.predict_scores(x_test)
    print("scores")
    print(scores)
    print("true")
    print(y_test)


def test_fate_attention_ranking():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="simple_max", n_objects=5, n_train_instances=100,
                                        n_test_instances=5,
                                        random_state=random_state)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    attention = ScaledDotProductAttention(scale=False, weighted=False)
    attention_pooling = PMA(k=1, mab=MAB(
        multi_head=MultiHeadAttention(ScaledDotProductAttention(scale=False, weighted=False), num_heads=1)))
    learner = FATEObjectRanker(n_object_features=1, attention_function_preselection=attention,
                               n_hidden_joint_layers=3, n_hidden_joint_units=3, attention_pooling=attention_pooling)

    learner.fit(x_train, y_train, epochs=5, verbose=2)

    print(learner.model.summary())

    prediction = learner.predict(x_test)
    print("prediction")
    print(prediction)
    print("true")
    print(y_test)


def test_feta_attention_ranking_tsp():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="tsp", n_objects=5, n_train_instances=100,
                                        n_test_instances=5,
                                        random_state=random_state)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    attention = ScaledDotProductAttention(scale=False, weighted=False)
    callbacks = [AdvancedTensorBoard(
                            # data_visualization_func="tsp_2d",
                            # metric_for_visualization="TSPRelativeDifference_requiresX",
                            # metric_for_visualization_requires_x=True,
                            log_gradient_norms=False,
                            visualization_frequency=1,
                            log_attention=False,
                            save_space=True,
                            log_lr=False,
                            save_visualization_data=False,
                            num_visualizations_per_epoch=1,
                            histogram_freq=10,
                            write_graph=False,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            update_freq="epoch",
                            log_dir="./tensorboard_logs/feta_ranker_tsp/" + datetime.datetime.now().strftime(
                             "%Y%m%d-%H%M%S")
                            )]
    attention_pooling = {"PMA": {"k": 1, "mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention_config": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}
    learner = FETAObjectRanker(n_objects=5, n_hidden=2, n_units=8, add_zeroth_order_model=True,
                               max_number_of_objects=5, n_object_features=2, attention_function_preselection=None,
                               n_hidden_joint_layers=3, n_hidden_joint_units=3, attention_pooling=None)

    learner.fit(x_train, y_train, epochs=5, verbose=2, callbacks=callbacks)

    print(learner.model.summary())

    # K.clear_session()
    prediction = learner.predict(x_test)
    print("prediction")
    print(prediction)
    print("true")
    print(y_test)


def test_feta_attention_ranking_with_callback():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="medoid", n_objects=5, n_train_instances=100,
                                        n_test_instances=5,
                                        random_state=random_state)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    attention = ScaledDotProductAttention(scale=False, weighted=False)
    attention_pooling = {"PMA": {"k": 1, "mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention": {"ScaledDotProductAttention":
                                                                 {"weighted": False, "biased": False}}}}}}}}

    learner = FETAObjectRanker(n_objects=5, n_hidden=2, n_units=8, add_zeroth_order_model=False,
                               max_number_of_objects=5, n_object_features=100,
                               attention_function_preselection=attention,
                               n_hidden_joint_layers=3, n_hidden_joint_units=3, attention_pooling=None)

    callbacks = [AdvancedTensorBoard(log_gradient_norms=False,
                                     visualization_frequency=1,
                                     log_attention=True,
                                     save_space=True,
                                     log_lr=True,
                                     save_visualization_data=False,
                                     histogram_freq=2,
                                     # batch_size=256,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=False,
                                     embeddings_freq=0,
                                     update_freq="epoch",
                                     log_dir="./tensorboard_logs/feta_attention/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                     )]
    learner.fit(x_train, y_train, epochs=5, verbose=2, callbacks=callbacks)

    print(learner.model.summary())

    prediction = learner.predict(x_test)
    print("prediction")
    print(prediction)
    print("true")
    print(y_test)


def test_multihead_efficient():
    matrix = np.asarray([[[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]]])

    batch_size = 1
    feature_dimension_previous = 4
    feature_dimension_after = 2
    num_objects = 4
    num_heads = 2

    input_matrix = K.placeholder(matrix.shape)
    combine_batch = split_features_to_batch(batch_size, feature_dimension_after, num_heads, num_objects, matrix)

    combine_features = combine_batch_to_features(batch_size, feature_dimension_after, feature_dimension_previous,
                                                 num_heads, num_objects, combine_batch)

    with tf.Session() as sess:
        results = sess.run(fetches=[combine_batch, combine_features],
                           feed_dict={input_matrix: matrix})

        result = results[1]
        assert_almost_equal(actual=result, desired=matrix)


def test_layer_normalization():
    matrix = np.asarray([[[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]]])
    print(matrix.shape)
    input = Input(shape=(4, 4))
    output = LayerNormalization()(input)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(matrix, matrix)

    prediction = model.predict(matrix)
    print(prediction)


def test_model_subclassing():
    matrix = np.asarray([[[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]]])

    class TestModel(Model):
        def __init__(self, input_shape=None):
            super(TestModel, self).__init__()

            if input_shape is not None:
                self.dense_1 = Dense(units=2, input_shape=input_shape[0])
                self.dense_2 = Dense(units=2, input_shape=input_shape[1])
                self.dense_3 = Dense(units=4)

                self.build_other_layers()
            else:
                self.dense_1 = Dense(units=2)
                self.dense_2 = Dense(units=2)
                self.dense_3 = Dense(units=4)

        def build_other_layers(self):
            self.concat = Concatenate()

        # note: build is NOT called automatically when used as model, only when used as layer!!
        def build(self, input_shape):
            self.build_other_layers()

            super(TestModel, self).build(input_shape)

        def call(self, inputs, mask=None):
            return self.dense_3(self.concat([self.dense_1(inputs[0]), self.dense_2(inputs[1])]))

    model = TestModel(input_shape=[(4, 4), (4, 4)])
    model.compile(optimizer="SGD", loss="mean_squared_error")

    model.fit([matrix, matrix], matrix)

    print(model.summary())
    print(model.built)

    print(model.predict([matrix, matrix]))
