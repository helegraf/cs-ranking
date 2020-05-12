from datetime import datetime

import numpy as np
from keras import Input, Model
from keras.layers import TimeDistributed, Dense, Reshape
import keras.backend as K

from csrank import FETAObjectRanker, ObjectRankingDatasetGenerator
from csrank.attention.set_transformer.modules import ScaledDotProductAttention
from csrank.callbacks import AdvancedTensorBoard
from csrank.visualization.weights import visualize_attention_scores


def metric_with_x_wrapper(x):
    def metric_with_x(y, z):
        return K.sum(y - z) + K.sum(x)

    return metric_with_x


def test_feta_attention_ranking_tsp():
    random_state = np.random.RandomState(seed=42)
    gen = ObjectRankingDatasetGenerator(dataset_type="tsp", n_objects=5, n_train_instances=100,
                                        n_test_instances=5,
                                        random_state=random_state)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    attention = {"ScaledDotProductAttention":{"weighted": False, "biased": False}}
    callbacks = [AdvancedTensorBoard(
        inputs=np.asarray([x_train[0]]),
        targets=np.asarray([y_train[0]]),
        prediction_visualization="tsp_2d",
        metric_for_visualization="TSPRelativeDifference_requiresX",
        metric_for_visualization_requires_x=True,
        log_gradient_norms="all",
        log_attention=True,
        save_space=True,
        log_lr=True,
        histogram_freq=1,
        write_graph=False,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        update_freq="epoch",
        log_dir="./tensorboard_logs/feta_ranker_tsp/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )]
    attention_pooling = {"PMA": {"k": 1, "mab": {"MAB": {"multi_head": {
        "MultiHeadAttention": {"num_heads": 1, "attention_config": {"ScaledDotProductAttention":
                                                                        {"weighted": False, "biased": False}}}}}}}}
    learner = FETAObjectRanker(n_objects=5, n_hidden=2, n_units=8, add_zeroth_order_model=True,
                               max_number_of_objects=5, n_object_features=2, attention_function_preselection_config=attention,
                               num_attention_function_preselection_layers=1,
                               n_hidden_joint_layers=3, n_hidden_joint_units=3, attention_pooling=None)

    learner.fit(x_train, y_train, epochs=5, verbose=2, callbacks=callbacks)

    print(learner.model.summary())

    # K.clear_session()
    prediction = learner.predict(x_test)
    print("prediction")
    print(prediction)
    print("true")
    print(y_test)


def test_advanced_tensorboard():
    # configure case
    seed = 42

    n_instances = 10
    n_objects = 5
    n_features = 3

    # create data
    random = np.random.RandomState(seed=seed)
    x_data = random.uniform(low=-1, high=1, size=(n_instances, n_objects, n_features))
    y_data = np.reshape(np.asarray([np.sum(np.square(instance)) for instance in x_data]), newshape=(n_instances, 1))

    print("\n")
    print("x_data")
    print(x_data)
    print("y_data")
    print(y_data)

    # create model
    inputs = Input(shape=(n_objects, n_features))
    attention_1_layer = ScaledDotProductAttention(weighted=False, scale=False)
    attention_1 = attention_1_layer(inputs)
    attention_2 = ScaledDotProductAttention(weighted=False, scale=False)(attention_1)
    output = TimeDistributed(Dense(units=n_features))(attention_2)
    output = TimeDistributed(Dense(units=n_features))(output)
    output = Reshape(target_shape=(n_objects * n_features,))(output)
    output = Dense(units=1)(output)

    model = Model(inputs=inputs, outputs=output)

    lol = metric_with_x_wrapper(inputs)

    # compile
    model.compile(optimizer="SGD", loss="mean_squared_error", metrics=[lol])
    print(model.summary())

    attention_outputs = attention_1_layer.get_attention_layer_inputs_outputs()[0]

    visualizations = [(attention_outputs['query'], attention_outputs['key'], attention_outputs['scores'],
                       visualize_attention_scores)]

    callbacks = [AdvancedTensorBoard(inputs=np.asarray([x_data[0]]),
                                     targets=np.asarray([y_data[0]]),
                                     log_lr=True,
                                     log_gradient_norms="global",
                                     visualizations=visualizations,
                                     histogram_freq=1,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=False,
                                     embeddings_freq=0,
                                     update_freq="epoch",
                                     log_dir="./tensorboard_logs/set_transformer/" + datetime.now().strftime(
                                                  "%Y%m%d-%H%M%S")
                                     )]
    callbacks[0].set_additional_intermediate_tensors_to_evaluate([attention_1, attention_2])

    # fit
    model.fit(x=x_data, y=y_data, epochs=10, verbose=2, callbacks=callbacks, validation_split=.1)
    print(model.inputs)
    print(model.outputs)
    print(model.targets)

    # predict
    model.predict(x=x_data)