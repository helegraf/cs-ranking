"""Experiment runner for one dataset.

Usage:
  generalization_experiment.py (--n_objects=<n_objects> --dataset_type=<dataset_type>)

  generalization_experiment.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --n_objects=<n_objects>         Number of Objects of the random generator [default: 5]
  --dataset_type=<dataset_type>   Synthetic Dataset type
"""
import inspect
import logging
import os

import numpy as np
import pandas as pd
from docopt import docopt

from csrank import ObjectRankingDatasetGenerator, FETAObjectRanker, RankNet, ListNet, RankSVM, FATEObjectRanker, \
    ExpectedRankRegression
from csrank.callbacks import DebugOutput, LRScheduler
from csrank.metrics import zero_one_rank_loss_for_scores
from csrank.tensorflow_util import get_tensor_value, configure_numpy_keras
from csrank.util import rename_file_if_exist, setup_logging

MODEL = "aModel"

ERROR_OUTPUT_STRING = 'Out of sample error {} : {} for n_objects {}'

logger = None


def generate_dataset(dataset_type, n_objects=5, random_state=42):
    parameters = {"n_features": 2, "n_objects": n_objects, "n_train_instances": 10000, "n_test_instances": 100000,
                  "dataset_type": dataset_type, "random_state": random_state}
    generator = ObjectRankingDatasetGenerator(**parameters)
    return generator.get_single_train_test_split()


def get_evaluation_result(seed, model, epochs, dataset_type, x_train, y_train, x_test=None, y_test=None,
                          test_inst_sizes=None, callbacks=[DebugOutput()]):

    # fit model to training data
    model.fit(x_train, y_train, callbacks=callbacks, epochs=epochs)

    # test model on different testing datasets
    eval_results = {}

    # this means it may have seen parts of the testing data during training if you only do it with the same size
    # therefore we use the actual split data that was generated before if test_inst_sizes is not given
    if test_inst_sizes is not None:
        for n_objects in test_inst_sizes:
            _, _, x_test, y_test = generate_dataset(dataset_type, n_objects=n_objects,
                                                    random_state=seed + n_objects * 5)
            single_test_eval(eval_results, model, n_objects, x_test, y_test)
    else:
        single_test_eval(eval_results, model, y_test.shape[0, 0], x_test, y_test)

    return eval_results


def single_test_eval(eval_results, model, n_objects, x_test, y_test):
    y_pred_scores = model.predict_scores(x_test, batch_size=x_test.shape[0])
    metric_loss = get_tensor_value(zero_one_rank_loss_for_scores(y_test, y_pred_scores))
    logger.info(ERROR_OUTPUT_STRING.format("zero_one_rank_loss", str(np.mean(metric_loss)), n_objects))
    eval_results[n_objects] = metric_loss


def quick_experiment(n_objects, test_inst_sizes, dataset_type, epochs=1500, keras_seed_range=2**32, numpy_seed=42):
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(dirname, "logs", "generalizing_mean_{}_{}.log".format(dataset_type, n_objects))
    df_path = os.path.join(dirname, "logs", "generalizing_mean_{}_{}.csv".format(dataset_type, n_objects))
    log_path = rename_file_if_exist(log_path)
    df_path = rename_file_if_exist(df_path)

    random_state = np.random.RandomState(numpy_seed)
    seed = random_state.randint(keras_seed_range)

    rows_list = []

    setup_logging(log_path=log_path)
    configure_numpy_keras(seed=seed)

    global logger
    logger = logging.getLogger('Generalization Experiment')
    x_train, y_train, x_test, y_test = generate_dataset(dataset_type, n_objects=n_objects, random_state=seed)
    n_instances, n_train_objects, n_features = x_train.shape

    logger.info("############################# With FATERanker ##############################")  # 53
    point = {'n_hidden_set_units': 86, 'n_hidden_set_layers': 4, 'n_hidden_joint_units': 42, 'n_hidden_joint_layers': 5,
             'reg_strength': 4.815764337557941e-07, 'learning_rate': 0.00287838171819095, 'batch_size': 334}
    fate = FATEObjectRanker(n_objects=n_train_objects, n_object_features=n_features)
    fate.set_tunable_parameters(**point)
    params = {'epochs_drop': 152, 'drop': 0.08534838347396632}
    lr = LRScheduler(**params)
    result = get_evaluation_result(seed, fate, epochs, dataset_type, x_train, y_train, test_inst_sizes,
                                   callbacks=[lr, DebugOutput()])
    result[MODEL] = "FATERanker"
    rows_list.append(result)

    logger.info("############################# With FETARanker ##############################")  # 22
    point = {'n_hidden': 6, 'n_units': 21, 'learning_rate': 0.002877977772158241, 'reg_strength': 0.006856379975844599,
             'batch_size': 688}
    feta = FETAObjectRanker(n_objects=n_train_objects, n_object_features=n_features)
    feta.set_tunable_parameters(**point)
    params = {'epochs_drop': 199, 'drop': 0.02498577223683132}
    lr = LRScheduler(**params)
    result = get_evaluation_result(seed, feta, epochs, dataset_type, x_train, y_train, test_inst_sizes,
                                   callbacks=[lr, DebugOutput()])
    result[MODEL] = "FETARanker"
    rows_list.append(result)

    logger.info("############################# With RankNet ##############################")  # 24
    point = {'n_hidden': 2, 'n_units': 38, 'learning_rate': 0.0004869994896776387, 'reg_strength': 0.1,
             'batch_size': 410}
    rank_net = RankNet(n_objects=n_train_objects, n_object_features=n_features)
    rank_net.set_tunable_parameters(**point)
    params = {'epochs_drop': 250, 'drop': 0.01}
    lr = LRScheduler(**params)
    result = get_evaluation_result(seed, rank_net, epochs, dataset_type, x_train, y_train, test_inst_sizes,
                                   callbacks=[lr, DebugOutput()])
    result[MODEL] = "RankNet"
    rows_list.append(result)

    logger.info("############################# With ListNet ##############################")  # 53
    point = {'n_hidden': 13, 'n_units': 25, 'learning_rate': 0.0044822356571576245,
             'reg_strength': 0.00017570468364928768, 'batch_size': 805}
    list_net = ListNet(n_top=3, n_objects=n_train_objects, n_object_features=n_features)
    list_net.set_tunable_parameters(**point)
    params = {'epochs_drop': 166, 'drop': 0.06140848438671757}
    lr = LRScheduler(**params)
    result = get_evaluation_result(seed, list_net, epochs, dataset_type, x_train, y_train, test_inst_sizes,
                                   callbacks=[lr, DebugOutput()])
    result[MODEL] = "ListNet"
    rows_list.append(result)

    logger.info("############################# With ExpectedRankRegression ##############################")  # 3
    point = {'tol': 0.0006463744551066736, 'alpha': 7.0942908812201645e-06, 'l1_ratio': 0.048064055105687134}
    err = ExpectedRankRegression(n_objects=n_train_objects, n_object_features=n_features)
    err.set_tunable_parameters(**point)
    result = get_evaluation_result(seed, err, epochs, dataset_type, x_train, y_train, test_inst_sizes)
    result[MODEL] = "ERR"
    rows_list.append(result)

    logger.info("############################# With RankSVM ##############################")  # 34
    point = {'tol': 0.49644832732622124, 'C': 12}
    svm = RankSVM(n_objects=n_train_objects, n_object_features=n_features)
    svm.set_tunable_parameters(**point)
    result = get_evaluation_result(seed, svm, epochs, dataset_type, x_train, y_train, test_inst_sizes)
    result[MODEL] = "RankSVM"
    rows_list.append(result)

    df = pd.DataFrame(rows_list)
    df = df.set_index(MODEL).T
    df.to_csv(df_path)
    pd.read_csv(df_path)
