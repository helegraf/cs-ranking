"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  experiment_cv.py --n_train_instances=<n_train_instances=> --n_test_instances=<n_test_instances> --n_objects=<n_objects> --seed=<seed>
  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                                     Show this screen.
  --n_train_instances==<n_train_instances=>     Number of training instances for dataset.
  --n_test_instances=<n_test_instances>         Number of testing instances for dataset.
  --n_objects=<n_objects>                       Number of objects for dataset.
  --seed=<seed>                                 Seed for dataset.
"""
import inspect
import logging

import os

import pickle as pk
import sys
from datetime import datetime

import h5py
import numpy as np
from docopt import docopt
from numpy.testing import assert_almost_equal
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf

from csrank.dataset_reader.objectranking.object_ranking_data_generator import save_tsp_dataset, \
    ObjectRankingDatasetGenerator, load_tsp_dataset
from csrank.experiments import *
from csrank.experiments.dbconnection_modified import ModifiedDBConnector
from csrank.experiments.util import learners, create_callbacks, all_metrics
from csrank.metrics import make_ndcg_at_k_loss
from csrank.tensorflow_util import configure_numpy_keras, get_loss_statistics
from csrank.tuning import ParameterOptimizer
from csrank.util import create_dir_recursively, duration_till_now, seconds_to_time, \
    print_dictionary, get_duration_seconds, setup_logging, rename_file_if_exist

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVED_DATA_FOLDER = 'saved_data'


def do_experiment():
    # extract arguments
    arguments = docopt(__doc__)
    n_train_instances = int(arguments["--n_train_instances"])
    n_test_instances = int(arguments["--n_test_instances"])
    n_objects = int(arguments["--n_objects"])
    seed = int(arguments["--seed"])
    sorted = True

    gen = ObjectRankingDatasetGenerator(dataset_type='tsp', n_train_instances=n_train_instances,
                                        n_test_instances=n_test_instances, n_objects=n_objects,
                                        random_state=np.random.RandomState(seed=42), sorted=sorted)
    x_train, y_train, x_test, y_test = gen.get_single_train_test_split()

    save_tsp_dataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     n_train_instances=n_train_instances, n_test_instances=n_test_instances, n_objects=n_objects,
                     seed=seed, path=SAVED_DATA_FOLDER, sorted=sorted)

    x_train_2, y_train_2, x_test_2, y_test_2 = load_tsp_dataset(n_train_instances=n_train_instances,
                                                                n_test_instances=n_test_instances, n_objects=n_objects,
                                                                seed=seed, path=SAVED_DATA_FOLDER, sorted=sorted)

    print(x_train)
    print("--")
    print(y_train)

    assert_almost_equal(actual=x_train_2, desired=x_train, decimal=7)
    assert_almost_equal(actual=x_test_2, desired=x_test, decimal=7)
    assert_almost_equal(actual=y_train_2, desired=y_train, decimal=7)
    assert_almost_equal(actual=y_test_2, desired=y_test, decimal=7)


if __name__ == "__main__":
    do_experiment()
