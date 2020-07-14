"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  experiment_cv.py --c_index=<id> --job_id=<job_id> --config_file_name=<config_file_name> --table_name=<table_name>
  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --c_index=<c_index>                     Index given by the cluster to specify which job
                                        is to be executed [default: 0]
  --job_id=<job_id>                       JobId which needs to be re-evaluated
                                        is to be executed [default: 0]
  --config_file_name=<config_file_name>   File name of the database config
  --table_name=<table_name>             table from which to get configs
"""
import inspect
import logging

import os

import pickle as pk
import sys
from datetime import datetime

import h5py
import numpy as np

from csrank.dataset_reader.objectranking.object_ranking_data_generator import load_tsp_dataset
from csrank.losses import knapsack_loss_value_wrapper_wrapper, knapsack_loss_weight_wrapper_wrapper

np.random.seed(0)
from docopt import docopt
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf

from csrank.callbacks import AdvancedTensorBoard
from csrank.experiments import *
from csrank.experiments.dbconnection_modified import ModifiedDBConnector
from csrank.experiments.util import learners, create_callbacks
from csrank.metrics import make_ndcg_at_k_loss, tsp_loss_relative_wrapper, knapsack_weight_wrapper_wrapper, \
    knapsack_wrapper_wrapper, knapsack_value_wrapper
from csrank.tensorflow_util import configure_numpy_keras, get_loss_statistics, get_mean_loss
from csrank.tuning import ParameterOptimizer
from csrank.util import create_dir_recursively, duration_till_now, seconds_to_time, \
    print_dictionary, get_duration_seconds, setup_logging, rename_file_if_exist

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
OPTIMIZER_FOLDER = 'optimizers'
PREDICTIONS_FOLDER = 'predictions'
MODEL_FOLDER = 'models'
ERROR_OUTPUT_STRING = 'Out of sample error %s : %0.4f'
SAVED_DATA_FOLDER = 'saved_data'

theano_learners = ["mixed_logit_model", "generalized_extreme_value", "nested_logit_model",
                   "multinomial_logit_model"]


def exit_orderly_in_case_of_error(error_message, db_connector, job_id):
    error_message = str(error_message)
    print("Error running evaluation for job {}\n".format(job_id), error_message)
    db_connector.append_error_string_in_running_job(job_id=job_id, error_message=error_message)


def replace_kernel_regularizer_params(learner_params):
    for key in list(learner_params.keys()):
        if key == 'kernel_regularizer_params':
            learner_params['kernel_regularizer'] = regularizers[learner_params['kernel_regularizer']](
                **learner_params[key])
            del learner_params['kernel_regularizer_params']
        elif isinstance(learner_params[key], dict):
            replace_kernel_regularizer_params(learner_params[key])

    return learner_params


def do_experiment():
    start = datetime.now()

    print(sys.argv)
    print("TensorFlow built with CUDA-support", tf.compat.v1.test.is_built_with_cuda())
    print("GPU is available", tf.compat.v1.test.is_gpu_available())

    # extract arguments
    arguments = docopt(__doc__)
    cluster_id = int(arguments["--c_index"])
    job_id = int(arguments["--job_id"])
    config_file_name = arguments["--config_file_name"]
    table_jobs = arguments["--table_name"]

    # configure postgres database connector
    config_file_path = os.path.join(DIR_PATH, 'database_configs', config_file_name)
    db_connector = ModifiedDBConnector(config_file_path=config_file_path, table_jobs=table_jobs)

    # get job description
    job_description = db_connector.get_job_for_id(job_id=job_id, cluster_id=cluster_id)
    if job_description is not None:
        try:
            # # # GENERAL SETUP # # #

            # parse job parameters
            dataset_name = job_description["dataset"]
            dataset_params = job_description["dataset_params"]
            fold_id = int(job_description["fold_id"])
            n_inner_folds = int(job_description["n_inner_folds"])
            learning_problem = job_description["learning_problem"]

            seed = int(job_description["seed"])

            learner_name = job_description["learner_name"]
            learner_params = job_description["learner_params"]
            learner_fit_params = job_description["learner_fit_params"]

            use_hp = bool(job_description["use_hp"])
            hp_iterations = int(job_description["hp_iterations"])
            hp_ranges = job_description["hp_ranges"]
            hp_fit_params = job_description["hp_fit_params"]

            duration = job_description["duration"]
            time_out_eval = get_duration_seconds(job_description["time_out_eval"])
            results_table_name = job_description["results_table_name"]
            hash_value = job_description["hash_value"]

            # init random state
            random_state = np.random.RandomState(seed=seed + fold_id)

            # set up optimizer path
            optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(hash_value))
            create_dir_recursively(optimizer_path, True)

            # set up log path
            log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "{}.log".format(hash_value))
            log_path = rename_file_if_exist(log_path)
            create_dir_recursively(log_path, True)

            # set up path for pickling module
            pickle_path = os.path.join(DIR_PATH, MODEL_FOLDER, "{}.pickle".format(hash_value))
            create_dir_recursively(pickle_path, True)

            # set up logger
            setup_logging(log_path=log_path)
            configure_numpy_keras(seed=seed, log_device_placement_if_is_gpu_available=False)

            logger = logging.getLogger('Experiment')
            logger.info("DB config filePath {}".format(config_file_path))
            logger.info("Arguments {}".format(arguments))
            logger.info("Job Description {}".format(print_dictionary(job_description)))

            if learner_name not in theano_learners:
                if "callbacks" in learner_fit_params.keys():
                    # if there is callback, set file name to include hash for identification
                    if "AdvancedTensorBoard" in learner_fit_params["callbacks"].keys():
                        # replace path
                        log_dir = learner_fit_params["callbacks"]["AdvancedTensorBoard"]["log_dir"]
                        learner_fit_params["callbacks"]["AdvancedTensorBoard"]["log_dir"] = \
                            os.path.join(log_dir, table_jobs[5:], dataset_params["dataset_type"], learner_name,
                                         hash_value)

            # # # DATA SETUP # # #

            # get data
            if dataset_params['dataset_type'] == 'tsp':
                data_foldr = os.path.join(DIR_PATH, '..', SAVED_DATA_FOLDER)
                x_train, y_train, x_test, y_test = load_tsp_dataset(
                    n_train_instances=dataset_params['n_train_instances'],
                    n_test_instances=dataset_params['n_test_instances'],
                    n_objects=dataset_params['n_objects'],
                    seed=seed,
                    path=data_foldr,
                    sorted=dataset_params['sorted'])
                standardizer = None
            else:
                dataset_params['random_state'] = random_state
                dataset_params['fold_id'] = fold_id
                dataset_reader = get_dataset_reader(dataset_name, dataset_params)
                x_train, y_train, x_test, y_test = dataset_reader.get_single_train_test_split()
                standardizer = dataset_reader.standardizer

                del dataset_reader

            # log data contents, get num_objects, delete internal reader info
            n_objects = log_test_train_data(x_train, x_test, logger)
            inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
            if learner_name in [MNL, PCL, NLM, GEV]:
                learner_fit_params['random_seed'] = seed + fold_id

            # # # HYPERPARAMETEROPTIMIZATION + TRAINING # # #

            # log and compute timing for training
            time_taken = duration_till_now(start)
            logger.info("Time Taken till now: {}  milliseconds".format(seconds_to_time(time_taken)))
            logger.info("Time spared for the out of sample evaluation : {} ".format(seconds_to_time(time_out_eval)))
            duration = get_duration_seconds(duration)
            total_duration = duration - time_taken - time_out_eval

            # set learner parameters
            learner_params['n_objects'], learner_params['n_object_features'] = x_train.shape[1:]
            learner_params["random_state"] = random_state
            if learner_name not in theano_learners:
                if "loss_function" in learner_params.keys():
                    loss_func_name = learner_params["loss_function"]
                    learner_params["loss_function"] = util.losses[loss_func_name]
                    if loss_func_name == 'knapsack_loss_wrapper_wrapper':
                        learner_params['loss_function'] = learner_params['loss_function'](dataset_params['capacity'],
                                                                                          standardizer.mean_,
                                                                                          standardizer.var_)
                logger.info("learner params {}".format(print_dictionary(learner_params)))
                if "optimizer" in learner_params.keys():
                    learner_params["optimizer"] = \
                        optimizers[learner_params["optimizer"]](**learner_params["optimizer_params"])
                    del learner_params["optimizer_params"]
                if "metrics" not in learner_params.keys():
                    learner_params["metrics"] = []
                replace_kernel_regularizer_params(learner_params)
            if dataset_params["dataset_type"] == "knapsack" and learner_name in ["feta_choice", "fate_choice",
                                                                                 "set_transformer_choice"]:
                learner_params["metrics_requiring_x"] = [knapsack_loss_value_wrapper_wrapper(standardizer.mean_,
                                                                                             standardizer.var_),
                                                         knapsack_loss_weight_wrapper_wrapper
                                                         (dataset_params['capacity'],
                                                          standardizer.mean_, standardizer.var_)]

            time_start_train = datetime.now()
            db_connector.log_start_training(job_id, time_start_train)

            if use_hp:
                # set hyperparameter optimizer parameters
                hp_params = create_optimizer_parameters_no_hash_file(learner_fit_params, hp_ranges, learner_params,
                                                                     learner_name)
                hp_params['optimizer_path'] = optimizer_path
                hp_params['random_state'] = random_state
                hp_params['learning_problem'] = learning_problem
                hp_params['validation_loss'] = lp_metric_dict[learning_problem].get(hp_params['validation_loss'], None)
                logger.info("optimizer params {}".format(print_dictionary(hp_params)))

                # set hyperparameter optimizer fit params
                hp_fit_params['n_iter'] = hp_iterations
                hp_fit_params['total_duration'] = total_duration
                hp_fit_params['cv_iter'] = inner_cv

                # start hyperparameter optimizer - training
                optimizer_model = ParameterOptimizer(**hp_params)
                optimizer_model.fit(x_train, y_train, **hp_fit_params)
                validation_loss = optimizer_model.validation_loss
                learner = optimizer_model.model

                # upload validation_loss
                db_connector.insert_validation_loss(validation_loss, job_id)
            else:
                # just train
                create_callbacks(learner_fit_params, x=x_train, y=y_train)

                learner_func = learners[learner_name]
                learner = learner_func(**learner_params)
                learner.fit(x_train, y_train, **learner_fit_params)

            # timing
            time_finished_train = datetime.now()
            time_finished_vis = datetime.now()
            if "callbacks" in learner_fit_params.keys():
                for callback in learner_fit_params["callbacks"]:
                    if isinstance(callback, AdvancedTensorBoard):
                        time_finished_train = callback.train_end_time

            db_connector.set_end_time(job_id, time_finished_vis, time_finished_train)

            # # # SAVING MODEL # # #

            # TODO save model

            # # # PREDICTION # # #

            get_results_and_upload('test', x_test, y_test, db_connector, hash_value, job_id, learner
                                   , learning_problem, logger, n_objects, results_table_name, dataset_params,
                                   standardizer)

            get_results_and_upload('train', x_train, y_train, db_connector, hash_value, job_id, learner,
                                   learning_problem, logger, n_objects, results_table_name, dataset_params,
                                   standardizer)

            db_connector.finish_job(job_id=job_id, cluster_id=cluster_id)

        # # # ERROR-HANDLING # # #

        # in case of exception, log exception in database and logger
        except Exception as e:
            if hasattr(e, 'message'):
                message = e.message
            else:
                message = e

            exit_orderly_in_case_of_error(error_message=message, db_connector=db_connector, job_id=job_id)
            # raise error again so the whole error is logged
            raise e

        except (KeyboardInterrupt, GeneratorExit, SystemExit) as e:
            exit_orderly_in_case_of_error(error_message=sys.exc_info()[0].__name__, db_connector=db_connector
                                          , job_id=job_id)
            # raise error again so the whole error is logged
            raise e


def get_results_and_upload(case, data_x, data_y, db_connector, hash_value, job_id, learner, learning_problem, logger,
                           n_objects, results_table_name, dataset_params, standardizer):
    # set batch size
    if isinstance(data_x, dict):
        batch_size = 1000
    else:
        size = sys.getsizeof(data_x)
        batch_size = data_x.shape[0]
        logger.info("Test dataset size {}".format(size))

    # do the actual predictions
    predicted_scores, y_pred = get_scores(learner, batch_size, data_x, data_y, logger, standardizer)

    # # # WRITING BACK RESULTS # # #

    # write predicted scores to file
    if isinstance(predicted_scores, dict):
        pred_file = os.path.join(DIR_PATH, PREDICTIONS_FOLDER, "{}_{}.pickle".format(hash_value, case))
        create_dir_recursively(pred_file, True)
        f = open(pred_file, "wb")
        pk.dump(predicted_scores, f)
        f.close()
    else:
        pred_file = os.path.join(DIR_PATH, PREDICTIONS_FOLDER, "{}_{}.h5".format(hash_value, case))
        create_dir_recursively(pred_file, True)
        f = h5py.File(pred_file, 'w')
        f.create_dataset('scores', data=predicted_scores)
        f.create_dataset('inputs', data=data_x)
        f.create_dataset('outputs', data=data_y)
        f.close()
    logger.info("Saved predictions at: {}".format(pred_file))

    # insert results into database
    results = {'job_id': str(job_id), 'train_test': "\'" + case + "\'"}
    metric_dict = lp_metric_dict[learning_problem]
    if dataset_params['dataset_type'] == 'knapsack':
        metric_dict = {"Knapsack_requiresX": knapsack_wrapper_wrapper,
                       "Knapsack_Weight_requiresX": knapsack_weight_wrapper_wrapper,
                       "Knapsack_Value_requiresX": knapsack_value_wrapper, **metric_dict}
    for name, evaluation_metric in metric_dict.items():
        predictions = predicted_scores

        # set predictions accordingly if metric works on labels instead of scores
        if evaluation_metric in metrics_on_predictions:
            logger.info("Metric on predictions")
            predictions = y_pred

        # prepare knapsack
        if "Knapsack" in name and name != "Knapsack_Value_requiresX":
            evaluation_metric = evaluation_metric(dataset_params["capacity"])

        # prepare ndcg
        if "NDCG" in name:
            evaluation_metric = make_ndcg_at_k_loss(k=n_objects)
            predictions = y_pred

        # compute loss
        if name not in ["Informedness"]:
            metric_loss_min, metric_loss_max, metric_loss_mean, metric_loss_std = \
                get_loss_statistics(name, evaluation_metric, data_y, predictions, data_x, standardizer)
        else:
            metric_loss_mean = get_mean_loss(evaluation_metric, data_y, y_pred)
            metric_loss_min = metric_loss_max = metric_loss_mean
            metric_loss_std = 0

        logger.info(ERROR_OUTPUT_STRING % (name, metric_loss_mean))

        write_loss_to_results(name + "_min", results, metric_loss_min)
        write_loss_to_results(name + "_max", results, metric_loss_max)
        write_loss_to_results(name + "_mean", results, metric_loss_mean)
        write_loss_to_results(name + "_std", results, metric_loss_std)
    # upload results
    db_connector.insert_results(results=results, results_table_name=results_table_name)


def write_loss_to_results(name, results, value):
    # write loss into results
    if np.isnan(value):
        results[name] = "\'Infinity\'"
    else:
        results[name] = "{0:.4f}".format(value)


if __name__ == "__main__":
    do_experiment()
