"""Results visualizer that egts results from a database

Usage: results_visualizer.py --config_file_name=<config_file_name> --job_table_name=<job_table_name>
--result_table_name=<result_table_name> results_visualizer.py (-h | --help)

Options:
  -h --help                                 Show this screen.
  --config_file_name=<config_file_name>     File name of the database config
  --job_table_name=<job_table_name>         table in which the jobs are
  --result_table_name=<result_table_name>   table in which the results are
"""
import inspect
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from docopt import docopt

from csrank import get_dataset_reader
from csrank.experiments.dbconnection_modified import ModifiedDBConnector
from csrank.visualization.predictions import add_tsp_figure_to_plot

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def visualize_single_tsp_result():
    hash = "e85916a6da8dc4efsrfsefsfsfe"
    dataset_name = "synthetic_or"
    dataset_params = {"dataset_type": "tsp", "n_objects": 10, "n_train_instances": 1024, "n_test_instances": 64}
    fold_id = 1
    num_visualize = 3
    seed = 1234

    # get data
    random_state = np.random.RandomState(seed=seed + fold_id)
    dataset_params['random_state'] = random_state
    dataset_params['fold_id'] = fold_id
    dataset_reader = get_dataset_reader(dataset_name, dataset_params)
    x_train, y_train, x_test, y_test = dataset_reader.get_single_train_test_split()

    # read predictions
    prediction_file_test = "predictions/" + hash + "_test.h5"
    f_test = h5py.File(prediction_file_test, 'r')
    test_pred = np.array(f_test.get('scores'))

    prediction_file_train = "predictions/" + hash + "_train.h5"
    f_train = h5py.File(prediction_file_train, 'r')
    train_pred = np.array(f_train.get('scores'))

    # create figure
    fig2 = plt.figure(constrained_layout=True, figsize=(16, 8))
    spec2 = gridspec.GridSpec(ncols=num_visualize, nrows=2, figure=fig2)

    plt.title("test")
    for i in range(num_visualize):
        add_tsp_figure_to_plot \
            (figure=fig2, gridspec_location=spec2[0, i], x=x_test[i], true=y_test[i], prediction=test_pred[i])

    plt.title(" ")
    plt.title(" ")

    for i in range(num_visualize):
        add_tsp_figure_to_plot \
            (figure=fig2, gridspec_location=spec2[1, i], x=x_train[i], true=y_train[i], prediction=train_pred[i])
        plt.title("train")

    plt.show()


def visualize_run_results(dbconnection, results_table_name):
    # jobs
    jobs = [8, 9, 10]
    # metrics
    metric = "tsprelativedifference_requiresx"
    metric_name = "TSP relative difference"

    headers, results = dbconnection.get_results_for_job(jobs, results_table_name)

    # get indices
    index_metric = 0
    index_train_test = 0
    for col in range(len(headers)):
        column_name = headers[col]
        if column_name.startswith(metric) and column_name.endswith("mean"):
            index_metric = col

        if column_name == "train_test":
            index_train_test = col

    train_results = []
    test_results = []
    for result in results:
        if result[index_train_test] == "train":
            train_results.append(result)
        else:
            test_results.append(result)

    train_means = [result[index_metric] for result in train_results]
    test_means = [result[index_metric] for result in test_results]

    # do the visualization
    labels = [str(job_id) for job_id in jobs]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    print(train_means)
    print(test_means)
    rects1 = ax.bar(x - width / 2, train_means, width, label='Train')
    rects2 = ax.bar(x + width / 2, test_means, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc
    ax.set_ylim(top=max(max(train_means), max(test_means)) * 1.15)
    ax.set_title('Scores by train and test for metric ' + metric_name)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_yticklabels([])
    ax.set_xticklabels(labels)
    legend = ax.legend(loc='lower right', facecolor='white', framealpha=1)
    frame = legend.get_frame()
    frame.set_linewidth(0)
    # frame.set_color('white')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


arguments = docopt(__doc__)
config_file_name = arguments["--config_file_name"]
table_jobs = arguments["--job_table_name"]
table_results = arguments["--result_table_name"]
config_file_path = os.path.join(DIR_PATH, 'database_configs', config_file_name)
db_connector = ModifiedDBConnector(config_file_path=config_file_path, table_jobs=table_jobs)

visualize_run_results(db_connector, table_results)
