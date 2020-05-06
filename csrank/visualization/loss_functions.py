import numpy as np
import elkai
import matplotlib.pyplot as plt
import itertools

from matplotlib import gridspec
from scipy.spatial.distance import cdist
import keras.backend as K
import tensorflow as tf

from csrank.losses import tsp_dist_matrix_loss_wrapper, tsp_probability_matrix_loss, \
    pairwise_comparison_quadratic_loss_wrapper, hinged_rank_loss, smooth_rank_loss, plackett_luce_loss, \
    make_smooth_ndcg_loss
from csrank.metrics import path_len
from csrank.visualization.predictions import plot_path, set_up_figure
from csrank.visualization.util import tableau_10_colorblind_color_scheme


def visualize_tsp_loss_functions():
    # properties
    n_objects = 6
    vis = plt.plot
    #vis = plt.scatter

    # generate some data
    random_state = np.random.RandomState(seed=7)
    x = random_state.random_integers(low=0, high=10000, size=(1, n_objects, 2))
    matrix = cdist(x[0], x[0])
    matrix = matrix.astype(int)
    y = [np.asarray(np.argsort(elkai.solve_int_matrix(matrix)), dtype=int)]
    # this doesnt change anything
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # visualize it
    fig2 = plt.figure(constrained_layout=True, figsize=(14, 8))
    spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)
    ax = fig2.add_subplot(spec2[0, 0])
    plot_path(x_instances=x[0], rankings=y[0], color=tableau_10_colorblind_color_scheme['dark_blue'],
              line_style="--", label="shortest")
    set_up_figure(ax)
    plt.title("Example Path")

    # generate all different paths
    permutations = [[0, *elem] for elem in itertools.permutations(np.arange(1, n_objects))]
    permutations = [permutations[elem] for elem in range(0, len(permutations), 2)]

    # produce lengths for all the paths and sort
    lengths = [path_len(distances=matrix, ranking=permutation) for permutation in permutations]
    lengths = normalize_min_max(lengths)
    sorted_lengths = np.argsort(lengths)
    print(sorted_lengths)
    lengths_by_length = np.asarray(lengths)[sorted_lengths]

    # add plot for true values
    ax = fig2.add_subplot(spec2[0, 1])
    x_placeholder = np.arange(len(lengths))
    vis(x_placeholder, lengths_by_length, label="true_length")

    # add plot for loss
    # 1. sort the predictions by how good they are
    sorted_predictions = np.asarray(permutations)[sorted_lengths]

    # 2. define execution graph
    x_tensor = K.placeholder(shape=(None, n_objects, 2))
    y_pred_tensor = K.placeholder(shape=(None, n_objects))
    y_true_tensor = K.placeholder(shape=(None, n_objects))
    dist_matrix_loss = tsp_dist_matrix_loss_wrapper(x_tensor)(y_true_tensor, y_pred_tensor)
    probability_matrix_loss = tsp_probability_matrix_loss(y_true_tensor, y_pred_tensor)
    quadratic_comparison = pairwise_comparison_quadratic_loss_wrapper(x_tensor)(y_true_tensor, y_pred_tensor)
    hrl = hinged_rank_loss(y_true_tensor, y_pred_tensor)
    srl = smooth_rank_loss(y_true_tensor, y_pred_tensor)
    pll = plackett_luce_loss(y_true_tensor, y_pred_tensor)
    msnl = make_smooth_ndcg_loss(y_true_tensor, y_pred_tensor)

    # 3. execute
    with tf.Session() as sess:
        metric_values = []

        for prediction in sorted_predictions:
            metric_values.append(sess.run(fetches=[dist_matrix_loss,
                                                   probability_matrix_loss,
                                                   quadratic_comparison,
                                                   hrl,
                                                   srl,
                                                   pll,
                                                   msnl],
                                          feed_dict={
                                              x_tensor: x,
                                              y_true_tensor: y,
                                              y_pred_tensor: [prediction]
                                          }))

    # 4. plot
    metric_values = np.asarray(metric_values)
    # metric_values_1 = normalize_min_max(metric_values[:, 0])
    # vis(x_placeholder, metric_values_1, label="dist")

    metric_values_2 = normalize_min_max(metric_values[:, 1])
    vis(x_placeholder, metric_values_2, label="probability")

    # metric_values_3 = normalize_min_max(metric_values[:, 2])
    # vis(x_placeholder, metric_values_3, label="quadratic")

    # metric_values_4 = normalize_min_max(metric_values[:, 3])
    # vis(x_placeholder, metric_values_4, label="hinged_rank_loss")
    #
    # metric_values_5 = normalize_min_max(metric_values[:, 4])
    # vis(x_placeholder, metric_values_5, label="smooth_rank_loss")
    #
    # metric_values_6 = normalize_min_max(metric_values[:, 5])
    # vis(x_placeholder, metric_values_6, label="plackett_luce_loss")
    #
    # metric_values_7 = normalize_min_max(metric_values[:, 6])
    # vis(x_placeholder, metric_values_7, label=" make_smooth_ndcg_loss")

    # experimental = [[metric_values_1[i] + metric_values_2[i] + metric_values_3[i]]
    #                 for i in range(len(lengths_by_length))]
    # experimental = np.asarray(experimental) / 3
    # vis(x_placeholder, experimental[:, 0, 0], label="experimental")

    # show results
    plt.legend(frameon=False)
    plt.show()


def normalize_min_max(lengths):
    lengths += np.min(lengths)
    lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
    return lengths


visualize_tsp_loss_functions()
