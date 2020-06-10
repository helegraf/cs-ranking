import numpy as np
import matplotlib.pyplot as plt

from csrank.visualization.util import tableau_10_colorblind_color_scheme


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    mat_size_x = len(matrix)
    for x in range(mat_size_x):
        mat_size_y = len(matrix[x])
        for y in range(mat_size_y):
            w = matrix[x, y]
            x_index = y
            y_index = x
            # for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x_index - size / 2, (mat_size_x - y_index) - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.autoscale_view()
    ax.invert_yaxis()


def visualize_attention_scores(query, key, scores):
    n_obj_key = len(key)
    n_obj_query = len(query)
    key = np.transpose(key)
    n_features = len(key)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(n_features, n_obj_key), height_ratios=(n_features, n_obj_query),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 1])
    ax_query = fig.add_subplot(gs[1, 0], sharey=ax)
    ax_query.set_title("Query")

    ax_key = fig.add_subplot(gs[0, 1], sharex=ax)
    ax_key.set_title('Key')

    hinton(scores, ax=ax)
    hinton(query, ax=ax_query)
    hinton(key, ax=ax_key)

# np.random.seed(19680801)
#
# n_obj_total = 20
# n_features = 4
#
# #scores = np.random.rand(n_obj_total, n_obj_total) - 0.5
# scores = np.identity(n_obj_total)
# scores[0, 0] = -1
# scores[0, n_obj_total-1] = -.5
# #query = np.random.rand(n_obj_total, n_features) - 0.5
# query = (np.arange(n_obj_total * n_features) / (n_obj_total * n_features)) - 0.5
# query = query.reshape(n_obj_total, n_features)
# query[0, n_features-1] = 1
# #key = np.random.rand(n_obj_total, n_features) - 0.5
# key = (np.empty(shape=(n_obj_total * n_features)) / (n_obj_total * n_features)) - 0.5
# key = key.reshape(n_obj_total, n_features)
# key[0, n_features-1] = 1
#
# visualize_attention_scores(query, key, scores)
#
# plt.show()

def visualize_lr_acc(lr_logs, loss_logs, val_loss_logs=None):
    dark_grey = tableau_10_colorblind_color_scheme['dark_grey']

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_xlabel("learning rate", color=dark_grey)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['bottom'].set_color(dark_grey)
    ax.spines['left'].set_color(dark_grey)

    # ax.xaxis.label.set_color(dark_grey)
    ax.tick_params(axis='x', colors=dark_grey)
    ax.tick_params(axis='y', colors=dark_grey)

    # plot both graphs
    plt.plot(lr_logs, loss_logs, color=tableau_10_colorblind_color_scheme['orange'], linestyle="-", label="loss")
    if val_loss_logs is not None:
        plt.plot(lr_logs, val_loss_logs, color=tableau_10_colorblind_color_scheme['blue'], linestyle="--", label="validation loss")
    ax.set_ylim(bottom=0)
    plt.legend(frameon=False)
