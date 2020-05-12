import io

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from csrank.visualization.util import tableau_10_colorblind_color_scheme


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def figure_to_bytes(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    return buf.getvalue()


def bytes_to_tensor(bytes):
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(bytes, channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def tsp_figure(input, true, prediction=None, metric=None, epoch=None):
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)

    # plot both predictions and correct path
    plot_path(input, true, tableau_10_colorblind_color_scheme['dark_blue'], "--", "shortest")
    plot_path(input, prediction, tableau_10_colorblind_color_scheme['orange'], "-", "predicted")

    set_up_figure(ax)

    # add legend
    plt.legend(frameon=False)

    # add additional information that is available
    additional_info = ""
    if epoch is not None:
        additional_info = "Epoch {}".format(epoch)
    if metric is not None:
        if not additional_info == "":
            additional_info += ", "
        additional_info += "Quality {}".format(round(metric, 3))

    if not additional_info == "":
        plt.title("Shortest Path vs. Predicted Path\n{}".format(additional_info))
    else:
        plt.title("Shortest Path vs. Predicted Path")

    return figure


def set_up_figure(ax):
    # remove chart junk and add labels that are necessary
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axis('equal')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def add_tsp_figure_to_plot(figure, gridspec_location, x, true, prediction):
    figure.add_subplot(gridspec_location)
    plot_path(x, true, tableau_10_colorblind_color_scheme['dark_blue'], "--", "shortest")
    plot_path(x, prediction, tableau_10_colorblind_color_scheme['orange'], "-", "predicted")
    plt.legend()


def plot_path(x_instances, rankings, color, line_style, label):
    path = np.argsort(rankings)

    for i in range(len(path)):
        x_and_y = np.append(x_instances[path[i - 1]], x_instances[path[i]]).reshape((2, 2))
        if i == 0:
            plt.plot(x_and_y[:, 0], x_and_y[:, 1], color=color, linestyle=line_style, label=label)
        else:
            plt.plot(x_and_y[:, 0], x_and_y[:, 1], color=color, linestyle=line_style)


def create_image_plotting_graph(iteration):
    matplotlib_img_bytes = tf.placeholder(dtype='string')
    tensorflow_img = bytes_to_tensor(matplotlib_img_bytes)
    summary_img = tf.summary.image("prediction_vis/"+str(iteration), tensor=tensorflow_img)
    merged = tf.summary.merge([summary_img])

    return matplotlib_img_bytes, merged


def create_attention_plotting_graph(iteration, layer_name):
    matplotlib_img_bytes = tf.placeholder(dtype='string')
    tensorflow_img = bytes_to_tensor(matplotlib_img_bytes)
    summary_img = tf.summary.image("attention_vis/{}/{}".format(layer_name, str(iteration)), tensor=tensorflow_img)
    merged = tf.summary.merge([summary_img])

    return matplotlib_img_bytes, merged


def create_scalar_plotting_graph(metric_name):
    value_placeholder = tf.placeholder(dtype='float')
    tmp_data = value_placeholder + 0
    summary_scalar = tf.summary.scalar("metric/" + metric_name, tensor=tmp_data)
    return value_placeholder, summary_scalar
