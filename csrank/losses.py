import numpy as np
import tensorflow as tf
from keras import backend as K

from csrank.tensorflow_util import tensorify

__all__ = ['hinged_rank_loss', 'make_smooth_ndcg_loss', 'smooth_rank_loss',
           'plackett_luce_loss', 'tsp_dist_matrix_loss_wrapper', 'tsp_probability_matrix_loss']


def identifiable(loss_function):
    def wrap_loss(y_true, y_pred, *args, **kwargs):
        alpha = 1e-4
        ss = tf.reduce_sum(tf.square(y_pred), axis=1)
        ss = tf.cast(ss, tf.float32)
        return alpha * ss + loss_function(y_true, y_pred, *args, **kwargs)

    return wrap_loss


@identifiable
def hinged_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)

    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')
    diff = y_pred[:, :, None] - y_pred[:, None]
    hinge = K.maximum(mask * (1 - diff), 0)
    n = K.sum(mask, axis=(1, 2))

    return K.sum(hinge, axis=(1, 2)) / n


@identifiable
def smooth_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')
    exped = K.exp(y_pred[:, None] - y_pred[:, :, None])
    result = K.sum(exped * mask, axis=[1, 2])
    return result / K.sum(mask, axis=(1, 2))


@identifiable
def plackett_luce_loss(y_true, s_pred):
    y_true = tf.cast(y_true, dtype='int32')
    s_pred = tf.cast(s_pred, dtype='float32')
    m = tf.shape(y_true)[1]
    raw_max = tf.reduce_max(s_pred, axis=1, keepdims=True)
    max_elem = tf.stop_gradient(tf.where(
        tf.is_finite(raw_max),
        raw_max,
        tf.zeros_like(raw_max)))
    exped = tf.exp(tf.subtract(s_pred, max_elem))
    masks = tf.greater_equal(y_true, tf.range(m)[:, None, None])
    tri = exped * tf.cast(masks, tf.float32)
    lse = tf.reduce_sum(tf.log(tf.reduce_sum(tri, axis=2)), axis=0)
    return lse - tf.reduce_sum(s_pred, axis=1)


def make_smooth_ndcg_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_objects = K.max(y_true) + 1.
    y_true_f = K.cast(y_true, 'float32')
    relevance = n_objects - y_true_f - 1.
    log_term = K.log(relevance + 2.) / K.log(2.)
    exp_relevance = K.pow(2., relevance) - 1.
    gains = exp_relevance / log_term

    # Calculate ideal dcg:
    idcg = K.sum(gains, axis=-1)

    # Calculate smoothed dcg:
    exped = K.exp(y_pred)
    exped = exped / K.sum(exped, axis=-1, keepdims=True)
    # toppred, toppred_ind = tf.nn.top_k(gains * exped, k)
    return 1 - K.sum(exped * gains, axis=-1) / idcg


def tsp_dist_matrix_loss_wrapper(x):
    """
    A wrapper for the function tsp_dist_matrix_loss.

    This wrapper allows the function to use the additional parameter of the object features internally, while it can
    be used with the two parameters of true and predicted values externally, conforming to interface restrictions.

    Parameters
    ----------
    x : tensor
        a tensor matrix of object features (3 axes)

    Returns
    -------
    A tsp_dist_matrix_loss function that can be used with the two parameters of true and predicted values,
    while internally having access to the corresponding objects and their features.
    """
    # tensorify
    x = tensorify(x)

    @identifiable
    def tsp_dist_matrix_loss(y_true, y_pred):
        """
        Computes a loss specific to TSP-datasets using actual distances between objects.

        The loss is computed by multiplying the pairwise distance matrix between objects with the pairwise hinge loss
        matrix.

        Parameters
        ----------
        y_true : tensor
            tensor matrix of true values (2 axes)
        y_pred : tensor
            tensor matrix of predicted values (2 axes)

        Returns
        -------
        A tensor matrix (1 axis) containing a loss value for each input instance
        """
        # tensorify
        y_true = tensorify(y_true)
        y_pred = tensorify(y_pred)

        # compute distances
        distance_matrix = l2_matrix(x, x)
        distance_matrix = tf.linalg.l2_normalize(distance_matrix)

        # compute matrix of pairwise hinge loss
        hinge_matrix = pairwise_hinge_matrix(y_true, y_pred)

        # sum multiplication of results
        multiplier = distance_matrix * hinge_matrix

        return K.sum(multiplier, axis=(1, 2))

    return tsp_dist_matrix_loss


def l1_matrix(a, b):
    """
    Computes the L1 (Manhattan) distance for each instance of a and b
    Parameters
    ----------
    a : tensor
        a tensor representing list of instances consisting of objects with real coordinates (3 axes)
    b : tensor
        a tensor representing list of instances consisting of objects with real coordinates (3 axes)

    Returns
    -------
    a tensor representing the pairwise distances between objects for each instance (3 axes)
    """
    return K.sum(K.abs(a[:, None] - b[:, :, None]), axis=-1)


def l2_matrix(a, b):
    """
    Computes the L2 (Euclidian) distance for each instance of a and b
    Parameters
    ----------
    a : tensor
        a tensor representing list of instances consisting of objects with real coordinates (3 axes)
    b : tensor
        a tensor representing list of instances consisting of objects with real coordinates (3 axes)

    Returns
    -------
    a tensor representing the pairwise distances between objects for each instance (3 axes)
    """
    return K.sqrt(K.sum(K.square(a[:, None] - b[:, :, None]), axis=-1))


@identifiable
def tsp_probability_matrix_loss(y_true, y_pred):
    """
    Computes a loss specific to TSP-datasets using rank probabilities between objects.

    The loss is computed by multiplying the rank probability matrix (containing estimates of the probabilities for each
    object pair A, B that A is ranked before B, estimated as by the equation below) between objects with the pairwise
    hinge loss matrix.

    .. math:: exp(U_A)/(exp(U_A)+exp(U_B))

    Parameters
    ----------
    y_true : tensor
        tensor matrix of true values (2 axes)
    y_pred : tensor
        tensor matrix of predicted values (2 axes)

    Returns
    -------
    A tensor matrix (1 axis) containing a loss value for each input instance
    """
    # tensorify
    y_true = tensorify(y_true)
    y_pred = tensorify(y_pred)

    # compute matrix of U_A / (U_A + U_B)
    exp_pred = K.exp(y_pred)
    exp_matrix = exp_pred[:, None] / (exp_pred[:, None] + exp_pred[:, :, None])

    # compute matrix of pairwise hinge loss
    hinge_matrix = pairwise_hinge_matrix(y_true, y_pred)
    multiplier = exp_matrix * hinge_matrix

    # sum multiplication of results
    return K.sum(multiplier, axis=(1, 2))


def pairwise_hinge_matrix(y_true, y_pred):
    """
    Computes the pairwise hinge loss matrix for given ground truth values and predictions

    Parameters
    ----------
    y_true : tensor
        tensor matrix of true values (2 axes)
    y_pred : tensor
        tensor matrix of predicted values (2 axes)

    Returns
    -------
    A tensor representing a matrix containing the pairwise hinge loss between predicted values (3 axes)
    """

    # mask: only compute entries in the matrix where i > j
    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')

    # compute actual hinge loss
    return mask * (1 - (y_pred[:, :, None] - y_pred[:, None]))