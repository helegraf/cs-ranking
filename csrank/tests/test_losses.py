import numpy as np
from keras import backend as K
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import cdist
import tensorflow as tf

from csrank.losses import hinged_rank_loss, smooth_rank_loss, plackett_luce_loss, l1_matrix, l2_matrix, \
    pairwise_hinge_matrix, tsp_dist_matrix_loss_wrapper, tsp_probability_matrix_loss, \
    pairwise_comparison_quadratic_loss_wrapper

decimal = 3


def test_pairwise_comparison_quadratic_loss():
    x = [[[0, 0], [3, 1], [4, 2], [2, 4], [1, 3]]]
    pairwise_comparison_quadratic_loss = pairwise_comparison_quadratic_loss_wrapper(K.constant(x))
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    assert_almost_equal(
        actual=K.eval(
            pairwise_comparison_quadratic_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        desired=np.array([1.45]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            pairwise_comparison_quadratic_loss(
                y_true_tensor, K.constant(np.array([[0., .1, .2, .3, .4]]))
            )
        ),
        desired=np.array([3.e-05]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            pairwise_comparison_quadratic_loss(
                y_true_tensor, K.constant(np.array([[0., .3, .1, .4, .2]]))
            )
        ),
        desired=np.array([-0.031]),
        decimal=decimal,
    )


def test_tsp_probability_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    assert_almost_equal(
        actual=K.eval(
            tsp_probability_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        #desired=np.array([5.]),
        desired=np.array([25.]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            tsp_probability_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., .1, .2, .3, .4]]))
            )
        ),
        #desired=np.array([6.621]),
        desired=np.array([28.45]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            tsp_probability_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., .3, .1, .4, .2]]))
            )
        ),
        #desired=np.array([5.872]),
        desired=np.array([26.508]),
        decimal=decimal,
    )


def test_tsp_distance_loss():
    x = [[[0, 0], [3, 1], [4, 2], [2, 4], [1, 3]]]
    tsp_dist_matrix_loss = tsp_dist_matrix_loss_wrapper(K.constant(x))
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    assert_almost_equal(
        actual=K.eval(
            tsp_dist_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        desired=np.array([9.08]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            tsp_dist_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., .1, .2, .3, .4]]))
            )
        ),
        desired=np.array([10.944]),
        decimal=decimal,
    )

    assert_almost_equal(
        actual=K.eval(
            tsp_dist_matrix_loss(
                y_true_tensor, K.constant(np.array([[0., .3, .1, .4, .2]]))
            )
        ),
        desired=np.array([10.178]),
        decimal=decimal,
    )


def mask_sum(y_true):
    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')
    n = K.sum(mask, axis=(1, 2))
    return n


def test_hinge_loss_matrix():
    y_true_input = np.asarray([[0., .1, .2, .3]])
    y_pred_input = np.asarray([[.3, .2, .1, 0]])
    y_pred_input_2 = np.asarray([[1., 2., 3., 4.]])

    y_true = K.placeholder(shape=y_true_input.shape)
    y_pred = K.placeholder(shape=y_true_input.shape)
    res = pairwise_hinge_matrix(y_pred=y_pred, y_true=y_true)
    mask = mask_sum(y_true)

    with tf.Session() as sess:
        result1, mask_1 = sess.run(fetches=[res, mask], feed_dict={y_true: y_true_input, y_pred: y_true_input})
        result2, mask_2 = sess.run(fetches=[res, mask], feed_dict={y_true: y_true_input, y_pred: y_pred_input})
        result3, mask_3 = sess.run(fetches=[res, mask], feed_dict={y_true: y_true_input, y_pred: y_pred_input_2})

        assert_almost_equal(actual=np.sum(result1), desired=7., decimal=decimal)
        assert_almost_equal(actual=np.sum(result2), desired=5., decimal=decimal)


def test_l1_distance():
    matrix = np.asarray([[[1, 0], [0, 0], [0, 2]]])
    input_ = K.placeholder(shape=matrix.shape)

    result = l1_matrix(input_, input_)

    with tf.Session() as sess:
        result = sess.run(fetches=[result], feed_dict={input_: matrix})[0]
        correct_result = cdist(matrix[0], matrix[0], metric="cityblock")
        assert_almost_equal(actual=result, desired=[correct_result], decimal=decimal)


def test_l2_distance():
    matrix = np.asarray([[[1, 0], [0, 0], [0, 2]]])
    input_ = K.placeholder(shape=matrix.shape)

    result = l2_matrix(input_, input_)

    with tf.Session() as sess:
        result = sess.run(fetches=[result], feed_dict={input_: matrix})[0]
        correct_result = cdist(matrix[0], matrix[0], metric="euclidean")
        assert_almost_equal(actual=result, desired=[correct_result], decimal=decimal)


def test_hinged_rank_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    # Predicting all 0, gives an error of 1.0:
    assert_almost_equal(
        actual=K.eval(
            hinged_rank_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        desired=np.array([1.]),
        decimal=decimal,
    )

    # Predicting the correct ranking improves, but penalizes by difference of
    # scores:
    assert_almost_equal(
        actual=K.eval(
            hinged_rank_loss(
                y_true_tensor, K.constant(np.array([[.2, .1, .0, -0.1, -0.2]]))
            )
        ),
        desired=np.array([0.8]),
        decimal=decimal,
    )


def test_plackett_luce_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)
    assert_almost_equal(
        actual=K.eval(
            plackett_luce_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        desired=np.array([4.78749]),
        decimal=decimal,
    )


def test_smooth_rank_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    # Predicting all 0, gives an error of 1.0:
    assert_almost_equal(
        actual=K.eval(
            smooth_rank_loss(
                y_true_tensor, K.constant(np.array([[0., 0., 0., 0., 0.]]))
            )
        ),
        desired=np.array([1.]),
        decimal=decimal,
    )

    # Predicting the correct ranking improves, but penalizes by difference of
    # scores:
    assert_almost_equal(
        actual=K.eval(
            smooth_rank_loss(y_true_tensor,
                             K.constant(np.array([[.2, .1, .0, -0.1, -0.2]])))),
        desired=np.array([0.82275984]))
