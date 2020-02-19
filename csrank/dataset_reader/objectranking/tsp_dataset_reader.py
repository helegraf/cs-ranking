import logging
import os
# import math
# import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
import elkai
# from sklearn.utils.multiclass import type_of_target
# import matplotlib.pyplot as plt
# import tensorflow as tf

from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.dataset_reader.util import standardize_features
#
# import csrank as cs
# from csrank import losses
# from csrank import metrics


class TSPDatasetReader(DatasetReader):
    def __init__(self, n_features=2, n_objects_train=4, n_objects_test=4, n_train_instances=10, n_test_instances=2,
                 random_state=None, include_id=False, filename="TSP_excerpt.csv", **kwargs):

        super(TSPDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder="", **kwargs)

        self.dataset_file = os.path.join(self.dirname, filename)
        self.n_features = int(n_features)
        self.n_objects_train = int(n_objects_train)
        self.n_objects_test = int(n_objects_test)
        self.n_train_instances = int(n_train_instances)
        self.n_test_instances = int(n_test_instances)
        self.n_instances = self.n_train_instances + self.n_test_instances
        self.random_state = check_random_state(random_state)
        self.include_id = bool(include_id)
        self.logger = logging.getLogger(TSPDatasetReader.__name__)

        self.__load_dataset__()

    def __load_dataset__(self):
        """
        Loads the dataset file (consisting of cities with ids and x-y coordinates) into an array of objects with
        features.

        The id-feature is included depending on user-options.

        Returns
        -------

        """
        self.logger.debug("Loading dataset")

        # read file; convert to numpy array
        self.objects = pd.read_csv(self.dataset_file, sep=',').to_numpy()

        # remove id if necessary
        if not self.include_id:
            self.objects = self.objects[:, 1:]

    def splitter(self, iter_, other):
        """
        Create new splits.

        Parameters
        ----------
        iter_ : iterator giving indices of instances in the
        other :
        """

        for train_idx, test_idx in iter:
            x_train, y_train, x_test, y_test = self.get_single_train_test_split()
            x_train, x_test = standardize_features(x_train, x_test)
            yield x_train, y_train, x_test, y_test

    def get_dataset_dictionaries(self):
        """
        Returns the dataset for the latest request of a single split or multiple splits.

        Returns
        -------
        The data + its labels for the latest split
        """
        return self.X, self.Y

    def get_single_train_test_split(self):
        # shuffle the data
        self.random_state.shuffle(self.objects)
        index = 0

        # create training data
        index, x_data_train, y_labels_train = self.create_dataset(index, self.n_train_instances, self.n_objects_train,
                                                                  self.objects.shape[1])

        # create test data
        index, x_data_test, y_labels_test= self.create_dataset(index, self.n_test_instances, self.n_objects_test,
                                                               self.objects.shape[1])

        return x_data_train, y_labels_train, x_data_test, y_labels_test

    def create_dataset(self, index, n_instances, n_objects, n_features):
        # sample instances
        x_data = np.empty((n_instances, n_objects, n_features))
        for instance in range(n_instances):
            # sample n_objects objects
            for object_ in range(n_objects):
                x_data[instance, object_] = self.objects[index]
                index = index + 1

        # predict labels with algorithm
        y_labels = np.zeros((n_instances, n_objects), dtype=int)
        for instance in range(n_instances):
            # fill matrix with distances (multiplied for higher precision)
            matrix = cdist(x_data[instance], x_data[instance])
            matrix = (matrix * 1000).astype(int)

            # predict labels
            y_labels[instance] = np.asarray(np.argsort(elkai.solve_int_matrix(matrix)), dtype=int)

        # scale to [0, 1]
        print("before", x_data.shape)
        x_data = np.asarray(
            [[[x_data[j, i, 0]/5200, x_data[j, i, 1]/3400] for i in range(len(x_data[j]))] for j in range(len(x_data))])
        print("after", x_data.shape)
        return index, x_data, y_labels

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)


# seed = 1234
# fold_id = 1
# random_state = np.random.RandomState(seed=seed + fold_id)
# reader = TSPDatasetReader(include_id=False, n_objects_train=5, n_objects_test=5, n_train_instances=5,
#                           n_test_instances=2, filename="cities.csv", random_state=random_state)
#
# x_train, y_train, x_test, y_test = reader.get_single_train_test_split()
#
# print("What is happening!")
# print("x_train")
# print(x_train)
# print("y_train")
# print(y_train)


# log_dir="/home/hgraf/tensorboard/logs/test_tsp/" + "dist_loss_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# fate = cs.FATEObjectRanker(n_object_features=2, loss_function=losses.tsp_dist_matrix_loss_wrapper,
#                            loss_function_requires_x_values=True)
# #fate = cs.FATEObjectRanker(n_object_features=2, loss_function=losses.tsp_probability_matrix_loss,
#    #                        loss_function_requires_x_values=False)
# #
#
#
# #fate.fit(x_train, y_train)
# fate.fit(x_train, y_train, callbacks=[tensorboard_callback])
#
# predictions = fate.predict(x_test)
#
# print("x_train", x_train.shape)
# print("y_train", y_train.shape)
# print("x_test", x_test.shape)
# print("y_test", y_test.shape)
# print("y_pred", predictions.shape)
#
# print(y_test)
# print(predictions)
#
# loss = metrics.tsp_loss(x_test, y_test, predictions)
# print("loss", loss)
# print("loss diffs", loss[2]-loss[1])
#
# y_test = y_test.astype(int)
#
#
# def plot_path(x_instances, rankings, color, line_style, label):
#     path = np.argsort(rankings)
#
#     global i, x_and_y
#     for i in range(len(path)):
#         x_and_y = np.append(x_instances[path[i - 1]], x_instances[path[i]]).reshape((2, 2))
#         if i == 0:
#             plt.plot(x_and_y[:, 0], x_and_y[:, 1], color=color, linestyle=line_style, label=label)
#         else:
#             plt.plot(x_and_y[:, 0], x_and_y[:, 1], color=color, linestyle=line_style)
#
#
# plot_path(x_test[0], y_test[0], "red", "-", "true")
# plot_path(x_test[0], predictions[0], "blue", "--", "predicted")
#
# print(fate.predict_scores(x_test))
#
# plt.legend()
# plt.title("Shortest Path (LKV algorithm) vs. predicted path \n(difference {}, quality {})"
#           .format(round((loss[2]-loss[1])[0], 2),
#                   round((loss[1][0]/loss[2][0] * 100)), 2))
# plt.show()
