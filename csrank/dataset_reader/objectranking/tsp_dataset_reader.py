import logging
import os
import math

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import elkai
from sklearn.utils.multiclass import type_of_target


from csrank import FATEChoiceFunction
from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.dataset_reader.util import standardize_features

import csrank as cs
from csrank import losses
from csrank import metrics


class TSPDatasetReader(DatasetReader):
    def __init__(self, n_features=2, n_objects_train=4, n_train_instances=10, n_test_instances=2, random_state=None,
                 include_id=False, filename="TSP_excerpt.csv", **kwargs):

        super(TSPDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder="", **kwargs)

        self.dataset_file = os.path.join(self.dirname, filename)
        self.n_features = n_features
        self.n_objects_train = n_objects_train
        self.n_objects_test = 4
        self.n_train_instances = n_train_instances
        self.n_test_instances = n_test_instances
        self.n_instances = self.n_train_instances + self.n_test_instances
        self.random_state = check_random_state(random_state)
        self.include_id = include_id
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

        # assign labels
        # TODO: refer to an algorithm for finding the optimal solution
        # create matrix for labeller

        # predict labels
        y_labels = np.zeros((n_instances, n_objects), dtype=float)
        print("target type ",  type_of_target(y_labels))
        for instance in range(n_instances):
            # fill matrix with distances
            matrix = np.empty((n_objects, n_objects), dtype=float)
            for objectsse in range(n_objects):
                matrix[objectsse] = dist(x_data[instance][objectsse-1][0]-x_data[instance][objectsse][0],
                                               x_data[instance][objectsse-1][1]-x_data[instance][objectsse][1])

            y_labels[instance] = np.empty(n_objects).astype(float)
            y_labels[instance] = np.asarray(elkai.solve_int_matrix(matrix), dtype=float)

        print(np.asarray([1, 2, 3, 4], dtype=int))
        print(y_labels)
        print("target type ", type_of_target(y_labels))
        return index, x_data, y_labels

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)


def dist(a, b):
    return math.sqrt(a * a + b * b)


reader = TSPDatasetReader(include_id=False, n_objects_train=10, n_train_instances=1000, n_test_instances=100,
                          filename="cities.csv")

x_train, y_train, x_test, y_test = reader.get_single_train_test_split()
print(x_train)
print(y_train)

fate = cs.FATEObjectRanker(n_object_features=2, loss_function=losses.tsp_dist_matrix_loss_wrapper,
                           loss_function_requires_x_values=True)
fate.fit(x_train, y_train)

predictions = fate.predict(x_test)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)
print("y_pred", predictions.shape)


print(metrics.tsp_loss(x_test, y_test, predictions))
print(y_test)
print(predictions)