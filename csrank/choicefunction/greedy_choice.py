import logging

import numpy as np

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class GreedyKnapsack(ChoiceFunctions, Learner):
    def __init__(self, capacity, n_object_features, n_objects, **kwargs):
        self.logger = logging.getLogger(GreedyKnapsack.__name__)
        self.capacity = capacity
        self.model = None
        self.threshold = 0.5
        self.n_object_features = n_object_features
        self.n_objects = n_objects

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, standardizer, **kwargs):
        X = X.reshape(-1, self.n_object_features)
        X = standardizer.inverse_transform(X)
        X = X.reshape(-1, self.n_objects, self.n_object_features)
        Y = np.zeros(X.shape[:-1])

        for instance_num in range(len(X)):
            instance = X[instance_num]
            # divide value by weight to get profitability index
            values = [obj[1] / obj[0] for obj in instance]
            order = np.flip(np.argsort(values))

            current_capacity = 0
            for item in range(len(order)):
                if (current_capacity + instance[order[item]][0]) <= self.capacity:
                    Y[instance_num][order[item]] = 1
                    current_capacity += instance[order[item]][0]
        return Y

    def predict_scores(self, X, standarizer, **kwargs):
        return self._predict_scores_fixed(X, standarizer, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        pass
