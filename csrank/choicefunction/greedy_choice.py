import logging

import numpy as np

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class GreedyKnapsack(ChoiceFunctions, Learner):
    def __init__(self, capacity, **kwargs):
        self.logger = logging.getLogger(GreedyKnapsack.__name__)
        self.capacity = capacity
        self.model = None

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, **kwargs):
        Y = np.zeros(X.shape[:-1])

        for instance in X:
            # divide value by weight to get profitability index
            values = [obj[1]/obj[0] for obj in instance]
            order = np.argsort(values)

            current_capacity = 0
            item = 0
            while(current_capacity + instance[order[item]][0] < self.capacity):
                Y[order[item]] = 1

        return Y

    def predict_scores(self, X, **kwargs):
        return self._predict_scores_fixed(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        if isinstance(scores, dict):
            result = dict()
            for n, score in scores.items():
                result[n] = self._predict_scores_fixed(score)
        else:
            result = self._predict_scores_fixed(X)
        return result

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        pass
