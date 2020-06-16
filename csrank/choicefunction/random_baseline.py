import logging

from sklearn.utils import check_random_state

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class RandomChoice(ChoiceFunctions, Learner):
    def __init__(self, random_state, **kwargs):
        self.random_state = check_random_state(random_state)
        self.threshold = 0
        self.logger = logging.getLogger(RandomChoice.__name__)
        self.model = None

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, **kwargs):
        return self.random_state.uniform(low=-1, high=1, size=X.shape[:2])

    def predict_scores(self, X, **kwargs):
        self.logger.info("Predicting scores")
        return self._predict_scores_fixed(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        pass
