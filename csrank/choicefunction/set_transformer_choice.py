import logging

from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

from csrank.choicefunction.choice_functions import ChoiceFunctions
from csrank.core.set_transformer_network import SetTransformer


class SetTransformerChoice(SetTransformer, ChoiceFunctions):

    def __init__(self, threshold=0.5, loss_function=binary_crossentropy, loss_function_requires_x_values=False,
                 **kwargs):
        super(SetTransformerChoice, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerChoice.__name__)
        self.threshold = threshold

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, verbose=0, **kwargs):
        if tune_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                super().fit(x_train, y_train, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(x_val, y_val, thin_thresholds=thin_thresholds, verbose=verbose)
        else:
            super().fit(X, Y, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)