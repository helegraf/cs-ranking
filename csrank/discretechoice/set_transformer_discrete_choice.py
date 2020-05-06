import logging

from csrank.core.set_transformer_network import SetTransformer
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser


class SetTransformerDiscreteChoice(SetTransformer, DiscreteObjectChooser):

    def __init__(self, loss_function='categorical_hinge', loss_function_requires_x_values=False, **kwargs):
        super(SetTransformerDiscreteChoice, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerDiscreteChoice.__name__)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)
