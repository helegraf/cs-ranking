import logging

from csrank.core.set_transformer_network import SetTransformer
from csrank.losses import hinged_rank_loss
from csrank.objectranking.object_ranker import ObjectRanker


class SetTransformerObjectRanker(SetTransformer, ObjectRanker):
    def __init__(self, loss_function=hinged_rank_loss, loss_function_requires_x_values=False, **kwargs):
        super(SetTransformerObjectRanker, self) \
            .__init__(loss_function=loss_function, loss_function_requires_x_values=loss_function_requires_x_values,
                      **kwargs)

        self.logger = logging.getLogger(SetTransformerObjectRanker.__name__)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)