from typing import List

from zsl.word_embeddings.model import WordToVector
from zsl.word_embeddings.bert_strategy import Sum4LastLayers
from transformers import RobertaModel, RobertaTokenizer
from .bert_model import BERTModel

__all__ = ["ROBERTAModel"]


class ROBERTAModel(BERTModel):
    """ use a ROBERTA Model, a bigger version of BERT to convert classes into their embeddings
    """

    def __init__(self, list_tag : List[str], big: bool = False, window : int = 100):
        WordToVector.__init__(self, list_tag)
        self.window_size = window

        self.model_size = "roberta-large" if big else "roberta-base"

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_size, padding=True, truncation=True)
        self.model = RobertaModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.merging_strategy = Sum4LastLayers()

        self.model.eval()