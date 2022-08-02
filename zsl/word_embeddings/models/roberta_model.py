from typing import List

from ..model import WordToVector
from ..bert_strategy import Sum4LastLayers
from .bert_model import BERTModel

from transformers import RobertaModel, RobertaTokenizer

__all__ = ["ROBERTAModel"]


class ROBERTAModel(BERTModel):

    def __init__(self, list_tag : List[str], big: bool = False, window : int = 100):
        WordToVector.__init__(self, list_tag)
        self.window_size = window

        self.model_size = "roberta-large" if big else "roberta-base"

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_size, padding=True, truncation=True)
        self.model = RobertaModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.merging_strategy = Sum4LastLayers()

        self.model.eval()