from typing import List

from zsl.word_embeddings.bert_strategy import Sum4LastLayers
from zsl.word_embeddings.model import WordToVector
from transformers import RobertaModel, RobertaTokenizer
from .docbert_model import DocBERTModel


__all__ = ["DocBERTAModel"]

class DocBERTAModel(DocBERTModel):

    def __init__(self, list_tag : List[str], big : bool = False):
        WordToVector.__init__(self, list_tag)
        self.window_size = "document"

        self.model_size = "roberta-large" if big else "roberta-base"

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_size, padding=True, truncation=True)
        self.model = RobertaModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.max_size = self.tokenizer.model_max_length

        self.merging_strategy = Sum4LastLayers()

        self.model.eval()