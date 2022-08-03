
import torch
import logging
import tqdm

from typing import List
from word_embeddings.bert_strategy import Sum4LastLayers
from word_embeddings.article import ArticleRetriever
from word_embeddings.model import WordToVector

from transformers import BertTokenizer, BertModel

__all__ = ["BERTModel"]

class BERTModel(WordToVector):

    temp_dir = "./temp"

    def __init__(self, list_tag : List[str], big: bool = False, window : int = 100):
        super(BERTModel, self).__init__(list_tag)
        self.window_size = window

        self.model_size = "bert-large-uncased" if big else "bert-base-uncased"

        self.tokenizer = BertTokenizer.from_pretrained(self.model_size, padding=True, truncation=True)
        self.model = BertModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.merging_strategy = Sum4LastLayers()

        self.model.eval()

    def _one_pass(self, inputs):
        with torch.no_grad():
            outputs = self.model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"])

        hidden_states = outputs[2]

        # [# layers, # batches, # tokens, # features] ==> [# tokens, # layers, # features]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        return self.merging_strategy.merge(token_embeddings[0])

    def convert(self, article_ret : ArticleRetriever):
        """ convert all word in their embeddings"""

        if len(self.list_tags) == 0:
            raise Exception("no tags yet !")

        logging.info("Starting converting tokens...")
        nb_token = len(self.list_tags)
        for i, tag in tqdm(enumerate(self.list_tags), total = nb_token, desc=f"{'converting to embedding':30}"):
            
            if tag in self.embeddings: continue

            article = article_ret.get_article(tag)
            if article.summary is None or self.window_size == 0:
                self.embeddings[tag] = self._one_pass(self.tokenizer(tag, return_tensors = "pt"))
                continue

            sub_ids = self.tokenizer.encode(tag + ". " + article.summary)[0:self.window_size]
            subinputs = {   "input_ids": torch.IntTensor(sub_ids).unsqueeze(0), \
                            "token_type_ids": torch.IntTensor([0 for k in range(len(sub_ids))]).unsqueeze(0), \
                            "attention_mask": torch.IntTensor([1 for k in range(len(sub_ids))]).unsqueeze(0)  }

            self.embeddings[tag] = self._one_pass(subinputs)