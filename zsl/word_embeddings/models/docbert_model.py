import torch

from .bert_model import BERTModel
from zsl.word_embeddings.bert_strategy import Sum4LastLayers
from zsl.word_embeddings.model import WordToVector
from tqdm import tqdm
from typing import List
from math import ceil
from transformers import BertTokenizer, BertModel

import logging
log = logging.getLogger(__name__)


__all__ = ["DocBERTModel"]

class DocBERTModel(BERTModel):

    def __init__(self, list_tag : List[str], big : bool = False):
        WordToVector.__init__(self, list_tag)
        self.window_size = "document"

        self.model_size = "bert-large-uncased" if big else "bert-base-uncased"

        self.tokenizer = BertTokenizer.from_pretrained(self.model_size, padding=True, truncation=True)
        self.model = BertModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.max_size = self.tokenizer.model_max_length
        self.merging_strategy = Sum4LastLayers()

        self.model.eval()

    def _one_pass(self, subinputs):
        with torch.no_grad():
            outputs = self.model(input_ids = subinputs["input_ids"], attention_mask = subinputs["attention_mask"])

        hidden_states = outputs[2]

        # [# layers, # batches, # tokens, # features] ==> [# tokens, # layers, # features]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        return self.merging_strategy.merge(token_embeddings[0])


    def convert(self, article_ret):
        """ convert all word in their embeddings"""

        if len(self.list_tags) == 0:
            raise Exception("no tags yet !")

        log.info("Starting converting tokens...")
        nb_token = len(self.list_tags)
        for i, tag in tqdm(enumerate(self.list_tags), total = nb_token, desc=f"{'converting to embedding':30}", ncols=80):
            
            if tag in self.embeddings: continue

            article = article_ret.get_article(tag)

            if article.summary is None:
                log.warning(f"no article for {tag}")
                self.embeddings[tag] = self._one_pass(self.tokenizer(tag, return_tensors = "pt"))
                continue

            torch_cls = []

            ids = self.tokenizer.encode(article.summary)
            nb_token = len(ids)

            if nb_token < self.max_size:
                self.embeddings[tag] = self._one_pass(self.tokenizer(tag, return_tensors = "pt"))
                continue

            nb_pass = ceil(nb_token / self.max_size)
            log.info(f"{tag} is {nb_pass} pass")

            stop = 50
            for j in range(nb_pass):
                start = stop - 50
                stop = min(nb_token, start + self.max_size)
                
                sub_ids = ids[start:stop]

                subinputs = { "input_ids": torch.IntTensor(sub_ids).unsqueeze(0), \
                            "token_type_ids": torch.IntTensor([0 for k in range(len(sub_ids))]).unsqueeze(0), \
                            "attention_mask": torch.IntTensor([1 for k in range(len(sub_ids))]).unsqueeze(0)  }
                torch_cls.append(self._one_pass(subinputs))
                if stop == nb_token: break

            self.embeddings[tag] = torch.mean(torch.stack(tuple(t for t in torch_cls)), axis=0)