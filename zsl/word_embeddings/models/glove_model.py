import torch
import numpy as np

from typing import List
from zsl.word_embeddings.model import WordToVector, FixedEmbedding

__all__ = ["GloVEModel"]

class GloVEModel(FixedEmbedding):
    
    address = "https://nlp.stanford.edu/data/"
    all_dict = "glove_all"
    
    def __init__(self, list_tag : List[str]):
        WordToVector.__init__(self, list_tag)

        self.window_size = 300
        self.model_size = "GloVe"

        # filename = "glove.840B.300d.zip"
        filename = "glove.6B.zip"
        FixedEmbedding.__init__(self, GloVEModel.address, filename)

    def _in_list(self, word, list_tag, list_joined_tag):
        if word in list_tag:
            return word
        
        if word.capitalize() in list_tag:
            return word.capitalize()

        if word.lower() in list_tag:
            return word.lower()

        if word.replace("-"," ") in list_tag:
            return word.replace("-"," ")

        joined_ind = [i for i, item in list_joined_tag if item == word or item.capitalize() == word or item.lower() == word]
        if joined_ind:
            return list_tag[joined_ind]

        return False

    def _one_turn(self, resolve_dict = {}):
        uncaped_list_tag = [x.replace("_", " ") for x in self.list_tags]
        unk = [item if item not in resolve_dict.keys() else resolve_dict[item] for item in uncaped_list_tag]
        joined_unk = [(i, "".join(item)) for i, item in enumerate(unk) if len(item.split()) > 1]

        with open(self.downloader.path, "r") as file:
            for line in file:
                tag, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")

                r_tag = self._in_list(tag, unk, joined_unk)
                if r_tag is not False:
                    self.embeddings[r_tag] = torch.from_numpy(coefs)
                    unk.remove(r_tag)

        return unk