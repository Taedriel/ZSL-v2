import torch

from typing import List
from wikipedia2vec import Wikipedia2Vec
from ..model import WordToVector, FixedEmbedding


import logging
log = logging.getLogger(__name__)

__all__ = ["Wiki2VecModel"]


class Wiki2VecModel(FixedEmbedding):

    address = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/"
    
    def __init__(self, list_tag : List[str], size : int = 300):
        WordToVector.__init__(self, list_tag)
        self.window_size = size
        self.model_size = "wikipedia2vec"

        assert size in [100, 300, 500], f"size should be one of this value (100, 300, 500)"
        filename = f"enwiki_20180420_{self.window_size}d.pkl.bz2"
        FixedEmbedding.__init__(self, Wiki2VecModel.address, filename)
        
        self.model = Wikipedia2Vec.load(self.downloader.path)

    def _retrieve(self, word):
        try: return self.model.get_word_vector(word)
        except: pass

        try: return self.model.get_word_vector(word.capitalize())
        except: pass

        try: return self.model.get_word_vector(word.lower())
        except: pass

        try: return self.model.get_entity_vector(word)
        except: pass

        try: return self.model.get_entity_vector(word.capitalize())
        except: pass
        
        try: return self.model.get_entity_vector(word.lower())
        except: pass

        return None
    
    def _one_turn(self, resolve_dict = {}):
        unk = []

        for word in self.list_tags:
            w = word.replace("_", " ")
            if w in self.embeddings: continue

            if w in resolve_dict:
                embed = self._retrieve(resolve_dict[w])
            else:
                embed = self._retrieve(w)

            if embed is None:
                log.warning(f"{w} cannot be retrieved.")
                unk.append(word)
            else:
                self.embeddings[w] = torch.from_numpy(embed)
        
        return unk