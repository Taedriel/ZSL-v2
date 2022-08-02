from typing import List
from scipy.spatial.distance import cityblock

import numpy as np
import torch

__all__ = ["BERTMergeStrategy", "Sum4LastLayers", "Concat4LastLayer", "SimilarityStrategy", "CosineSim", "EuclidianDistSim", "ManhattanDistSim"]

class BERTMergeStrategy:
    """ strategy to extract BERT embeddings
    
    different approach exist, 
    see https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/BERT/bert-feature-extraction-contextualized-embeddings.png
    for more possible strategy
    """

    def merge(self, vector : List[List[float]]) -> torch.Tensor:
        raise NotImplementedError

class Sum4LastLayers(BERTMergeStrategy):

    def merge(self, vector : List[List[float]]) -> torch.Tensor:
        return torch.sum(vector[-4:], dim = 0)

class Concat4LastLayer(BERTMergeStrategy):

    def merge(self, vector : List[List[float]]) -> torch.Tensor:
        return torch.concat(vector[-4:], dim = 0)

class SimilarityStrategy:
    """strategy to compute similarity between embeddings. Cosine similarity should 
    be the only valid one in word embeddings, other aren't relevant
    """

    def sim(self, embed1 : List[float], embed2 : List[float]) -> float:
        raise NotImplementedError

class CosineSim(SimilarityStrategy):

    def sim(self, embed1 : List[float], embed2 : List[float]) -> float:
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(embed1, embed2)

class EuclidianDistSim(SimilarityStrategy):

    def sim(self, embed1 : List[float], embed2 : List[float]) -> float:
        return np.linalg.norm(embed1-embed2)

class ManhattanDistSim(SimilarityStrategy):

    def sim(self, embed1 : List[float], embed2 : List[float]) -> float:
        return cityblock(embed1, embed2)