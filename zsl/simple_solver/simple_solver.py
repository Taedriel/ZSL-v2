import torch
import numpy as np

from wikipedia2vec import Wikipedia2Vec
from ..word_embeddings import Downloader
from ..word_embeddings import EmbeddingsLoader

class Solver(EmbeddingsLoader):

    DEFAULT_MIN_LIST_RESULT = 10

    def __init__(self, embeddings):
        super(Solver, self).__init__(embeddings)

    def get_nearest_embedding_of(self, embedding, nb = 10):

        if nb > len(self.embeddings):
            raise Exception("nb too high, not enough token")

        nearest = []
        for tag, e in self.embeddings.items():

            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(embedding, e)

            nearest.append((tag, similarity))
        
        nearest.sort(key = lambda tup : tup[1])
        return nearest[-1:-nb-1:-1]

    def __call__(self, embeddeding, tag=None):
        result = self.get_nearest_embedding_of(embeddeding, min(Solver.DEFAULT_MIN_LIST_RESULT, len(self.embeddings)))
        if tag is not None:
            print(f"Nearest Word for {tag}:")
        for i in result:
            print(f"\t{i[0]:12}: {round(float(i[1]) * 100, 3)}%")
    
    def score(self, embedding, target):
        target_embeddings = self.embeddings[target]

        cos = torch.nn.CosineSimilarity(dim=0)
        return float(cos(embedding, target_embeddings))

    def least_squared_score(self, embedding, target):
        target_embeddings = self.embeddings[target]
        return float(np.linalg.norm(target_embeddings - embedding))

    def mean_squared_score(self, embedding, target):
        target_embeddings = self.embeddings[target]
        return float(np.square(np.subtract(embedding, target_embeddings)).mean())

class OutOfVocabSolver(Downloader):

    DEFAULT_MIN_LIST_RESULT = 10

    def __init__(self):
        address = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/"
        filename = "enwiki_20180420_300d.pkl.bz2"

        super(OutOfVocabSolver, self).__init__(address, filename)

        self.download()
        self.model = Wikipedia2Vec.load(self.path)

    def get_nearest_embedding_of(self, embedding, nb):
        embedding = np.array(embedding)
        return self.model.most_similar_by_vector(embedding, count=nb, min_count=nb)

    def __call__(self, embedding, tag = None):
        result = self.get_nearest_embedding_of(embedding,OutOfVocabSolver.DEFAULT_MIN_LIST_RESULT)
        if tag is not None:
            print(f"Nearest Word for {tag}:")
        for i in result:
            print(f"\t{repr(i[0]):12}: {round(float(i[1]) * 100, 3)}%")
