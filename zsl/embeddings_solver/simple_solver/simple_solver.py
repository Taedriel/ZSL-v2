import torch
import numpy as np

from zsl.word_embeddings.dowloader import Downloader
from zsl.word_embeddings.model import EmbeddingsLoader
from wikipedia2vec import Wikipedia2Vec

__all__ = ["Solver", "OutOfVocabSolver"]

class Solver(EmbeddingsLoader):
    """simple solver that compare the target embeddings to all the other embeddings to found the closest.
    """

    DEFAULT_MIN_LIST_RESULT = 10

    def __init__(self, embeddings):
        super(Solver, self).__init__(embeddings)

    def get_nearest_embedding_of(self, embedding, nb = 10):
        """return the top nearest embeddings compared to all other from the embeddings file

        Args:
            embedding (List[float]): the embedding to compare
            nb (int, optional): the number of element in the list. Defaults to 10.

        Raises:
            Exception: if there are not enoug embedding to fill the list

        Returns:
            List[float]: a list of most similar embeddings 
        """

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
        """return the cosine similarity between two embeddings>

        This method was intended to be use as a score function

        Args:.
            embedding (List[float]): the first embedding
            target (List[float]): the second embedding to compare with

        Returns:
            float: the score, i.e. the cosine similarity of the two embedding
        """
        target_embeddings = self.embeddings[target]

        cos = torch.nn.CosineSimilarity(dim=0)
        return float(cos(embedding, target_embeddings))

    def least_squared_score(self, embedding, target):
        """return the least square between two embeddings.

        This method was intended to be use as a score function


        Args:.
            embedding (List[float]): the first embedding
            target (List[float]): the second embedding to compare with

        Returns:
            float: the score, i.e. the least squared of the two embedding
        """
        target_embeddings = self.embeddings[target]
        return float(np.linalg.norm(target_embeddings - embedding))

    def mean_squared_score(self, embedding, target):
        """return the mean squared between two embeddings.

        This method was intended to be use as a score function

        Args:.
            embedding (List[float]): the first embedding
            target (List[float]): the second embedding to compare with

        Returns:
            float: the score, i.e. the least squared of the two embedding
        """
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
