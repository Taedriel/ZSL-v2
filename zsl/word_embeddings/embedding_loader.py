import torch
import logging
import tqdm

from typing import Tuple, List, Dict
from .bert_strategy import SimilarityStrategy

__all__ = ["EmbeddingsLoader"]

class EmbeddingsLoader:

    """Class that load an embeddings file to perform operation on it. Base class
     for multiple operations such as matrix similarity operations.

     All embeddings should be csv file with a one line header containing at least one columns named "embeddings"
     """

    def __init__(self, filename : str):

        self.file = filename
        self.embeddings = {}

        self._load_file()

    def _load_file(self) -> None:
        try:
            with open(self.file, "r") as f:
                lines = f.readlines()
                
            for line in lines[1:]:
                data = line.split(",")
                self.embeddings[data[0]] = torch.FloatTensor(list(map(float, data[1:])))

        except IOError as e:
            raise IOError(f"No file {self.file}")

class SimilarityMatrix(EmbeddingsLoader):

    def __init__(self, embeddings : Dict[str, List[float]], strategy : SimilarityStrategy):
        EmbeddingsLoader.__init__(self, embeddings)
        self.strategy = strategy
        self._create_matrix()
        self.computed : bool = False

    def _create_matrix(self) -> None:
        n_tokens = len(self.embeddings)
        self.cosine_sim_matrix : Dict[Dict[float]] = {}
        for tag in self.embeddings.keys():
            self.cosine_sim_matrix[tag] = {}

    def compute_sim(self) -> None:
        """ compute cosine similarity between all vectors """

        closed_list = []

        logging.info("Computing cosine similarity, this could take some time...")
        for tag, vector in tqdm(self.embeddings.items(), total = len(self.embeddings), desc=f"{'computing sim matrix':30}"):

            for otag, other_vector in self.embeddings.items():

                if otag == tag: continue
                # if (tag, otag) in closed_list or (otag, tag) in closed_list: continue

                similarity = self.strategy.sim(vector, other_vector)

                self.cosine_sim_matrix[otag][tag] = similarity
                self.cosine_sim_matrix[tag][otag] = similarity

                # closed_list.append((tag, otag))
                # closed_list.append((otag, tag))

        self.computed = True

    def export_sim_matrix(self, filename):
        if not self.computed:
            self.compute_sim()
        
        try:
            f = open(filename, "w")
        except OSError:
            raise OSError("Could not open file")

        with f:
            print("/", *[tag for tag in self.embeddings.keys()], sep = ",", file = f)

            for tag in self.embeddings.keys():
                print(tag, *[str(round(float(self.cosine_sim_matrix[tag][otag]), 3)) for otag in self.embeddings.keys()], sep = ",", file = f)

    def get_sim_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """return the similarity matrix of the embeddings
        """
        if not self.computed:
            self.compute_sim()

        X = len(self.embeddings)
        matrix = [[0 for j in range(X)] for i in range(X)]
        ids = []
        
        for i, tag in enumerate(self.embeddings.keys()):
            ids.append(tag)
            for j, otag in enumerate(self.embeddings.keys()):
                if i == j:
                    continue

                matrix[i][j] = self.cosine_sim_matrix[tag][otag]
                matrix[j][i] = self.cosine_sim_matrix[tag][otag]

        return ids, matrix

    def sim_between(self, token1 : str, token2 : str) -> float:
        v1 = self.embeddings[token1]
        v2 = self.embeddings[token2]

        if token2 not in self.cosine_sim_matrix[token1] or token1 not in self.cosine_sim_matrix[token2]:
            similarity = self.strategy.sim(v1, v2)
            self.computed = True

            self.cosine_sim_matrix[token1][token2] = similarity
            self.cosine_sim_matrix[token1][token2] = similarity

        return self.cosine_sim_matrix[token1][token2]