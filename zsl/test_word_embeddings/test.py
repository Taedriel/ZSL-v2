import logging
import torch
import tqdm
from time import perf_counter
from typing import Tuple

from scipy.stats import spearmanr

from simple_solver.simple_solver import Solver
from word_embeddings.model import WordToVector
from word_embeddings.article import ArticleRetriever
from word_embeddings.embedding_loader import SimilarityMatrix
from word_embeddings.bert_strategy import CosineSim

__all__ = ["EmbeddingDistanceTest", "SyntacticTest", "SimilarityTest"]

class Test:

    def __init__(self, name):
        self.name = name
        self.vocab = []
    
    def _start(self, model : WordToVector, articlesRetriever : ArticleRetriever):
        articlesRetriever.set_list_vocab(f"{self.name}.art", self.vocab)

        if articlesRetriever(force_reload = False):
            articlesRetriever.save()

        self.save_file = f"test-{self.name}-{model.model_size}-{model.window_size}.csv"
        model.set_list_class(self.vocab)

        if not model.check_embeddings_exist(self.save_file, articlesRetriever):
            model.convert(articlesRetriever)
            model.export(self.save_file)


    def _end(self):
        raise NotImplementedError

    def __call__(self, model, articlesRetriever):

        logging.info(f"Start test {self.name}")

        tic = perf_counter()
        self._start(model, articlesRetriever)
        result = self._end(model)
        toc = perf_counter()

        logging.info(f"End test {self.name}")

        return result, toc - tic

class EmbeddingDistanceTest(Test):

    def __init__(self, vocab, thresold, name):
        Test.__init__(self, name)

        self.thresold = thresold
        self.vocab = vocab

    def _end(self, model):
        
        sim_matrix = SimilarityMatrix(self.save_file, CosineSim())
        ids, cosine_mat = sim_matrix.get_sim_matrix()

        sim_list = []

        for i, idsa in tqdm(enumerate(ids), total = len(ids), desc=f"{'listing sim matrix':30}"):
            for j, idsb in enumerate(ids):

                if i == j: continue 

                cos_val = cosine_mat[i][j]

                if cos_val >= self.thresold:
                    sim_list.append((idsa, idsb, cos_val))

        sim_list.sort(key=lambda x: x[2], reverse = True)
        return len(sim_list) // 2


class SyntacticTest(Test):

    def __init__(self, quadruple_set : Tuple[str, str, str, str], name : str):
        Test.__init__(self, name)

        self.relations : Tuple[str, str, str, str] = quadruple_set

        for relation in self.relations:
            for item in relation:
                if item not in self.vocab:
                    self.vocab.append(item)

    def _end(self, model):

        solver = Solver(self.save_file)
        top1, top3, top5, top10 = 0, 0, 0, 0

        for w1, w2, w3, w4 in tqdm(self.relations, total=len(self.relations), desc=f"{'calculating relations':30}"):

            w1_emb = model.get_embedding_of(w1).numpy()
            w2_emb = model.get_embedding_of(w2).numpy()
            w3_emb = model.get_embedding_of(w3).numpy()

            totest = w1_emb - w2_emb + w3_emb
            result = solver.get_nearest_embedding_of(torch.from_numpy(totest), 13)
            filtered_result = list(filter(lambda x: x != None, map(lambda x: x[0] if x[0] not in [w1, w2, w3] else None, result)))

            if filtered_result[0] == w4:
                top1 += 1

            if w4 in filtered_result[:3]:
                top3 += 1

            if w4 in filtered_result[:5]:
                top5 += 1

            if w4 in filtered_result[:10]:
                top10 += 1

        
        return list(map(lambda x : x / len(self.relations), (top1, top3, top5, top10)))

class SimilarityTest(Test):

    def __init__(self, pair_set : Tuple[str, str, float], pair_name):
        Test.__init__(self, pair_name)

        self.pair : Tuple[str, str, float] = pair_set

        for w1, w2, i in self.pair:
            if w1 not in self.vocab:
                self.vocab.append(w1)
            if w2 not in self.vocab:
                self.vocab.append(w2)
    
    def _end(self, model):
        sim_list = []
        i_list = []

        sim_computer = SimilarityMatrix(self.save_file, CosineSim())

        for w1, w2, i in tqdm(self.pair, total=len(self.pair), desc=f"{'calculating pair':30}"):
            sim = sim_computer.sim_between(w1, w2)

            sim_list.append(sim)
            i_list.append(i)

        result = spearmanr(sim_list, i_list)
        return (result.correlation, result.pvalue)