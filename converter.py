import torch
from transformers import BertTokenizer, BertModel

import logging
logging.basicConfig(level = logging.INFO)

from typing import List


class WordToVecteur:

    def __init__(self, listTags : List[str] = []):
        self.listTags = listTags

    def export(self, filename):
        return NotImplementedError

    def importTagList(self, filename):
        return NotImplementedError

    def convert(self):
        return NotImplementedError


class BERTModel(WordToVecteur):

    def __init__(self, listTag : List[str] = [], big: bool = False):
        super(BERTModel, self).__init__(listTag)
        self.modelSize = "bert-large-uncased" if big else "bert-base-uncased"
        self.embeddings = []
        self.cosineSimMatrix = None

        self.tokenizer = BertTokenizer.from_pretrained(self.modelSize, padding=True, truncation=True,)
        self.model = BertModel.from_pretrained(self.modelSize, output_hidden_states = True)

        self.mergingStrategy = Sum4LastLayers()

        self.model.eval()


    def export(self, filename):
        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")
        
        try:
            f = open(filename, "w")
        except OSError:
            raise OSError("Could not open file")

        with f:
            for embedding in self.embeddings:
                line = ",".join(list(map(str, map(float, embedding[1]))))
                print(embedding[0], ",", line, sep="", file=f)

    def importTagList(self, filename):
        try:
            f = open(filename, "r")
        except OSError:
            return OSError("Could not open file")

        with f:
            data = f.read().split("\n")
            for item in data:
                if item not in self.listTags and str.strip(item) != "":
                    self.listTags.append(item)
            print(self.listTags)
            logging.info(f"Import finished : {len(self.listTags)} elements imported.")
            

    def convert(self):
        for i, tag in enumerate(self.listTags):

            inputs = self.tokenizer(tag, return_tensors = "pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs[2]

            # log.info(f"[{i}]","Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
            # log.info(f"[{i}]","Number of batches:", len(hidden_states[0]))
            # log.info(f"[{i}]","Number of tokens:", len(hidden_states[0][0]) - 2)
            # log.info(f"[{i}]","Number of hidden units:", len(hidden_states[0][0][0]))

            # [# layers, # batches, # tokens, # features] ==> [# tokens, # layers, # features]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)

            # apply different strategy to summarize word embeddings
            tokenized_text = self.tokenizer.tokenize(tag)
            for i, token in enumerate(tokenized_text):
                embed = self.mergingStrategy.merge(token_embeddings[i+1])
                self.embeddings.append((token, embed))

    def computeCoSim(self):
        """ compute cosine similarity between all vectors """
        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")

        logging.info("Computing cosine similarity, this could take some time...")

        nTokens = len(self.embeddings)
        self.cosineSimMatrix = [[0 for j in range(nTokens)] for i in range(nTokens)]

        for j, vector in enumerate(self.embeddings):

            for i, otherVector in enumerate(self.embeddings):

                if i == j:
                    continue
                
                cos = torch.nn.CosineSimilarity(dim=0)
                similarity = cos(vector[1], otherVector[1])

                self.cosineSimMatrix[i][j] = similarity
                self.cosineSimMatrix[j][i] = similarity

    def simBetween(self, token1, token2):
        if self.cosineSimMatrix is None:
            self.computeCoSim()

        index1 = [i for i, v in enumerate(self.embeddings) if v[0] == token1][0]
        index2 = [i for i, v in enumerate(self.embeddings) if v[0] == token2][0]

        return self.cosineSimMatrix[index1][index2]

    def getEmbeddingOf(self, token):
        res = [v for v in self.embeddings if v[0] == token]
        if len(res) == 0:
            raise Exception("no such token")
        
        return res[0]

    def getNearestEmbeddingOf(self, token, nb = 10):

        if nb > len(self.embeddings):
            raise Exception("nb too high, not enough token")

        embedding = self.getEmbeddingOf(token)

        nearest = []
        for e in self.embeddings:

            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(embedding, e[1])

            nearest.append((e[0], similarity))
        
        nearest.sort(key = lambda tup : tup[1])
        return nearest[:nb]

class Sum4LastLayers:

    def merge(self, vector):
        return torch.sum(vector[-4:], dim = 0)


if __name__ == "__main__":

    model = BERTModel(["cat", "dog", "snake", "mouse"])

    model.importTagList("tagList.csv")
    # model.convert()
    model.export("test.csv")