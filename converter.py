import torch
from transformers import BertTokenizer, BertModel

import logging
logging.basicConfig(level = logging.INFO)

from typing import List


class WordToVecteur:

    def __init__(self, list_tags : List[str] = []):
        self.list_tags = list_tags

    def export(self, filename):
        return NotImplementedError

    def importTagList(self, filename):
        return NotImplementedError

    def convert(self):
        return NotImplementedError


class BERTModel(WordToVecteur):

    def __init__(self, list_tag : List[str] = [], big: bool = False):
        super(BERTModel, self).__init__(list_tag)
        self.model_size = "bert-large-uncased" if big else "bert-base-uncased"
        self.embeddings = []
        self.cosine_sim_matrix = None

        # self.tokenizer = BertTokenizer.from_pretrained(self.model_size, padding=True, truncation=True,)
        self.model = BertModel.from_pretrained(self.model_size, output_hidden_states = True)

        self.merging_strategy = Sum4LastLayers()

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

    def import_tag_list(self, filename):
        self.embeddings.clear()
        self.cosine_sim_matrix = None

        try:
            f = open(filename, "r")
        except OSError:
            return OSError("Could not open file")

        with f:
            data = f.read().split("\n")
            for item in data:
                if item not in self.list_tags and str.strip(item) != "":
                    self.list_tags.append(item)
            print(self.list_tags)
            logging.info(f"Import finished : {len(self.list_tags)} elements imported.")
            

    def convert(self):
        """ convert all word in their embeddings"""

        logging.info("Starting converting tokens...")
        nb_token = len(self.list_tags)
        current_percent = 0

        for i, tag in enumerate(self.list_tags):
            
            percent_completion = (i / nb_token) * 100
            if percent_completion >= current_percent + 10:
                nearest_percent = (percent_completion // 10) * 10
                logging.info(f"{nearest_percent}% completed")
                current_percent = nearest_percent
            

            encoded = tz.encode_plus(
                text=tag,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 255,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )

            # Get the input IDs and attention mask in tensor format
            input_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']


            inputs = self.tokenizer(tag, return_tensors = "pt")

            with torch.no_grad():
                outputs = self.model(input_ids, attn_mask)

            hidden_states = outputs[2]

            # log.info(f"[{i}]","Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
            # log.info(f"[{i}]","Number of batches:", len(hidden_states[0]))
            # logging.info(f"[{i}] Number of tokens: {len(hidden_states[0][0]) - 2}")
            # log.info(f"[{i}]","Number of hidden units:", len(hidden_states[0][0][0]))

            # [# layers, # batches, # tokens, # features] ==> [# tokens, # layers, # features]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)

            # apply different strategy to summarize word embeddings
            tokenized_text = self.tokenizer.tokenize(tag)
            for i, token in enumerate(tokenized_text):
                embed = self.merging_strategy.merge(token_embeddings[i+1])
                self.embeddings.append((token, embed))

    def compute_sim(self):
        """ compute cosine similarity between all vectors """
        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")

        logging.info("Computing cosine similarity, this could take some time...")

        n_tokens = len(self.embeddings)
        self.cosine_sim_matrix = [[1 for j in range(n_tokens)] for i in range(n_tokens)]

        for j, vector in enumerate(self.embeddings):

            for i, other_vector in enumerate(self.embeddings):

                if i == j:
                    continue

                
                cos = torch.nn.CosineSimilarity(dim=0)
                similarity = cos(vector[1], other_vector[1])

                self.cosine_sim_matrix[i][j] = similarity
                self.cosine_sim_matrix[j][i] = similarity

    def export_sim_matrix(self, filename):
        if self.cosine_sim_matrix == None:
            self.compute_sim()
        
        try:
            f = open(filename, "w")
        except OSError:
            raise OSError("Could not open file")

        with f:
            print("/", ",".join([tag[0] for tag in self.embeddings]), sep = ",", file = f)

            for j, tag_y in enumerate(self.embeddings):
                print(tag_y[0], ",".join( [str(round(float(self.cosine_sim_matrix[j][i]), 3)) for i in range(len(self.embeddings))]), sep = ",", file = f)

    def sim_between(self, token1, token2):
        if self.cosine_sim_matrix is None:
            self.compute_co_sim()

        index1 = [i for i, v in enumerate(self.embeddings) if v[0] == token1][0]
        index2 = [i for i, v in enumerate(self.embeddings) if v[0] == token2][0]

        return self.cosine_sim_matrix[index1][index2]

    def get_embedding_of(self, token):
        res = [v for v in self.embeddings if v[0] == token]
        if len(res) == 0:
            raise Exception("no such token")
        
        return res[0]

    def get_nearest_embedding_of(self, embedding, nb = 10):

        if nb > len(self.embeddings):
            raise Exception("nb too high, not enough token")

        nearest = []
        for e in self.embeddings:

            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(embedding, e[1])

            nearest.append((e[0], similarity))
        
        nearest.sort(key = lambda tup : tup[1])
        return nearest[-nb:]

class Sum4LastLayers:

    def merge(self, vector):
        return torch.sum(vector[-4:], dim = 0)


if __name__ == "__main__":

    model = BERTModel(["cat", "dog", "snake", "mouse"])

    model.importTagList("tagList.csv")
    # model.convert()
    model.export("test.csv")