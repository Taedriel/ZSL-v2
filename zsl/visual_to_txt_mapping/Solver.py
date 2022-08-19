from .imports import *

class EmbeddingsLoader:

    def __init__(self, filename : str):

        self.file = filename
        self.embeddings = {}

        self._load_file()

    def _load_file(self):
        try:
            with open(self.file, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.split(",")
                self.embeddings[data[0]] = torch.FloatTensor(list(map(float, data[1:])))

        except IOError as e:
            raise IOError(f"No file {self.file}")

class SimilarityCompute(EmbeddingsLoader):

    def __init__(self, embeddings):
        super(SimilarityCompute, self).__init__(embeddings)


    def compute_sim(self):
        """ compute cosine similarity between all vectors """
        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")

        logging.info("Computing cosine similarity, this could take some time...")

        if self.cosine_sim_matrix is None:
            n_tokens = len(self.embeddings)
            self.cosine_sim_matrix = [[1 for j in range(n_tokens)] for i in range(n_tokens)]

        for j, vector in tqdm(enumerate(self.embeddings), total = len(self.embeddings)):

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
        index1, v1 = [(i, v[1]) for i, v in enumerate(self.embeddings) if v[0] == token1][0]
        index2, v2 = [(i, v[1]) for i, v in enumerate(self.embeddings) if v[0] == token2][0]

        if self.cosine_sim_matrix is None:
            n_tokens = len(self.embeddings)
            self.cosine_sim_matrix = [[1 for j in range(n_tokens)] for i in range(n_tokens)]

        if self.cosine_sim_matrix[index1][index2] == 0 or self.cosine_sim_matrix[index2][index1]:
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(v1, v2)

            self.cosine_sim_matrix[index1][index2] = similarity
            self.cosine_sim_matrix[index2][index1] = similarity

        return self.cosine_sim_matrix[index1][index2]

class Solver(EmbeddingsLoader):

    def __init__(self, embeddings, nb_predictions):
        super(Solver, self).__init__(embeddings)
        self.nb = nb_predictions

    def get_nearest_embedding_of(self, embedding):

        if self.nb > len(self.embeddings):
            raise Exception("nb too high, not enough token")

        nearest = []
        for tag, e in self.embeddings.items():

            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(embedding, e)

            nearest.append((tag, similarity))
        
        nearest.sort(key = lambda tup : tup[1])
        return nearest[-1:-self.nb-1:-1]