import torch

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
