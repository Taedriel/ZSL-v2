import torch
import json

from .article import ArticleRetriever
from .embedding_loader import EmbeddingsLoader
from .dowloader import Downloader
from typing import List
from zsl.misc import dict2csv

__all__ = ["WordToVector", "FixedEmbedding"]

class WordToVector:
    """Base class for others model.
    """

    def __init__(self, list_tags : List[str] = []):
        """Initialize a model with an empy list of vocabulary

        Args:
            list_tags (List[str], optional): list of class to transform. Defaults to [].
        """
        self.list_tags = list_tags
        self.embeddings = {}

    def set_list_class(self, list_class : List[str]):
        """change the list of class

        Args:
            list_class (List[str]): the new list of class
        """
        self.list_tags = list_class
        self.reset_embeddings()

    def check_embeddings_exist(self, filename : str, article_ret : ArticleRetriever):
        """check if a file containing these embeddings already exist. 
        
        If a file with the same name,containing the first same embedding, then we assume that this file contain current embeddings 
        and we can skip the converting step

        Args:
            filename (str): filename to check
            article_ret (ArticleRetriever): the article retriver to use

        Returns:
            bool: whether a file with same embedding exist
        """
        temp_tags_list = self.list_tags
        first_tag = self.list_tags[0]
        self.list_tags = [first_tag]

        self.convert(article_ret)
        try:
            loader = EmbeddingsLoader(filename)
        except OSError:
            self.list_tags = temp_tags_list
            return False

        first_embedding = self.embeddings[first_tag]
        to_compare = loader.embeddings[first_tag]

        intersect = set(temp_tags_list) & set(loader.embeddings.keys())

        self.list_tags = temp_tags_list
        if torch.equal(first_embedding, to_compare) and len(intersect) == len(temp_tags_list):
            return True

        return False

    def convert(self, article_ret : ArticleRetriever):
        """convert all class in embeddings

        Args:
            article_ret (ArticleRetriever): the Article retriever to use 

        Raises:
            NotImplementedError: if no convert method exist for a given model
        """
        raise NotImplementedError

    def reset_embeddings(self):
        """erase all the embeddings
        """
        self.embeddings.clear()

    def get_embedding_of(self, token : str) -> List[float]:
        """return the embedding of a class if this class exist

        Args:
            token (str): the class to get the embedding from

        Raises:
            Exception: if no such class exist

        Returns:
            List[float]: the embedding of the class
        """
        if token not in self.embeddings:
            raise Exception(f"no such token {token}")
        
        return self.embeddings[token]

    def get_class_list(self) -> List[str]:
        """return the list of all class

        Returns:
            List[str]: all the class that have been or will be converted.
        """
        return self.embeddings.keys()

    def export(self, filename):
        """export all the embeddings in filename under a .csv format.
           Raise exception if embeddings hasn't been calculed yet."""

        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")
        
        dict2csv(filename, self.embeddings)

class FixedEmbedding(WordToVector):
    """In certain situation, we don't need to generate embedding as they're already generated, and stored in file.
    This class allow to use this kind of embedding by first downloading all the pre trained embedding, and by adding 
    a human verification on the convert part, in case of name mismatch between the model used and the class input.

    This is a base class
    """

    def __init__(self, base_addr : str, file_zipname : str):
        self.downloader = Downloader(base_addr, file_zipname)
        self.downloader.download()

    def check_embeddings_exist(self, filename : str, article_ret : ArticleRetriever):
        return False

    def _one_turn(self, resolve_dict = {}):
        raise NotImplementedError

    def convert(self, ar, stop_at_first = False):
        resolve_filename = f"./temp/{ar.name[:-4]}_resolve.json"
        resolve = {}

        while True: 
            unk_list = self._one_turn(resolve)
        
            if unk_list is None or len(unk_list) == 0 or stop_at_first:
                break

            print(len(unk_list), "items haven't been found, resolve mode.")

            with open(resolve_filename, 'w') as f:
                resolve_dict = {word: "" for word in unk_list}
                json.dump(resolve_dict, f, indent = 4)

            input("press enter to resume resolve")

            resolve = {}
            with open(resolve_filename, 'r') as f:
                resolve = json.load(f)
                assert(type(resolve) == type(dict()))