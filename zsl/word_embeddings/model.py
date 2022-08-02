import torch
import json

from typing import List
from .article import ArticleRetriever
from .embedding_loader import EmbeddingsLoader
from .dowloader import Downloader
from ..misc import dict2csv

class WordToVector:

    def __init__(self, list_tags : List[str] = []):
        self.list_tags = list_tags
        self.embeddings = {}

    def set_list_class(self, list_class : List[str]):
        self.list_tags = list_class
        self.reset_embeddings()

    def check_embeddings_exist(self, filename : str, article_ret : ArticleRetriever):
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

    def convert(self, article_ret : WikipediaArticleRetriever):
        raise NotImplementedError

    def reset_embeddings(self):
        self.embeddings.clear()

    def get_embedding_of(self, token):
        if token not in self.embeddings:
            raise Exception(f"no such token {token}")
        
        return self.embeddings[token]

    def get_class_list(self):
        return self.embeddings.keys()

    def export(self, filename):
        """export all the embeddings in filename under a .csv format.
           Raise exception if embeddings hasn't been calculed yet."""

        if len(self.embeddings) == 0:
            raise Exception("Tags not converted yet !")
        
        dict2csv(filename, self.embeddings)

class FixedEmbedding(WordToVector):

    def __init__(self, base_addr : str, file_zipname : str):
        self.downloader = Downloader(base_addr, file_zipname)
        self.downloader.download()

    def check_embeddings_exist(self, filename : str, article_ret : WikipediaArticleRetriever):
        return False

    def _one_turn(self, resolve_dict = {}):
        print("here")
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