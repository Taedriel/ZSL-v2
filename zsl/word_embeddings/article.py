from typing import List, Tuple
from os.path import exists, join

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet

import wikipedia
import warnings
import logging
import pickle 
import tqdm

wikipedia.set_rate_limiting(True)
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

__all__ = ["customArticle", "ArticleRetriever", "WikipediaArticleRetriever", "WordNetArticleRetriever", "ArticleViewer" ]

class customArticle:
    """ store a wikipedia article for further processing by models"""

    def __init__(self, index : int, title : str, realtitle : str, summary : str, ambiguous : bool):
        self.index : int = index
        self.title : str = title
        self.realtitle : str = realtitle
        self.summary :str = summary
        self.ambiguous :bool = ambiguous

class ArticleRetriever:

    """ Class in charge of retrieveing article from different sources and store them in orer
    to not re retrieve them. 
    
    Act as a proxy between wikipedia and the model. This class save all the article 
    retrieve in a dict using the name given. Further call to this retriever will 
    then load the previously saved file if it hasn't been deleted.
    """

    article_dir = "./article"

    def __init__(self, name : str = None, list_title : List[str] = []):

        self.name : str = name
        if self.name is None:
            self.name = "temp"

        self.list_title : List[str] = list_title
        self.modified : bool = False
        self._load()

    def _load(self):
        if not exists(self.get_filename()):
            self.articles_map = {}
            logging.info(f"creating file {self.get_filename()}")
        else:
            with open(self.get_filename(), "rb") as mapfile:
                self.articles_map = pickle.load(mapfile)
                assert(type(self.articles_map) == type(dict()))
            logging.info(f"loading file {self.get_filename()} with {len(self.articles_map)} articles")
    
    def set_list_vocab(self, new_name : str, list_title : List[str]):
        logging.info("changing vocab, reloading file...")
        self.list_title : List[str] = list_title
        self.name = new_name
        self._load()

    def get_filename(self) -> str:
        """ return the filename of the file where article are saved"""
        return join(WikipediaArticleRetriever.article_dir, self.name)

    def load_article(self, title : str, force_reload : bool = False) -> customArticle:
        """ retrieve an article from wikipedia. If forcce reload is specified, re check the article 
        is summary isn't present or if article not alread in the dict""" 

        if title not in self.articles_map:
            self.modified = True
            realtitle, summary, ambiguous = self._retrieve_article(title)
            self.articles_map[title] = customArticle(len(self.articles_map), title, realtitle, summary, ambiguous)

        if title in self.articles_map and self.articles_map[title].summary == None and force_reload:
            self.modified = True
            realtitle, summary, ambiguous = self._retrieve_article(title)
            self.articles_map[title].summary = summary

        return self.articles_map[title]

    def load_all_articles(self, force_reload : bool = False) -> None:
        """retrieve all article from the vocab from sources"""
        
        logging.info(f"Starting loading articles... [Force reload : {force_reload}]")
        nb_success = 0

        nb_article = len(self.list_title)
        for i, title in tqdm(enumerate(self.list_title), total=nb_article, desc=f"{'loading articles':30}"):
            self.load_article(title, force_reload)

            if self.articles_map[title].summary is not None: 
                nb_success += 1

        logging.info(f"Finished loading {nb_success} article(s) / {nb_article} ({round(nb_success / nb_article * 100, 1)}%)!")
        return self.modified

    def __call__(self, force_reload : bool = False) -> None:
        return self.load_all_articles(force_reload)

    def _retrieve_article(self, title : str, closed_list : List[str]) -> Tuple[str, str, bool]:
        raise NotImplementedError

    def get_article(self, title) -> customArticle:
        """return the article if it's present, else, try to retrieve it"""

        if title not in self.articles_map:
            self.load_article(title)

        return self.articles_map[title]
        
    def save(self):
        """save the articles in a binary format using pickle"""
        logging.info(f"saving the file {self.get_filename()}")
        with open(self.get_filename(), "wb") as mapfile:
            pickle.dump(self.articles_map, mapfile)

class WikipediaArticleRetriever(ArticleRetriever):

    def __init__(self, name: str = None, list_title: List[str] = []):
        ArticleRetriever.__init__(self, name, list_title)

    def get_filename(self) -> str:
        """ return the filename of the file where article are saved"""
        return join(WikipediaArticleRetriever.article_dir, "Wiki-" + self.name)

    def _retrieve_article(self, title : str, closed_list : List = []) -> Tuple[str, str, bool]: 
        closed_list.append(title)
        try:
            article = wikipedia.page(title, auto_suggest=False, redirect=True)
            return (article.title, article.summary, False)

        except wikipedia.PageError as e:
            search_result = wikipedia.search(title, suggestion = False)

            logging.warning(f"{title} misspelled or article missing. Best find is {search_result[0]}")
            if search_result[0] is not None and search_result[0] not in closed_list:            
                return self._retrieve_article(search_result[0], closed_list)  
            else: return (None, None, None)

        except wikipedia.DisambiguationError as e:
            logging.warning(f"{title} is ambiguous, fallback on {e.options[0]}")
            return (None, None, None)
            # if e.options[0] is not None and e.options[0] not in closed_list:
            #     res = self._retrieve_article(e.options[0], closed_list)
            #     return (res[0], res[1], True)
        return (None, None, None)

class WordNetArticleRetriever(ArticleRetriever):

    def __init__(self, name: str = None, list_title: List[str] = []):
        super().__init__(name, list_title)

    def get_filename(self) -> str:
        """ return the filename of the file where article are saved"""
        return join(WikipediaArticleRetriever.article_dir, "Word-" + self.name)

    def _retrieve_article(self, title: str, closed_list : List = []) -> Tuple[str, str, bool]:
        result = wordnet.synsets(title)
        if len(result) > 0:
            return (title, result[0].definition(), True)

        return (None, None, None)

class ArticleViewer():

    def __init__(self, filename):
        self.name = filename

        if not exists(self.name):
            raise FileNotFoundError()
        else:
            with open(self.name, "rb") as mapfile:
                self.articles_map = pickle.load(mapfile)

    def get(self, title):
        return self.articles_map[title]

    def get_all_articles(self):
        return self.articles_map.keys()
