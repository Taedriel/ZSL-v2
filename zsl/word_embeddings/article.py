from xxlimited import Str
import wikipedia
import warnings
import pickle 
import nltk

nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet
from typing import List, Tuple
from os.path import exists, join
from tqdm import tqdm

import logging
log = logging.getLogger(__name__)

wikipedia.set_rate_limiting(True)
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

__all__ = ["customArticle", "ArticleRetriever", "WikipediaArticleRetriever", "WordNetArticleRetriever", "ArticleViewer" ]

class customArticle:
    """ store the summary of a wikipedia article for further processing by models"""

    def __init__(self, index : int, title : str, realtitle : str, summary : str, ambiguous : bool):
        self.index : int = index
        self.title : str = title
        self.realtitle : str = realtitle
        self.summary :str = summary
        self.ambiguous :bool = ambiguous

class ArticleRetriever:

    """ Class in charge of retrieveing article from different sources and store them in orer
    to not retrieve them a second time. 
    
    Act as a proxy between wikipedia and the model. This class save all the article 
    retrieve in a dictionnary using the name given. Further call to this retriever will 
    then load the previously saved file if it hasn't been deleted.
    """

    def __init__(self, name : str = None, list_title : List[str] = []):

        self.name : str = name
        if self.name is None:
            self.name = "temp"

        self.list_title : List[str] = list_title
        self.modified : bool = False
        self.__load()

    def __load(self):
        """
        load the file if it exist, otherwise, create it
        """
        if not exists(self.get_filename()):
            self.articles_map = {}
            log.info(f"creating file {self.get_filename()}")
        else:
            with open(self.get_filename(), "rb") as mapfile:
                self.articles_map = pickle.load(mapfile)
                assert(type(self.articles_map) == type(dict()))
            log.info(f"loading file {self.get_filename()} with {len(self.articles_map)} articles")
    
    def set_list_vocab(self, new_name : str, list_title : List[str]) -> None:
        """set the list of vocab to be retrieved.

        each time this method is called, proceed then to load automatically all the newly added class

        Args:
            new_name (str): new name for the file where the article while be saved
            list_title (List[str]): list of vocabulary to retrieve
        """
        log.info("changing vocab, reloading file...")
        self.list_title : List[str] = list_title
        self.name = new_name
        self.__load()

    def get_filename(self) -> str:
        """ return the filename of the file where articles are saved"""
        # return join(WikipediaArticleRetriever.article_dir, self.name)
        return self.name

    def load_article(self, title : str, force_reload : bool = False) -> customArticle:
        """ retrieve an article from the source. If force reload is specified, download the article
        even if the article has already been download once
        """ 

        if title not in self.articles_map:
            self.modified = True
            realtitle, summary, ambiguous = self.__retrieve_article(title)
            self.articles_map[title] = customArticle(len(self.articles_map), title, realtitle, summary, ambiguous)

        if title in self.articles_map and self.articles_map[title].summary == None and force_reload:
            self.modified = True
            realtitle, summary, ambiguous = self.__retrieve_article(title)
            self.articles_map[title].summary = summary

        return self.articles_map[title]

    def load_all_articles(self, force_reload : bool = False) -> None:
        """retrieve the summary of all article from the vocab, from the source

            Args:
                force_reload (bool, optional): whether to try retrieving all articles, even already retrieved one. Defaults to False.

            Returns:
                bool : whether any change has been made to the article compared to last execution. 
                If True, the new file may need to be saved using the save method
        """
        
        log.info(f"Starting loading articles... [Force reload : {force_reload}]")
        nb_success = 0

        nb_article = len(self.list_title)
        for i, title in tqdm(enumerate(self.list_title), total=nb_article, desc=f"{'loading articles':30}", ncols=80):
            self.load_article(title, force_reload)

            if self.articles_map[title].summary is not None: 
                nb_success += 1

        log.info(f"Finished loading {nb_success} article(s) / {nb_article} ({round(nb_success / nb_article * 100, 1)}%)!")
        return self.modified

    def __call__(self, force_reload : bool = False) -> None:
        """see load_all_articles
        """
        return self.load_all_articles(force_reload)

    def __retrieve_article(self, title : str, closed_list : List[str]) -> Tuple[str, str, bool]:
        raise NotImplementedError

    def get_article(self, title) -> customArticle:
        """return the article if it's present, else, try to retrieve it"""

        if title not in self.articles_map:
            self.load_article(title)

        return self.articles_map[title]
        
    def save(self) -> None:
        """save the articles in a binary format using pickle
        
            Summary are saved in a dictionnary with the key being the named of the class
        """
        log.info(f"saving the file {self.get_filename()}")
        with open(self.get_filename(), "wb") as mapfile:
            pickle.dump(self.articles_map, mapfile)

class WikipediaArticleRetriever(ArticleRetriever):
    """use Wikipedia as a source for summary
    """

    def __init__(self, name: str = None, list_title: List[str] = []):
        ArticleRetriever.__init__(self, name, list_title)

    def get_filename(self) -> str:
        """ return the filename of the file where article are saved"""
        from os import sep
        path = self.name.split(sep)
        return join(*path[:-1], "Wiki-" + path[-1])

    def __retrieve_article(self, title : str, closed_list : List = []) -> Tuple[str, str, bool]: 
        closed_list.append(title)
        try:
            article = wikipedia.page(title, auto_suggest=False, redirect=True)
            return (article.title, article.summary, False)

        except wikipedia.PageError as e:
            search_result = wikipedia.search(title, suggestion = False)

            log.warning(f"{title} misspelled or article missing. Best find is {search_result[0]}")
            if search_result[0] is not None and search_result[0] not in closed_list:            
                return self.__retrieve_article(search_result[0], closed_list)  
            else: return (None, None, None)

        except wikipedia.DisambiguationError as e:
            log.warning(f"{title} is ambiguous, skipping...")
            return (None, None, None)
            log.warning(f"{title} is ambiguous, fallback on {e.options[0]}")
            # if e.options[0] is not None and e.options[0] not in closed_list:
            #     res = self._retrieve_article(e.options[0], closed_list)
            #     return (res[0], res[1], True)
        return (None, None, None)

class WordNetArticleRetriever(ArticleRetriever):
    """use WordNet as source for summary
    """

    def __init__(self, name: str = None, list_title: List[str] = []):
        super().__init__(name, list_title)

    def get_filename(self) -> str:
        """ return the filename of the file where article are saved"""
        from os import sep
        path = self.name.split(sep)
        return join(*path[:-1], "Word-" + path[-1])

    def __retrieve_article(self, title: str, closed_list : List = []) -> Tuple[str, str, bool]:
        result = wordnet.synsets(title)
        if len(result) > 0:
            return (title, result[0].definition(), True)

        return (None, None, None)

class ArticleViewer():
    """Load a previously saved article file to inspect its content.
    """

    def __init__(self, filename : str):
        """load an article file

        Args:
            filename (str): path to the article file to load

        Raises:
            FileNotFoundError: if the path is incorrect
        """
        self.name = filename

        if not exists(self.name):
            raise FileNotFoundError()
        else:
            with open(self.name, "rb") as mapfile:
                self.articles_map = pickle.load(mapfile)

    def get(self, title : str) -> str:
        """try retrieveing an summary from the file

        Args:
            title (str): the name of the class to retrieve

        Raises:
            KeyError : if the class is not in the article file

        Returns:
            str: the summary of the class
        """
        return self.articles_map[title]

    def get_all_articles(self) -> List[str]:
        """return all class saved in the file

        Returns:
            List[str]: the list of class name saved
        """
        return self.articles_map.keys()
