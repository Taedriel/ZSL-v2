""" Word Embeddings module - subpart of the zsl package 

This package contains multiple tools to work with word embeddings. It contains tools for:
    * generate word embeddings using multiple models
    * test word embeddings using multiple word embeddings test
    * download and store article's summary from source like Wikipedia or Word2Vec

Examples:
    You can generate embeddings for any vocabulary file you want using the command::

        $ python -m word_embeddings --vocab path/to/vocab/file

    see help for more details

    You can also test models using the command::

        $ python -m word_embedding.tests --models 0 --tests 0 1 2 3

    see help for more details

Differents model are: 
    * BERT
    * ROBERTA
    * DocBERT, a BERT model that can be fed an entire document
    * DocBERTA, a ROBERTA model that can be fed an entire document
    * pre-trained GloVe
    * pre-trained Wikipedia2Vec

Differents test for word embeddings are:
    * Sintactic Test
    * Similarity Test
    * Distance Test

More information for each model and test in their own documentation

"""

from .article import *
from .bert_strategy import *
from .dowloader import *
from .embedding_loader import *
from .model import *

from .models import *
from .tests import *

__all__ = ["customArticle", "ArticleRetriever", "WikipediaArticleRetriever", "WordNetArticleRetriever", \
            "ArticleViewer","BERTMergeStrategy", "Sum4LastLayers", "Concat4LastLayer", "SimilarityStrategy", \
            "CosineSim", "EuclidianDistSim", "ManhattanDistSim", "Downloader", "EmbeddingsLoader", "WordToVector", \
            "FixedEmbedding", "TestPipeline", "EmbeddingDistanceTest", "SyntacticTest", "SimilarityTest", \
            "BERTModel", "DocBERTModel", "DocBERTAModel", "GloVEModel", "ROBERTAModel", "Wiki2VecModel"]