import gc
import logging
from typing import List
from . import *
from word_embeddings.article import *
from word_embeddings.models import * 

def all_models_gen(list_int):
    if 0 in list_int: yield BERTModel(    [], big = False,    window = 0  )    
    if 1 in list_int: yield BERTModel(    [], big = False,    window = 300)
    if 2 in list_int: yield BERTModel(    [], big = False,    window = 512)
    if 3 in list_int: yield BERTModel(    [], big = True,     window = 0  )
    if 4 in list_int: yield BERTModel(    [], big = True,     window = 300)
    if 5 in list_int: yield BERTModel(    [], big = True,     window = 512)
    if 6 in list_int: yield ROBERTAModel( [], big = False,    window = 0  )
    if 7 in list_int: yield ROBERTAModel( [], big = False,    window = 300)
    if 8 in list_int: yield ROBERTAModel( [], big = False,    window = 512)
    if 9 in list_int: yield ROBERTAModel( [], big = True,     window = 0  )
    if 10 in list_int: yield ROBERTAModel( [], big = True,     window = 300)
    if 11 in list_int: yield ROBERTAModel( [], big = True,     window = 512)
    if 12 in list_int: yield DocBERTModel( [], big = False                 )
    if 13 in list_int: yield DocBERTModel( [], big = True                  )
    if 14 in list_int: yield DocBERTAModel([], big = False                 )
    if 15 in list_int: yield DocBERTAModel([], big = True                  )   
    if 16 in list_int: yield Wiki2VecModel([]                              )
    if 17 in list_int: yield GloVEModel(   []                              )

def split_test(name):
    print()
    print("#" * 80)
    print("#", name.center(78, " "), "#", sep="")
    print("#" * 80)
    print()

import argparse

parser = argparse.ArgumentParser(description='test a word embeddings using different test')
parser.add_argument('--models', dest='models', nargs="+", type=int, help='list of the models to test :\n\
    0: bert w0\n\
    1: bert w300\n\
    2: bert w512\n\
    3: bert large w0\n\
    4: bert large w300\n\
    5: bert large w512\n\
    6: roberta w0\n\
    7: roberta w300\n\
    8: roberta w512\n\
    9: roberta large w0\n\
    10: roberta large w300\n\
    11: roberta large w512\n\
    12: docbert\n\
    13: docbert large\n\
    14: docberta\n\
    15: docberta large\n\
    16: wiki2vec\n\
    17: glove)')
parser.add_argument('--tests', dest='tests', nargs="+", type=int, help='test to execute :\n\
    0: similarity Wordsim\n\
    1: similarity Simlex\n\
    2: Syntactic GoogleAnalogy\n\
    3: EmbeddingDistance CIFAR100')
parser.add_argument('--artRetriever', dest='art_retriever', type=str, help='either to use wordnet ("wo") summary or wikipedia ("wi") summary for model that need one')

args = parser.parse_args()

if args.art_retriever == "wi" or args.art_retriever == None:
    retriever = WikipediaArticleRetriever()
elif args.art_retriever == "wo":
    retriever = WordNetArticleRetriever()

for model in all_models_gen(args.models):
    split_test(f"{model.model_size} {model.window_size}")
    TestPipeline(model, retriever, [t for i, t in enumerate(TestPipeline.list_test) if i in args.tests]).execute()
    gc.collect()
