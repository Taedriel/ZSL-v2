import gluonnlp as nlp

from .test import *

import logging
log = logging.getLogger(__name__)

cifar100 = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", \
            "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "computer_keyboard", "couch", "crab", "crocodile", "cup", \
            "dinosaur", 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'lamp', 'lawn_mower', 'leopard', 'lion', \
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', \
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',\
            'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', \
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

__all__ = ["TestPipeline"]

class TestPipeline():

    list_test = [ SimilarityTest(nlp.data.WordSim353('all'), "Wordsim353"), 
                  SimilarityTest(nlp.data.SimLex999('all') , "SimLex999") ,
                  SyntacticTest(nlp.data.GoogleAnalogyTestSet(), "GoogleAnalogy"),
                  EmbeddingDistanceTest(cifar100, 0.80, "Imagenet")
                ]

    def __init__(self, model, articleRetriever, list_test = None):
        self.model = model
        self.articleRetriver = articleRetriever

        if list_test == None:
            log.info(f"no test provided, fallback on all tests available")
            self.list_test = TestPipeline.list_test
        else:
            self.list_test = list_test

    def execute(self):

        for i, test in enumerate(self.list_test):

            print(f"Test {i} : {test.name}".center(80, "="))
            res, time_elapsed = test(self.model, self.articleRetriver)
            print(f"\n{res}")
            print(f"{round(time_elapsed, 2)} sec.".center(80, "="))