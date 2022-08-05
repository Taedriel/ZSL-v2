import sys

from zsl.fsl_classification.sub_pipeline import cleanImages, evaluate, get_metainfo, downloadGoogleImages, train_model
sys.path.append("./zsl")

from typing import List
from torch import Tensor
#from Orange.data import Table

import zsl


runtime = {}

def check_file_presence():
    from os.path import exists

    file_to_check = []

    for file in file_to_check:
        if not exists(file):
            raise FileNotFoundError(file)

def image_to_text_embedding(image_path : str) -> List[float] or Tensor:
    return list(range(300))

def text_embedding_to_classes(embedding : List[float or Tensor]) -> List[str]:

    embedding = zsl.HiCA(embedding, runtime["prior_knowledge_table"], runtime["superclass_embeddings"]) \
        .solve(lambda x : 0.25 + 0.05 * x, lambda x : 0.1 + 0.05 * x)

    return embedding

def classes_to_prediction(image_path : str, plausible_classes : List[str]) -> List[str]:

    downloadGoogleImages(plausible_classes, reset=True)
    PATH_DATA, conversion_type, _ = get_metainfo(CUB=False, IMAGES=True, OMNIGLOT=False)
    cleanImages(PATH_DATA, plausible_classes, conversion_type)
    supportSet = train_model(PATH_DATA, plausible_classes)
    predicted_class = evaluate(supportSet, plausible_classes)

    return [predicted_class]

def run_pipeline(image_path : str, intermediate_result = False):

    #text_embedding = image_to_text_embedding(image_path)

    #plausible_classes = text_embedding_to_classes(text_embedding)

    return classes_to_prediction(image_path, ["Africanleopard", "Ape"])

def preprocess():
    pass

def generate_embeddings():
    pass




if __name__ == "__main__":
    check_file_presence()
    preprocess()
    result, _ = run_pipeline("examples/002.png", intermediate_result = False)