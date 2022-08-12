import sys
import os

from zsl.fsl_classification.sub_pipeline import get_imageNet_negative_images, clean_images, evaluate, get_metainfo, download_google_images, train_model
sys.path.append("./zsl")

from typing import List
from torch import Tensor
from Orange.data import Table

import zsl


runtime = {}

def check_file_presence():
    from os.path import exists

    file_to_check = [
        "./ressources/Ayoub-average.csv",
        "./ressources/class_map_imagenet.csv",
        "./ressources/custom-wikipedia2vec-300_superclass.csv"
    ]

    for file in file_to_check:
        if not exists(file):
            raise FileNotFoundError(file)

def image_to_text_embedding(image_path : str) -> List[float] or Tensor:
    return list(range(300))

def text_embedding_to_classes(embedding : List[float or Tensor]) -> List[str]:

    embedding = zsl.WeSeDa(embedding, runtime["prior_knowledge_table"], runtime["superclass_embeddings"]) \
        .solve(lambda x : None, lambda x : 0.1 + 0.05 * x)

    plausible = []

    for dic in embedding:
        if type(dic) == type(dict()):
            plausible.append(dic["cluster_name"])

    return plausible


def classes_to_prediction(image_path : str, plausible_classes : List[str]) -> List[str]:
    """
    predict the label of the unknown image

    Parameters
    ----------
    image_path :
        the path to the query
    plausible_classes :
        the list of hypohtetical classes the image could be

    Return
    ------
    the label of the image
    """

    get_imageNet_negative_images(6, 10)
    download_google_images(plausible_classes, reset=False)
    PATH_DATA, conversion_type, _ = get_metainfo(CUB=False, IMAGES=True, OMNIGLOT=False)
    clean_images(PATH_DATA, plausible_classes, conversion_type) # beware that it will clean omniglot or cub if true
    supportSet = train_model(PATH_DATA, plausible_classes, conversion_type)
    predicted_class = evaluate(zsl.fsl_classification.HEAD+"pipeline/unknown image/", supportSet, plausible_classes, conversion_type)

    return [predicted_class]

def run_pipeline(image_path : str, intermediate_result = False):

    text_embedding = image_to_text_embedding(image_path)

    plausible_classes = text_embedding_to_classes(text_embedding)

    prediction = classes_to_prediction(image_path, plausible_classes)
    
    if intermediate_result:
        return text_embedding, plausible_classes, prediction
    else:
        return prediction


def preprocess():

    os.makedirs("./zsl/fsl_classification/pipeline/images/", exist_ok=True)

    generic_table = Table("./ressources/Ayoub-average.csv")
    supp_info_table = Table("./ressources/class_map_imagenet.csv")

    runtime["prior_knowledge_table"] = zsl.WeSeDa.left_join(generic_table, supp_info_table)
    runtime["superclass_embeddings"] = Table("./ressources/custom-wikipedia2vec-300_superclass.csv")


if __name__ == "__main__":
    check_file_presence()
    preprocess()
    text_embedding, plausible_classes, prediction = run_pipeline("examples/002.png", intermediate_result = True)
    print(text_embedding, plausible_classes, prediction)
