import sys
import os

from zsl.fsl_classification.sub_pipeline import get_imageNet_negative_images, clean_images, evaluate, get_metainfo, download_google_images, train_model
sys.path.append("./zsl")

from zsl.visual_to_txt_mapping.mapping_generator import generate_textual_mapping

from typing import List
from torch import Tensor
from Orange.data import Table

import zsl

runtime = {}

def check_file_presence(filename : str) -> str or None:
    from os.path import exists, join

    abs_path = join(os.getcwd(), filename)
    
    if not exists(abs_path):
        raise FileNotFoundError(abs_path)
    else:
        return abs_path
            

def image_to_text_embedding(image_path : str) -> List[float] or Tensor:
    return generate_textual_mapping(image_path, "zsl/visual_to_txt_mapping/model/mapping_model.model")

def text_embedding_to_classes(embedding : List[float or Tensor]) -> List[str]:
    """perform hierarchical clustering on the embedding and return the list of classes predicted

    Args:
        embedding (List[float or Tensor]): the list of embedding that form prior knowledge

    Returns:
        List[str]: a list of plausible classes
    """

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

def preprocess():
    """load ressource files and prepare the runtime environment"""
    os.makedirs("./zsl/fsl_classification/pipeline/images/", exist_ok=True)

    generic_table = Table(check_file_presence("ressources/embeddings/general-mapping-average.csv"))
    supp_info_table = Table(check_file_presence("ressources/class_map_imagenet.csv"))

    runtime["prior_knowledge_table"] = zsl.WeSeDa.left_join(generic_table, supp_info_table)


def run_pipeline(image_path : str, intermediate_result = False) -> str:
    """run the pipeline on the image given in parameter

    Args:
        image_path (str): the path to the image to classify
        intermediate_result (bool, optional): whether to return intermediate result. Defaults to False.

    Returns:
        str: a class predicted
    """

    text_embedding = image_to_text_embedding(image_path)

    plausible_classes = text_embedding_to_classes(text_embedding)

    prediction = classes_to_prediction(image_path, plausible_classes)
    
    if intermediate_result:
        return text_embedding, plausible_classes, prediction
    else:
        return prediction



if __name__ == "__main__":
    preprocess()
    text_embedding, plausible_classes, prediction = run_pipeline("examples/002.png", intermediate_result = True)
    print(text_embedding, plausible_classes, prediction)
