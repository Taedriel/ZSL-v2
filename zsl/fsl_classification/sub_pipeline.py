from os import listdir, makedirs, system
from shutil import rmtree
from random import randint
from sre_constants import ASSERT
from turtle import down

from .constants import *
from .utils import *
from .utils_retrieval import get_classes_images_URLLIB
from .siamese_network import *
from .classes_for_model import *
from .classes_for_dataset import CleaningSetProvider
from .fsl_dataset_managment import *
from .globals import *

__all__ = ["get_imageNet_negative_images", "get_metainfo", "download_google_images", "clean_images", "train_model", "evaluate"]


def get_imageNet_negative_images(nb_classes : int, nb_example : int):
    """
    get random imagenet folders to add part (or every) of those images to the negative examples in the cleaning set.

    Parameters
    ----------
    nb_classes :
        number of random classes to download
    nb_example :
        number of image per class
    """

    rmtree(PATH+"ImageNetFetched/", ignore_errors=False)
    makedirs(PATH+"ImageNetFetched/")

    PATH_DDL = "./zsl/fsl_classification/ImageNet_ddl/"
    try:
        system("python3 "+PATH_DDL+"downloader.py -data_root "+HEAD+"pipeline/ImageNetFetched -number_of_classes "+str(nb_classes)+ " -images_per_class "+str(nb_example))
    except:
        system("python "+PATH_DDL+"downloader.py -data_root "+HEAD+"pipeline/ImageNetFetched -number_of_classes "+str(nb_classes)+ " -images_per_class "+str(nb_example))



def download_google_images(classes : int, reset=False):
    """
    download 20 .jpg from google per class specified. If a class specified is already downloaded, the program won't download it again.

    Parameters
    ----------
    classes :
        the list of classes to download
    reset :
        a boolean indicating if the /pipeline/images/ folder must be rm-ed before downloading the content
    """

    if reset:
        rmtree(PATH_IMAGES, ignore_errors=False)
        makedirs(PATH_IMAGES)

    new_classes = list(set(classes) - set(listdir(PATH_IMAGES)))

    if new_classes != []:
        print("new classes to download :", new_classes, "\n")
        get_classes_images_URLLIB(new_classes, download=True)


def get_metainfo(CUB : bool, IMAGES: bool, OMNIGLOT: bool) -> Tuple[str, str, List[str]]:

    """
    get general information about the dataset used.

    Beware that using CUB or Omniglot without having downloaded the dataset in pipeline/model/data/CUB/images/ or pipeline/model/dataO/omniglot-py/images_background/
    will result in an error. What's more, the program will still attempt to clean those dataset (when used in the pipeline)

    Parameters
    ----------
    CUB / IMAGES / OMNIGLOT :
        indicates which dataset to download.

    Return
    ------
    the path to the dataset, the type of image conversion (e.g use grayscale or not) and the list of classes in the dataset
    """

    assert (~CUB)&(IMAGES^OMNIGLOT) + CUB&(~(IMAGES&OMNIGLOT)) == True; "At least two dataset are set to true"
    
    list_lass = []

    if CUB or IMAGES:
        PATH_DATA = HEAD+"pipeline/model/data/CUB/images/" if CUB else HEAD+"pipeline/images/"
        conversion_type = "CUB" if CUB else "IMG"
        list_lass = listdir(PATH_DATA)

    elif OMNIGLOT:
        PATH_DATA = HEAD+"pipeline/model/dataO/omniglot-py/images_background/" 
        conversion_type = "OMNI"
        list_alphabet = listdir(PATH_DATA)
        choosen_alphabet = list_alphabet[randint(0, len(list_alphabet)-1)]
        list_lass = [choosen_alphabet+"/"+char for char in listdir(PATH_DATA+choosen_alphabet)]

    return (PATH_DATA, conversion_type, list_lass)



def clean_images(PATH_DATA : str, classes : List[str], conversion_type : str):
    """
    clean the google images previously downloaded

    Parameters
    ----------
    PATH_DATA :
        the path to the dataset
    classes :
        the classes in the dataset
    conversion_type :
        the type of conversion to use
    """

    provider = CleaningSetProvider(PATH_DATA, 20, 2, conversion_type)
    meta_set = provider.get_set_of_cleaning_sets([classes[randint(0, len(classes)-1)].replace(" ", "")])

    rows, columns = meta_set.lenght()

    i, j = randint(0, rows-1), randint(0, columns-1) 

    just_support = get_only_images([meta_set(i, j)[0]])
    just_query = meta_set(i, j)[1][0]
    full_support = torch.cat((just_support[:meta_set(i,j)[2]-1], meta_set(i, j)[1][0].unsqueeze(0)), 0)

    plot_images(full_support, title="all images", images_per_row=N_SHOT)

    plot_images(just_support, title="S("+str(i)+";"+str(j)+")", images_per_row=N_SHOT)
    plot_images(just_query, title="Q("+str(i)+";"+str(j)+")", images_per_row=1)

    cleaner = Cleaner(PATH_MODEL, model_cleaning, meta_set, cuda_)
    similarity_matrix = cleaner.clean_sets()



def train_model(PATH_DATA : str, classes : List[str], conversion_type : str) -> Tensor:
    """
    train a siamese network model with the specified dataset

    Parameters
    ----------
    PATH_DATA :
        the path to the training / testing dataset
    classes :
        the list of classes in the dataset
    conversion_type :
        the type of conversion to use

    Return
    ------
    the support set used for training
    """

    support_classes = [PATH_DATA+class_+"/" for class_ in classes]
    N_SHOT = get_n_shot(support_classes)
    support_set, _ = get_sets(support_classes, N_SHOT, 0, conversion_type)

    print("N_SHOT IS CONFIGURED TO BE", N_SHOT)

    just_support = get_only_images(support_set)
    plot_images(just_support, title="support set", images_per_row=5)
    training_model.training(support_set, (0, 0), 0)

    return support_set



def evaluate(image_path : str, support_set : Tensor, classes : List[str], conversion_type : str) -> str:
    """
    evaluate a query with the trained model

    Parameters
    ----------
    image_path :
        the path to the query
    support_set :
        the support set used in training
    classes :
        the list of classes in the dataset
    conversion_type :
        the type of conversion to use

    Return
    ------
    a list containing the label of the image
    """

    PATH_TO_UNKNOWN = image_path
    queries = listdir(PATH_TO_UNKNOWN)
    query = get_image_tensor(PATH_TO_UNKNOWN+queries[randint(0, len(queries)-1)], conversion_type=conversion_type).unsqueeze(0)
    plot_images(query, title="unkown image", images_per_row=1)

    evaluation_model = Tester(training_model.model)
    predicted_label = evaluation_model.query_evaluation(support_set, query)

    # possible error with classes[predictedLabel] to check thourougly
    return classes[predicted_label]


