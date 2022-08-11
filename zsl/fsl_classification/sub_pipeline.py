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



"""
@desc get random imagenet folders to add part (or every) of those images to the negative examples in the cleaning set.

@param nb_classes number of random classes to download
@param nb_example number of image per class
"""
def get_imageNet_negative_images(nb_classes : int, nb_example : int):

    rmtree(PATH+"ImageNetFetched/", ignore_errors=False)
    makedirs(PATH+"ImageNetFetched/")

    PATH_DDL = "./zsl/fsl_classification/ImageNet_ddl/"
    try:
        system("python3 "+PATH_DDL+"downloader.py -data_root "+HEAD+"pipeline/ImageNetFetched -number_of_classes "+str(nb_classes)+ " -images_per_class "+str(nb_example))
    except:
        system("python "+PATH_DDL+"downloader.py -data_root "+HEAD+"pipeline/ImageNetFetched -number_of_classes "+str(nb_classes)+ " -images_per_class "+str(nb_example))


"""
@desc download 20 images from google per class specified. If a class specified is already downloaded, the program won't download it again.

@param classes the list of classes to download
@param reset a boolean indicating if the /pipeline/images/ folder must be rm-ed before downloading the content
"""
def download_google_images(classes : int, reset=False):

    if reset:
        rmtree(PATH_IMAGES, ignore_errors=False)
        makedirs(PATH_IMAGES)

    new_classes = list(set(classes) - set(listdir(PATH_IMAGES)))

    if new_classes != []:
        print("new classes to download :", new_classes, "\n")
        get_classes_images_URLLIB(new_classes, download=True)


"""
@desc get general information about the dataset used

@param CUB / IMAGES / OMNIGLOT indicates which dataset to download.

@return the path to the dataset, the type of image conversion (e.g use grayscale or not) and the list of classes in the dataset

"""
def get_metainfo(CUB : bool, IMAGES: bool, OMNIGLOT: bool) -> Tuple[str, str, List[str]]:

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


"""
@desc clean the google images previously downloaded

@param PATH_DATA the path to the dataset
@classes the classes in the dataset
@conversion_type the type of conversion to use
"""
def clean_images(PATH_DATA : str, classes : List[str], conversion_type : str):

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


"""
@desc train a siamese network model with the specified dataset

@param PATH_DATA the path to the training / testing dataset
@param classes the list of classes in the dataset
@conversion_type the type of conversion to use

@return the support set used for training
"""
def train_model(PATH_DATA : str, classes : List[str], conversion_type : str) -> Tensor:

    support_classes = [PATH_DATA+class_+"/" for class_ in classes]
    N_SHOT = get_n_shot(support_classes)
    support_set, _ = get_sets(support_classes, N_SHOT, 0, conversion_type)

    print("N_SHOT IS CONFIGURED TO BE", N_SHOT)

    just_support = get_only_images(support_set)
    plot_images(just_support, title="support set", images_per_row=5)
    training_model.training(support_set, (0, 0), 0)

    return support_set


"""
@desc evaluate a query with the trained model

@param image_path the path to the query
@param supportSet the support set used in training
@param classes the list of classes in the dataset
@conversion_type the type of conversion to use

@return a list containing the label of the image
"""
def evaluate(image_path : str, support_set : Tensor, classes : List[str], conversion_type : str) -> str:

    PATH_TO_UNKNOWN = image_path
    queries = listdir(PATH_TO_UNKNOWN)
    query = get_image_tensor(PATH_TO_UNKNOWN+queries[randint(0, len(queries)-1)], conversion_type=conversion_type).unsqueeze(0)
    plot_images(query, title="unkown image", images_per_row=1)

    evaluation_model = Tester(training_model.model)
    predicted_label = evaluation_model.query_evaluation(support_set, query)

    # possible error with classes[predictedLabel] to check thourougly
    return classes[predicted_label]


