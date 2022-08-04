from os import listdir, remove
from random import randint

from .constants import *
from .utils import *
from .utils_retrieval import getClassesImagesURLLIB
from .siamese_network import *
from .classes_for_model import *
from .classes_for_dataset import CleaningSetProvider
from .fsl_dataset_managment import *
from .globals import *

__all__ = ["get_path_data", "getGoogleImages", "cleanImages", "train_model", "evaluate"]

def get_path_data(CUB, IMAGES, OMNIGLOT):
    
    listClass = []

    if CUB or IMAGES:
        PATH_DATA = HEAD+"pipeline/model/data/CUB/images/" if CUB else HEAD+"pipeline/images/"
        listClass = listdir(PATH_DATA)

    elif OMNIGLOT:
        PATH_DATA = HEAD+"pipeline/model/dataO/omniglot-py/images_background/" 
        listAlphabet = listdir(PATH_DATA)
        choosenAlphabet = listAlphabet[randint(0, len(listAlphabet)-1)]
        listClass = [choosenAlphabet+"/"+char for char in listdir(PATH_DATA+choosenAlphabet)]

    return PATH_DATA, listClass

def getGoogleImages(classes):

    if "undone.txt" in listdir(HEAD+PATH_IMAGES):
        getClassesImagesURLLIB(classes, retrieval_not_done=True)
        remove(HEAD+"pipeline/image/undone.txt")


def cleanImages(PATH_DATA, classes):

    provider = CleaningSetProvider(PATH_DATA, 20, 2)
    meta_set = provider.getSetOfCleaningSets([classes[randint(0, len(classes)-1)].replace(" ", "")])

    rows, columns = meta_set.lenght()

    i, j = randint(0, rows-1), randint(0, columns-1) 

    justSupport = getOnlyImages([meta_set(i, j)[0]])
    justQuery = meta_set(i, j)[1][0]
    fullSupport = torch.cat((justSupport[:meta_set(i,j)[2]-1], meta_set(i, j)[1][0].unsqueeze(0)), 0)

    plot_images(fullSupport, title="all images", images_per_row=N_SHOT)

    plot_images(justSupport, title="S("+str(i)+";"+str(j)+")", images_per_row=N_SHOT)
    plot_images(justQuery, title="Q("+str(i)+";"+str(j)+")", images_per_row=1)

    cleaner = Cleaner(PATH_MODEL, modelCleaning, meta_set, cuda_)
    simM = cleaner.cleanSets()


def train_model(PATH_DATA, classes):

    supportClasses = [PATH_DATA+class_+"/" for class_ in classes]
    N_SHOT = getMin(supportClasses)
    supportSet, _ = getSets(supportClasses, N_SHOT, 0)

    print("N_SHOT IS CONFIGURED TO BE", N_SHOT)

    justSupport = getOnlyImages(supportSet)
    plot_images(justSupport, title="support set", images_per_row=5)

    training_model.training(supportSet, (0, 0), 0)

    return supportSet


def evaluate(supportSet, classes):

    PATH_TO_UNKNOWN = HEAD+"pipeline/unkown image/"
    queries = listdir(PATH_TO_UNKNOWN)
    query = getImageTensor(PATH_TO_UNKNOWN+queries[randint(0, len(queries)-1)]).unsqueeze(0)
    plot_images(query, title="unkown image", images_per_row=1)

    evaluation_model = Tester(training_model.model)
    predictedLabel = evaluation_model.queryEvaluation(supportSet, [query])

    # possible error with classes[predictedLabel] to check thourougly
    return classes[predictedLabel]


