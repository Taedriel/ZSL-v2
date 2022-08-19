# THIS FILE IS NOT INTENDED FOR USE IN THE PIPELINE AND IS ONLY FOR SEPARATE TESTING

import os, random
from .constants import *
from .classes_for_model import *
from .globals import *
from .utils import *
from .utils_dataset import *
from .fsl_dataset_managment import get_sets

CUB, IMAGES, OMNIGLOT = False, True, False
PATH_DATA = ""
list_class = []

if CUB or IMAGES:
  PATH_DATA = HEAD+"pipeline/model/data/CUB/images/" if CUB else HEAD+"pipeline/images/"
  list_class = os.listdir(PATH_DATA)
elif OMNIGLOT:
  PATH_DATA = HEAD+"pipeline/model/dataO/omniglot-py/images_background/" 
  list_alphabet = os.listdir(PATH_DATA)
  choosen_alphabet = list_alphabet[random.randint(0, len(list_alphabet)-1)]
  list_class = [choosen_alphabet+"/"+char for char in os.listdir(PATH_DATA+choosen_alphabet)]


support_classes = get_n_random_classes_paths(PATH_DATA, list_class, N_WAY)

print(support_classes)
support_set, query_set = get_sets(support_classes, N_SHOT, N_QUERY)

just_support = get_only_images(support_set)
just_query = get_only_images(query_set)
plot_images(just_support, title="support set", images_per_row=N_SHOT)
plot_images(just_query, title="query set", images_per_row=N_QUERY)

training_model = Trainer(PATH_MODEL, model, False)
training_model.reset_model(reset_by_param=False)

training_needed = True
if training_needed:
  losses = training_model.training(support_set, (0, 0), 0)

show_data(losses, "loss during training", 3)

evaluation_model = Tester(training_model.model)
accuracy_esults, y_pred, y, correct_predictions, incorrect_predictions, index_incorrect_query = evaluation_model.evaluate_with_metric(support_set, query_set)

_, _, confM = get_matrix_report(y, y_pred)
print(accuracy_esults)
print(confM)
print(index_incorrect_query)