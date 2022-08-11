import os

from torch import Tensor
from typing import List, Tuple

# must finish by a /
HEAD = os.getcwd()+"/zsl/fsl_classification/"

PATH = HEAD+"pipeline/"
PATH_IMAGES = HEAD+"pipeline/images/"
PATH_MODEL = HEAD+"pipeline/model/"

global_size = 20
LEN_FOR_ONE_SCROLL = 20

N_WAY = 5
N_SHOT = 5
N_QUERY = 15

batch_size = 125