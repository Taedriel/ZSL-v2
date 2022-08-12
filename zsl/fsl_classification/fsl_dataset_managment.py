
from os import listdir
from random import randint

from .utils import natural_keys
from .utils_dataset import get_image_tensor

from torch import Tensor
from typing import List, Tuple
 

# each set is as follows, with I an image and c its label (a number) : 
"""
[
  [(I, c), (I, c), ..., (I, c), (I, c)],
  [(I, c), (I, c), ..., (I, c), (I, c)],
   .          .                    .
   .            .                  .
   .              .                .
   .                .              .
                      .            .
  [(I, c), (I, c), ..., (I, c), (I, c)]
]
"""

def get_folder_tensors_for_training(path : str, support_number : int, query_number : int, label : int, conversion_type : str) -> Tuple[Tensor, Tensor]:

  images = listdir(path)
  images.sort(key=natural_keys)
  support_i, query_i = [], []

  for i in range(0, support_number):
    ridx = randint(0, len(images)-1)
    try:
      support_i.append( (get_image_tensor(path+images[ridx], conversion_type=conversion_type), label) )
    except:
      print("support image could not be loaded")

    images.remove(images[ridx])
  
  for i in range(0, query_number):
    ridx = randint(0, len(images)-1)
    try:
      query_i.append( (get_image_tensor(path+images[ridx], conversion_type=conversion_type), label) )
    except:
      print("query image could not be loaded")

    images.remove(images[ridx])

  return support_i, query_i


def get_sets(paths : List[str], support_number : int, query_number : int, conversion_type : str) -> Tuple[Tensor, Tensor]:
  """
  get the support set and query set for training the model after cleaning

  Parameters
  ----------
  support_number :
    the n_shot parameter 
  query_number :
    the number of query

  Return
  ------
  the support and query set
  """

  support_set, query_set = [], []
  for label, path in enumerate(paths):
    Si, Qi = get_folder_tensors_for_training(path, support_number, query_number, label, conversion_type)
    support_set.append(Si)
    query_set.append(Qi)

  return support_set, query_set
