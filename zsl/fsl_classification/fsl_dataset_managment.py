
from os import listdir
from random import randint

from .utils import natural_keys
from .utils_dataset import getImageTensor
 

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

def getFolderTensorsForTraining(path, supportNumber, queryNumber, label, conversion_type):

  images = listdir(path)
  images.sort(key=natural_keys)
  support_i, query_i = [], []

  for i in range(0, supportNumber):
    ridx = randint(0, len(images)-1)
    try:
      support_i.append( (getImageTensor(path+images[ridx], conversion_type=conversion_type), label) )
    except:
      print("support image could not be loaded")

    images.remove(images[ridx])
  
  for i in range(0, queryNumber):
    ridx = randint(0, len(images)-1)
    try:
      query_i.append( (getImageTensor(path+images[ridx], conversion_type=conversion_type), label) )
    except:
      print("query image could not be loaded")

    images.remove(images[ridx])

  return support_i, query_i


"""
@desc get the support set and query set for training the model after cleaning

@param supportNumber the n_shot parameter 
@param queryNumber the number of query

@return the support and query set
"""
def getSets(paths, supportNumber, queryNumber, conversion_type):

  supportSet, querySet = [], []
  for label, path in enumerate(paths):
    Si, Qi = getFolderTensorsForTraining(path, supportNumber, queryNumber, label, conversion_type)
    supportSet.append(Si)
    querySet.append(Qi)

  return supportSet, querySet
