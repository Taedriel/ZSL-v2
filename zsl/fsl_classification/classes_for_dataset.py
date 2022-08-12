from os import listdir

from .constants import *
from .utils import natural_keys
from .utils_dataset import get_image_tensor

"""
Return the first elements of the Series.

This function is mainly useful to preview the values of the
Series without displaying all of it.

Parameters
----------
n : int
    Number of values to return.

Return
------
"""

class MetaSet:
  """
  The MetaSet class implement a way to handle images during the cleaning process. 

  It does so by constructing a set of matrices, each representing one aspect of the data
  """

  def __init__(self):
    """
    the lenght parameter is here in order to avoid testing the query against unrelated images
    """
    self.support_set_matrix = []
    self.query_set_matrix = []
    self.support_lenght_matrix = []
    self.imageName_set_matrix = []
    self.buffer = [[], [], []]


  def __call__(self, i : int, j : int) -> Tuple[Tensor, Tensor, int]:
    return (self.support_set_matrix[i][j], self.query_set_matrix[i][j], self.support_lenght_matrix[i][j])


  def lenght(self) -> Tuple[int, int]:
    rows = len(self.support_set_matrix)
    columns = len(self.support_set_matrix[0]) if self.support_set_matrix != [] else 0 

    return rows, columns


  def clear_buffer(self):
    self.buffer = [[], [], []]


  """
  @desc 

  @param arrayOfValues 
  """
  def add_to_buffer(self, array_of_value : List[int]):
    """
    add values to the buffer to be put into the matrices later

    Parameters
    ----------
    array_of_values :
      must have the form [support set Images, Query set images, lenght of support set]

    """

    for index in range(0, 3):
      self.buffer[index].append(array_of_value[index])



  def set_line(self):
    """
    add the values present in the buffer to each matrix and then clear the buffer
    """

    self.support_set_matrix.append(self.buffer[0])
    self.query_set_matrix.append(self.buffer[1])
    self.support_lenght_matrix.append(self.buffer[2])
    self.clear_buffer()



class CleaningSetProvider:
  """
  Class implementing the data structure creation for the cleaning part of the pipeline. 

  Among other thing, this class add negative examples to the support set after extracting the query.
  """

  def __init__(self, path_to_image : str, number_of_positives : int, number_of_negatives : int, conversion_type : str):
    self.path_to_negatives = HEAD+"pipeline/ImageNetFetched/imagenet_images/"
    self.path = path_to_image
    self.nb_negative = number_of_negatives
    self.nb_positive = number_of_positives

    self.type = conversion_type


  def get_subset_of_images(self, path : str) -> List[Tensor]:
    images = listdir(path)
    images.sort(key=natural_keys)
    images = images[0:self.nb_positive]

    return images


  def add_negative_examples(self, path_to_class : str, label : int, support_set : Tensor) -> Tensor:

    images = listdir(path_to_class)
    for i in range(0, self.nb_negative):
      support_set.append((get_image_tensor(path_to_class+images[i], conversion_type=self.type), label))

    return support_set


  def add_negative_classes(self, support_set : Tensor) -> Tensor:

    negative_classes = listdir(self.path_to_negatives)
    for i in range(0, len(negative_classes)):
      support_set = self.add_negative_examples(self.path_to_negatives+negative_classes[i]+"/", i+1, support_set)

    return support_set


  def get_folder_tensors_for_cleaning(self, path_to_class : str, label : int, q_index : int) -> Tuple[Tensor, Tensor, int]:
    """
    construct one element to be put into the metaSet.

    Parameters
    ----------
    path_to_class :
      the path to the class images
    label :
      the label of the class
    q_index :
      the index of the query image in the folder

    Return
    ------
    the corresponding support and query set + the original lenght of the support set
    """

    images = self.get_subset_of_images(path_to_class)
    original_lenght = len(images)

    query = (get_image_tensor(path_to_class+images[q_index], conversion_type=self.type), label)
    images.remove(images[q_index])

    support = []
    for image in images:
      support.append((get_image_tensor(path_to_class+image, conversion_type=self.type), label))
    
    support = self.add_negative_classes(support)

    return [support, query, original_lenght]


  def get_set_of_cleaning_sets(self, list_classes : List[str]) -> MetaSet:

    meta_set = MetaSet()
    for class_ in list_classes:

      list_exemples = self.get_subset_of_images(self.path+class_)
      for q_index in range(0, len(list_exemples)):
        one_element = self.get_folder_tensors_for_cleaning(self.path+class_+"/", 0, q_index)
        meta_set.add_to_buffer(one_element)

      meta_set.set_line()
      meta_set.imageName_set_matrix.append([self.path+class_+"/"+image for image in list_exemples])

    return meta_set
