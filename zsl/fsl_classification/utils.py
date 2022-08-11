import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from statistics import mean
import torch
from torchvision import utils
from sklearn.metrics import classification_report, confusion_matrix
from os import listdir
import re

from .constants import *

"""
@desc get mean and incertitude from a list of mesurements. See https://www.lycee-champollion.fr/IMG/pdf/mesures_et_incertitudes.pdf for the meaning of the variables

@param mesurements list of experiments, the title of the experiement is the first element.
@param k the student number

@return the string representing the mean and uncertainty of the experiments
"""
def get_ua(mesurements : List[float], k=1) -> Tuple[str, float, float]:

  title = mesurements[0]
  l = mesurements[1:]
  inv_n = 1.0/(len(l)-1)
  inv_ns = 1.0/pow(len(l), 0.5)
  m_ = round(mean(l), 2)
  dm2 = [pow(mi - m_, 2) for mi in l]
  pstd = pow(inv_n*sum(dm2), 0.5)

  u = round(k*pstd*inv_ns, 2)
  
  return title+str(m_) + "% +- " + str(u) + " (with 95% confidence)" if k!=1 else "", m_, u


def get_ua_list(list_of_experiences : List[List[float]], k=1):

  for mesurements in list_of_experiences:
    string, m_, u = get_ua(mesurements, k)
    print(string)


"""
@desc remove the label from each tuple (I, label) in a set of images (used for display)

@param set_ the set of images to remove the labels from

@return the set without label
"""
def get_only_images(set_ : Tensor) -> Tensor:
  
  just_set = []
  for class_ in set_:
    for image in class_:
      just_set.append(image[0])

  return torch.stack(just_set)


def plot_images(images : Tensor, title : str, images_per_row : int):
  plt.figure()
  plt.title(title)
  plt.imshow(utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0))


def save_file(filename : str, data : List[any]):
  file = open(PATH_MODEL+filename, "w+")
  for d in data:
    file.write(str(d)+"\n")
  file.close()


def show_regression(range_of_data : range, data : List[float], degree : int):
  coef = np.polyfit(range_of_data, data, degree)
  poly1d_fn = np.poly1d(coef) 
  plt.plot(range_of_data, data, '-yo', range_of_data, poly1d_fn(range_of_data), '--k')
  plt.show()
  print("regression polynome :\n")
  print(np.poly1d(poly1d_fn))
  print("started at", data[0], "and ended at", data[-1])
  print("\n")


"""
@desc show the data present in data

@param data the data to display
@param title the graph title
@param degree the degree of the polynome for approximation
@save_info list of (is the data to be saved, where)
"""
def show_data(data : List[float], title : str, degree : int, save_info=[False, ""]):

  path = save_info[1]
  if save_info[0]:
    save_file(path, data)

  number_of_iteration = range(0, len(data))

  plt.plot(number_of_iteration, data)
  plt.title(title)
  plt.show()
  show_regression(number_of_iteration, data, degree)


"""
@desc get the confusion matrix and additional information based on the result of 
the model test.

@param labels a list of number representing each class
@param predicted_labels a list of predicted class by the model

@return accuracy, confusion matrix and other details (see scipy)
"""
def get_matrix_report(labels : int, predicted_labels : List[int]) -> Tuple[any, float, any]:
    print("\n")
    list_of_classes = list(dict.fromkeys(labels))
    res = classification_report(labels, predicted_labels, target_names=["c"+str(i) for i in range(1, len(list_of_classes)+1)],  output_dict=True)
    return res, res['accuracy'], confusion_matrix(labels, predicted_labels)


"""
@desc verify if the numbers are in a specific range

@param interval a list of number to check
@param range_ the specific range 

@return the number of number in the specified range
"""
def number_in_interval(interval : List[float], range_ : List[int]):
  in_ = 0
  minX = range_[0]
  maxX = range_[1]
  for x in interval:
    if x >= minX and x < maxX:
      in_+=1
    
    if x == 1 and maxX == 1:
      in_+=1

  return in_


def get_distribution_on_predictions(predictions : List[float], bins : int):
  distribution = []
  for elem in bins:
    distribution.append(100*number_in_interval(predictions, elem)/len(predictions))

  return distribution


def create_histogram_out_of_predictions(predictions : List[float], title : str) -> List[float]:

  bins = [(i*0.1, (i+1)*0.1) for i in range(0, 10)]
  distribution = get_distribution_on_predictions(predictions, bins)

  fig, ax = plt.subplots()
  for i, percentage in enumerate(distribution):
    ax.add_patch(Rectangle((0.1*i, 0), 0.1, percentage, edgecolor='black'))
  plt.ylim(int(max(distribution))+2)
  plt.gca().invert_yaxis()
  plt.xlabel(title)
  plt.ylabel("number of predictions in %")
  plt.show()

  return distribution


"""
@desc separate the original distribution into two based on a value

@param dist the distribution to split in half
@param threshold the value that represent the split

@return two new distributions
"""
def get_similarity_distributions(dist : List[float], threshold : float):
  disimilar_distribution = []
  similar_distribution = []

  for x in dist:
    if x > threshold:
      similar_distribution.append(x)
    else:
      disimilar_distribution.append(x)

  return disimilar_distribution, similar_distribution


"""
@desc get the r parameter used to discriminate between good and bad images during cleaning

@param dist1 the lower distribution (from getSimilarityDistributions)
@param dist2 the higher distribution(ibid.) 

@return the r parameter of an image
"""
def get_r(disimilar_distribution : List[float], similar_distribution : List[float]) -> float:

  L1, L2 = len(disimilar_distribution), len(similar_distribution)
  return L2/L1 if L1 != 0 else -1


def print_distribution(dist : List[float], threshold : float):

  dist1, dist2 = get_similarity_distributions(dist, threshold)
  L1, L2 = len(dist1), len(dist2)
  r = get_r(dist1, dist2)
  
  print("r (L2/L1) = ", r if r >= 0 else "+inf")
  plt.hist(dist1,bins=10)
  plt.hist(dist2,bins=10)
  plt.xlabel("similarity to the query")
  plt.ylabel("number of images")
  plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


"""
@desc because classes after cleaning do not necesserily have the same number of elements, thus there's the need to avoid getting an out of range error
"""
def get_n_shot(support_classes : List[str]) -> int:

  lenghts = []
  for path_to_class in support_classes:
    lenghts.append(len(listdir(path_to_class)))

  return min(lenghts)


def print_similarity_matrix(M):
  for row in M:
    print(" ".join(list(map(lambda x: '{0: <8}'.format(x),row))), "\n")
