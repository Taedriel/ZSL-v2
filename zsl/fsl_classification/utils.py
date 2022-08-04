import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from statistics import mean
import torch
import utils
from sklearn.metrics import classification_report, confusion_matrix
from os import listdir
import re

from .constants import *

"""
@desc get mean and incertitude from a list of mesurements

@param mesurements list of experiments, the title of the experiement is the first element.
@param k the student number

@return the string representing the mean and uncertainty of the experiments
"""
def getUa(mesurements, k=1):

  title = mesurements[0]
  l = mesurements[1:]
  inv_n = 1.0/(len(l)-1)
  inv_ns = 1.0/pow(len(l), 0.5)
  m_ = round(mean(l), 2)
  dm2 = [pow(mi - m_, 2) for mi in l]
  pstd = pow(inv_n*sum(dm2), 0.5)
  u = round(k*pstd*inv_ns, 2)
  
  return title+str(m_) + "% +- " + str(u) + " (with 95% confidence)" if k!=1 else "", m_, u


def getUaList(listOfExperiences, k=1):

  for mesurements in listOfExperiences:
    string, m_, u = getUa(mesurements, k)
    print(string)


"""
@desc remove the label from each tuple (I, label) in a set of images (used for display)

@param set_ the set of images to remove the labels from

@return the set without label
"""
def getOnlyImages(set_):
  
  justSet = []
  for class_ in set_:
    for image in class_:
      justSet.append(image[0])

  return torch.stack(justSet)


def plot_images(images, title, images_per_row):
  plt.figure()
  plt.title(title)
  plt.imshow(utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0))


def saveFile(filename, data):
  file = open(PATH_MODEL+filename, "w+")
  for d in data:
    file.write(str(d)+"\n")
  file.close()


def showRegression(rangeOfData, data, degree):
  coef = np.polyfit(rangeOfData, data, degree)
  poly1d_fn = np.poly1d(coef) 
  plt.plot(rangeOfData, data, '-yo', rangeOfData, poly1d_fn(rangeOfData), '--k')
  plt.show()
  print("regression polynome :\n")
  print(np.poly1d(poly1d_fn))
  print("started at", data[0], "and ended at", data[-1])
  print("\n")


def showData(data, title, degree, saveInfo=[False, ""]):

  path = saveInfo[1]
  if saveInfo[0]:
    saveFile(path, data)

  numberOfIteration = range(0, len(data))

  plt.plot(numberOfIteration, data)
  plt.title(title)
  plt.show()
  showRegression(numberOfIteration, data, degree)


"""
@desc get the confusion matrix and additional information based on the result of 
the model test.

@param labels a list of number representing each class
@param predicted_labels a list of predicted class by the model

@return accuracy, confusion matrix and other details (see scipy)
"""
def getMatrixReport(labels, predicted_labels):
    print("\n")
    listOfClasses = list(dict.fromkeys(labels))
    res = classification_report(labels, predicted_labels, target_names=["c"+str(i) for i in range(1, len(listOfClasses)+1)],  output_dict=True)
    return res, res['accuracy'], confusion_matrix(labels, predicted_labels)


"""
@desc verify if the numbers are in a specific range

@param interval a list of number to check
@param range_ the specific range 

@return the number of number in the specified range
"""
def numberInInterval(interval, range_):
  in_ = 0
  minX = range_[0]
  maxX = range_[1]
  for x in interval:
    if x >= minX and x < maxX:
      in_+=1
    
    if x == 1 and maxX == 1:
      in_+=1

  return in_


def getDistributionOnPred(correctPredictions, bins):
  NumberDist = []
  for elem in bins:
    NumberDist.append(100*numberInInterval(correctPredictions, elem)/len(correctPredictions))

  return NumberDist


def createHistogramPreds(predictions, title):

  bins = [(i*0.1, (i+1)*0.1) for i in range(0, 10)]
  NumberDist = getDistributionOnPred(predictions, bins)

  fig, ax = plt.subplots()
  for i, percentage in enumerate(NumberDist):
    ax.add_patch(Rectangle((0.1*i, 0), 0.1, percentage, edgecolor='black'))
  plt.ylim(int(max(NumberDist))+2)
  plt.gca().invert_yaxis()
  plt.xlabel(title)
  plt.ylabel("number of predictions in %")
  plt.show()

  return NumberDist


"""
@desc separate the original distribution into two based on a value

@param dist the distribution to split in half
@param threshold the value that represent the split

@return two new distributions
"""
def getSimilarityDistributions(dist, threshold):
  dist1 = []
  dist2 = []

  for x in dist:
    if x > threshold:
      dist2.append(x)
    else:
      dist1.append(x)

  return dist1, dist2


"""
@desc get the r parameter used to discriminate between good and bad images during cleaning

@param dist1 the lower distribution (from getSimilarityDistributions)
@param dist2 the higher distribution(ibid.) 

@return the r parameter of an image
"""
def getR(dist1, dist2):

  L1, L2 = len(dist1), len(dist2)
  return L2/L1 if L1 != 0 else -1


def printDistributions(dist, threshold):

  dist1, dist2 = getSimilarityDistributions(dist, threshold)
  L1, L2 = len(dist1), len(dist2)
  r = getR(dist1, dist2)
  
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


def getMin(supportClasses):
  """
  because classes after cleaning do not necesserily have the same number of elements,
  thus there's the need to avoid getting an out of range error
  """
  lenghts = []
  for pathToClass in supportClasses:
    lenghts.append(len(listdir(pathToClass)))

  return min(lenghts)


def printSimMatrix(M):
  for row in M:
    print(" ".join(list(map(lambda x: '{0: <8}'.format(x),row))), "\n")


"""
@deprecated (do not use file anymore)

@desc get set of all classes to be used for cleaning from a text file

@param classesFile the path to the file
@param searchPrecision additional keywords

@return the list of all classes without spaces between keywords
"""
def createListOfClasses(classesFile, searchPrecision=""):

  classes = []
  index=0

  for animal in classesFile.readlines():
    animal = animal.strip("\n").replace(" ", "")
    classes.append(animal + searchPrecision)

  return classes
