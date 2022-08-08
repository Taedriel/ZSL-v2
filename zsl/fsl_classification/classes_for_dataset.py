from os import listdir

from .constants import *
from .utils import natural_keys
from .utils_dataset import getImageTensor

"""
@desc The MetaSet class implement a way to handle images during the cleaning process. 
It does so by constructing a set of matrices, each representing one aspect of the data
"""
class MetaSet:

  def __init__(self):
    """
    the lenght parameter is here in order to avoid testing the query against
    unrelated images
    """
    self.supportSetMatrix = []
    self.querySetMatrix = []
    self.supportLenghtMatrix = []
    self.imageNameSetMatrix = []
    self.buffer = [[], [], []]


  def __call__(self, i, j):
    return (self.supportSetMatrix[i][j], self.querySetMatrix[i][j], self.supportLenghtMatrix[i][j])


  def lenght(self):
    rows = len(self.supportSetMatrix)
    columns = len(self.supportSetMatrix[0]) if self.supportSetMatrix != [] else 0 

    return rows, columns


  def clearBuffer(self):
    self.buffer = [[], [], []]


  """
  @desc add values to the buffer to be put into the matrices later

  @param arrayOfValues must have the form [support set Images, Query set images, lenght of support set]
  """
  def addToBuffer(self, arrayOfValue):
    for index in range(0, 3):
      self.buffer[index].append(arrayOfValue[index])


  """
  @desc add the values present in the buffer to each matrix and then clear the buffer
  """
  def setLine(self):
    self.supportSetMatrix.append(self.buffer[0])
    self.querySetMatrix.append(self.buffer[1])
    self.supportLenghtMatrix.append(self.buffer[2])
    self.clearBuffer()



"""
@desc Class implementing the data structure creation for the cleaning part of the pipeline. 
Among other thing, this class add negative examples to the support set after extracting the query.
"""
class CleaningSetProvider:

  def __init__(self, pathToImage, numberOfPositives, numberOfNegatives, conversion_type):
    self.pathToNegatives = HEAD+"pipeline/ImageNetFetched/"
    self.path = pathToImage
    self.nbNegative = numberOfNegatives
    self.nbPositive = numberOfPositives

    self.type = conversion_type


  def getSubSetOfImages(self, path):
    images = listdir(path)
    images.sort(key=natural_keys)
    images = images[0:self.nbPositive]

    return images


  def addNegativeExamples(self, pathToClass, label, supportSet):

    images = listdir(pathToClass)
    for i in range(0, self.nbNegative):
      supportSet.append((getImageTensor(pathToClass+images[i], conversion_type=self.type), label))

    return supportSet


  def addNegativeClasses(self, supportSet):

    negativeClasses = listdir(self.pathToNegatives)
    for i in range(0, len(negativeClasses)):
      supportSet = self.addNegativeExamples(self.pathToNegatives+negativeClasses[i]+"/", i+1, supportSet)

    return supportSet


  """
  @desc construct one element to be put into the metaSet.

  @param pathToClass the path to the class images
  @param label the label of the class
  @param q_index the index of the query image in the folder

  @return the corresponding support and query set + the original lenght of the support set
  """
  def getFolderTensorsForCleaning(self, pathToClass, label, q_index):

    images = self.getSubSetOfImages(pathToClass)
    originalLenght = len(images)

    query = (getImageTensor(pathToClass+images[q_index], conversion_type=self.type), label)
    images.remove(images[q_index])

    support = []
    for image in images:
      support.append((getImageTensor(pathToClass+image, conversion_type=self.type), label))
    
    support = self.addNegativeClasses(support)

    return [support, query, originalLenght]


  def getSetOfCleaningSets(self, listClasses):

    meta_set = MetaSet()
    for class_ in listClasses:

      listEx = self.getSubSetOfImages(self.path+class_)
      for q_index in range(0, len(listEx)):
        oneElement = self.getFolderTensorsForCleaning(self.path+class_+"/", 0, q_index)
        meta_set.addToBuffer(oneElement)

      meta_set.setLine()
      meta_set.imageNameSetMatrix.append([self.path+class_+"/"+image for image in listEx])

    return meta_set
