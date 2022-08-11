from zsl.fsl_classification.classes_for_dataset import MetaSet
from zsl.fsl_classification.siamese_network import Siamese
from .constants import *
from .utils_sn import *
from .utils import plot_images, getSimilarityDistributions, getR, printSimMatrix

from torch import stack, tensor, save, no_grad
from torch.nn import BCELoss

from random import randint
from tqdm import tqdm
import itertools


def getIndexes(setOfImages : Tensor, nbClasses : int, cleaning=False) -> Tuple[int, int, int, int]:

  nbEx =  len(setOfImages) if cleaning else len(setOfImages[0])
  ci1, e1 = randint(0, nbClasses-1), randint(0, nbEx-1) 
  ci2, e2 = randint(0, nbClasses-1), randint(0, nbEx-1)

  return ci1, ci2, e1, e2


"""
the cleaning parameter is here because cleaning data and training data have
different shapes
"""
def getRandomPair(setOfImages : Tensor, cleaning=False) -> Tuple[Tensor, Tensor, int]:

  nbClasses = len(setOfImages)
  ci1, ci2, e1, e2 = getIndexes(setOfImages, nbClasses, cleaning)
  
  I1, I2, s = 0, 0, 0
  if cleaning:
      c1, c2 = setOfImages[e1][1], setOfImages[e2][1]
      I1, I2 = stack([setOfImages[e1][0]]), stack([setOfImages[e2][0]])
      s = 1.0 if c1 == c2 else 0.0
  else:
    I1, I2 = stack([setOfImages[ci1][e1][0]]), stack([setOfImages[ci2][e2][0]])
    s = 1.0 if ci1 == ci2 else 0.0

  return I1, I2, s


"""
@desc Class implementing the training of the model after the cleaning process
"""
class Trainer(ModelUtils):

    def __init__(self, path_model : str, model : Siamese, cuda_ : bool, cleaning : bool):

        super().__init__()
        self.cuda_ = cuda_
        self.cleaningPhase = cleaning
        self.model = model
        self.optimizer = super().getOptimizer(self.model.metric)
        self.LOSS = BCELoss()

        self.model_saver = ModelSaver(path_model)


    def resetModel(self, reset_by_param=False):

        if not reset_by_param:
            modelName = super().getModelName("train", self.cuda_)
            _, model, opti = self.model_saver.loadModel(modelName, self.model.metric, self.optimizer)
            self.model.metric, self.optimizer = model, opti
        else:
            self.model.metric = super().resetModelParam(self.model.metric)
            self.optimizer = super().getOptimizer(self.model.metric)


    def epoch(self, suppSet : Tensor) -> List[float]:

        lossForBatch = []
        for i in range(0, batchSize):

            I1, I2, s = getRandomPair(suppSet, self.cleaningPhase)
            I1, I2 = super().getImages(I1, I2)
            w, u, v = self.model.createCombinedVector(I1, I2)
            out = self.model(w)
            s = tensor([s]).cuda() if self.cuda_ else tensor([s])

            self.optimizer.zero_grad()
            loss = self.LOSS(out, s)
            loss.backward()
            self.optimizer.step()
            lossForBatch.append(loss.item())

        return lossForBatch


    """
    @desc training of the model

    @param supportSet the set of images as created by getSets
    @param epoch_loss used if the training is to resumed
    @param set_i used for tqdm (show on which class we are training)

    @return losses a list of the mean loss per epoch
    """
    def training(self, supportSet : Tensor, epoch_loss=(0, 0), set_i=0) -> List[int]:

        numberOfEpochs = 300-epoch_loss[0]
        valFreq = 10

        losses = [epoch_loss[1]] if epoch_loss[1] != 0 else []

        self.model.train()
        for epoch_i in tqdm(range(0, numberOfEpochs), desc="Traning on Set "+str(set_i)):

            batchLoss = self.epoch(supportSet)
            mean_loss = sum(batchLoss)*1.0/len(batchLoss)
            losses.append(mean_loss)

            if epoch_i % valFreq == 0 and epoch_i != 0:
                model_opti =  [self.model.metric, self.optimizer]
                self.model_saver.saveModel("SNTrain.pt", model_opti, epoch_i, mean_loss)

        save(self.model.metric.state_dict(), PATH_MODEL+"SN.pt")

        return losses



"""
@desc Class implementing the testing part of the pipeline after the cleaning process
"""
class Tester(ModelUtils):

    def __init__(self, model):
        super().__init__()
        self.model = model

    """
    @desc get the class that is the most represented in a list of predictions

    @param predictions a list of prediction with (pred, label) format

    @return the label of the most represented class
    """
    def getFirstClassBasedOnRepresentation(self, predictions : List[Tuple[Tensor, int]]) -> int:

        representation = [0]*N_WAY
        for pred in predictions:
            representation[pred[0]] += 1

        return representation.index(max(representation))


    def isModelCorrect(self, predictions : List[Tuple[Tensor, int]], queryClass : int) -> Tuple[int, int, float]:

        pred_sorted = sorted(predictions, key=lambda tup: tup[-1], reverse=True)
        first_five = pred_sorted[0:5]

        predictedClassLabel = self.getFirstClassBasedOnRepresentation(first_five)
        similarity = int(queryClass == predictedClassLabel)
        predictionScore = first_five[0][1]

        return similarity, predictedClassLabel, predictionScore


    def model_prediction(self, queryInfo : Tuple[Tensor, int], imageInfo : Tuple[Tensor, int], triplets : List[Tuple[Tensor, int]]) -> Tuple[List[Tuple[Tensor, int]], int, int, float]:

        image, imageClass = imageInfo
        query, queryClass = queryInfo

        query, image = super().getImages(query, image)
        w, u, v = self.model.createCombinedVector(query, image)
        p = self.model(w)
        triplets.append((imageClass, p))

        areReallySimilar, imageClass, prediction = self.isModelCorrect(triplets, queryClass)

        return triplets, areReallySimilar, imageClass, prediction


    def evaluateWithMetric(self, supportSet : Tensor, querySet : Tensor):

        triplets = []
        pred_labels = []
        query_labels = []
        correctPreds, incorrectPreds, indexIncorrectQuery = [], [], []
        correct = 0

        self.model.eval()
        with no_grad():

            for indexQuery, queryInfo in enumerate(list(itertools.chain(*querySet))):

                _, queryClass = queryInfo
                for imageInfo in list(itertools.chain(*supportSet)):
                    triplets, areReallySimilar, imageClass, prediction = self.model_prediction(queryInfo, imageInfo, triplets)
                    
                if areReallySimilar == 1: 
                    correct+=1
                    correctPreds.append(prediction)
                else:
                    incorrectPreds.append(prediction)
                    indexIncorrectQuery.append(indexQuery)

                triplets = []
                pred_labels.append(imageClass)
                query_labels.append(queryClass)

        return "\n accuracy :"+str(100.0*correct/(N_WAY*N_QUERY)), pred_labels, query_labels, correctPreds, incorrectPreds, indexIncorrectQuery
        

    def queryEvaluation(self, supportSet : Tensor, query : Tensor) -> int:

        triplets = []
        self.model.eval()
        with no_grad():

            for imageInfo in list(itertools.chain(*supportSet)):

                image, imageClass = imageInfo
                query, image = super().getImages(query, image)
                w, _, _ = self.model.createCombinedVector(query, image)
                p = self.model(w)
                triplets.append((imageClass, p))
            
        pred_sorted = sorted(triplets, key=lambda tup: tup[-1], reverse=True)
        first_five = pred_sorted[0:5]
        predictedClassLabel = self.getFirstClassBasedOnRepresentation(first_five)

        return predictedClassLabel



"""
@desc class implementing the cleaning part of the pipeline
"""
class Cleaner(ModelUtils):

    def __init__(self, path_model : str, model : Siamese, meta_set : MetaSet, cuda_ : bool):
        super().__init__()

        self.meta_set = meta_set
        self.threshold = 0.5

        self.model_t = Trainer(path_model, model, cuda_, True)
        self.model_saver = ModelSaver(path_model)

        self.falseVector = []
        self.trueVector = []

        self.cuda_ = cuda_


    def isQueryOutsider(self, Eij : Tuple[Tensor, Tensor, int]) -> float:

        model = self.model_t.model
        supportSet, querySet, lenght = Eij[0], Eij[1], Eij[2]-1
        preds = []

        model.eval()
        with no_grad():

            query_raw = querySet[0]
            for image_raw, imageClass in supportSet[:lenght]:

                query, image = self.model_t.getImages(query_raw, image_raw)
                w = model.createCombinedVector(query, image)
                p = model(w)
                res = round(p.cpu().item(), 2)
                preds.append(res)
            
        return preds


    #TODO CAN CHANGE THE RELOADING MODEL HERE(cleaning context)
    def getPredictionsForOneQuery(self, Eij : Tuple[Tensor, Tensor, int], indexSet : int) -> float:

        modelName = super().getModelName("clean", self.cuda_)
        _, model_, opti = self.model_saver.loadModel(modelName, self.model_t.model.metric, self.model_t.optimizer)
        self.model_t.model.metric, self.model_t.optimizer = model_, opti

        self.model_t.training(Eij[0], set_i=indexSet) 
        preds = self.isQueryOutsider(Eij)

        return preds


    """
    @desc decide if the query is to be kept or not based on the r parameter

    @param indices the position of the image in the matrix
    @param Qij the query
    @param r its r parameter has calculated by getR()
    @param imageToClean a list of images

    @return the updated list of images to remove
    """
    def decideOnImageType(self, indices : int, Qij : Tensor, r : float, imageToClean : List[str]) -> List[str]:

        tmp = imageToClean
        if 0 <= r < 1:
            self.falseVector.append(Qij)
            tmp.append(self.meta_set.imageName_set_matrix[indices[0]][indices[1]]) 
        else:
            self.trueVector.append(Qij)

        return tmp


    def getTrueAndFalseVectors(self, index : int, threshold : float, simMatrix : List[List[float]]) -> Tuple[List[str], List[float]]:

        rList = []
        pathToRemove = []
        for j in range(0, len(self.meta_set[0])):
            Eij = [*self.meta_set(index, j)]
            distFalse, distTrue = getSimilarityDistributions(simMatrix[j], threshold)
            r = getR(distFalse, distTrue)
            rList.append(r)
            pathToRemove = self.decideOnImageType((index, j), Eij[1][0], r, pathToRemove)

        return pathToRemove, rList


    def showSeparation(self):

        if self.falseVector != []:
            falseTensor = stack(self.falseVector)
            plot_images(falseTensor, "Images To Clean", images_per_row=len(self.falseVector))

        if self.trueVector != []:
            trueTensor = stack(self.trueVector)
            plot_images(trueTensor, "Images to keep", images_per_row=len(self.trueVector))


    def cleanSets(self):

        print("\n note that the cleaning process stops at the cumulative similarity matrix stage for now")

        rows, columns = self.meta_set.lenght()    
        for i in range(0, rows):

            simMatrix = []
            self.falseVector, self.trueVector = [], []
            for j in range(0, columns):

                Eij = [*self.meta_set(i, j)]
                preds = self.getPredictionsForOneQuery(Eij, i)
                simMatrix.append(preds[:j]+[1]+preds[j:])
                printSimMatrix(simMatrix)
                
                if False:
                    flattenedMatrix = list(itertools.chain(*simMatrix))
                    m, v = stats.norm.fit(flattenedMatrix)

                    print("threshold is", m-v)
                    pathToRemove, rList = self.getTrueAndFalseVectors(i, m-v, simMatrix)
                    self.showSeparation()

        return simMatrix
