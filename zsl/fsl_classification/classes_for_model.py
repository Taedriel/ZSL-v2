from zsl.fsl_classification.classes_for_dataset import MetaSet
from zsl.fsl_classification.siamese_network import Siamese
from .constants import *
from .utils_sn import *
from .utils import plot_images, get_similarity_distributions, get_r, print_similarity_matrix

from torch import stack, tensor, save, no_grad
from torch.nn import BCELoss

from random import randint
from tqdm import tqdm
import itertools
from scipy import stats


def get_indexes(set_of_images : Tensor, nb_classes : int, cleaning=False) -> Tuple[int, int, int, int]:

    nbEx =  len(set_of_images) if cleaning else len(set_of_images[0])
    ci1, e1 = randint(0, nb_classes-1), randint(0, nbEx-1) 
    ci2, e2 = randint(0, nb_classes-1), randint(0, nbEx-1)

    return ci1, ci2, e1, e2



def get_random_pair(set_of_images : Tensor, cleaning=False) -> Tuple[Tensor, Tensor, int]:
    """
    the cleaning parameter is here because cleaning data and training data have different shapes
    """

    nbClasses = len(set_of_images)
    ci1, ci2, e1, e2 = get_indexes(set_of_images, nbClasses, cleaning)

    I1, I2, s = 0, 0, 0
    if cleaning:
        c1, c2 = set_of_images[e1][1], set_of_images[e2][1]
        I1, I2 = stack([set_of_images[e1][0]]), stack([set_of_images[e2][0]])
        s = 1.0 if c1 == c2 else 0.0
    else:
        I1, I2 = stack([set_of_images[ci1][e1][0]]), stack([set_of_images[ci2][e2][0]])
        s = 1.0 if ci1 == ci2 else 0.0

    return I1, I2, s



class Trainer(ModelUtils):
    """
    Class implementing the training of the model after the cleaning process
    """

    def __init__(self, path_model : str, model : Siamese, cuda_ : bool, cleaning : bool):

        super().__init__()
        self.cuda_ = cuda_
        self.cleaning_phase = cleaning
        self.model = model
        self.optimizer = super().get_optimizer(self.model.metric)
        self.LOSS = BCELoss()

        self.model_saver = ModelSaver(path_model)


    def reset_model(self, reset_by_param=False):

        if not reset_by_param:
            model_name = super().get_model_name("train", self.cuda_)
            _, model, opti = self.model_saver.load_model(model_name, self.model.metric, self.optimizer)
            self.model.metric, self.optimizer = model, opti
        else:
            self.model.metric = super().reset_model_param(self.model.metric)
            self.optimizer = super().get_optimizer(self.model.metric)


    def epoch(self, support_set : Tensor) -> List[float]:

        loss_for_batch = []
        for i in range(0, batch_size):

            I1, I2, s = get_random_pair(support_set, self.cleaning_phase)
            I1, I2 = super().get_images(I1, I2)
            w, u, v = self.model.create_combined_vector(I1, I2)
            out = self.model(w)
            s = tensor([s]).cuda() if self.cuda_ else tensor([s])

            self.optimizer.zero_grad()
            loss = self.LOSS(out, s)
            loss.backward()
            self.optimizer.step()
            loss_for_batch.append(loss.item())

        return loss_for_batch


    
    def training(self, support_set : Tensor, epoch_loss=(0, 0), set_i=0) -> List[int]:
        """
        Training of the model

        Parameters
        ----------
        support_set :
            the set of images as created by getSets
        epoch_loss :
            used if the training is to resumed
        set_i :
            used for tqdm (show on which class we are training)

        Return
        ------
        losses a list of the mean loss per epoch
        """

        number_of_epochs = 300-epoch_loss[0]
        validation_frequence = 10

        losses = [epoch_loss[1]] if epoch_loss[1] != 0 else []

        self.model.train()
        for epoch_i in tqdm(range(0, number_of_epochs), desc="Traning on Set "+str(set_i)):

            batch_loss = self.epoch(support_set)
            mean_loss = sum(batch_loss)*1.0/len(batch_loss)
            losses.append(mean_loss)

            if epoch_i % validation_frequence == 0 and epoch_i != 0:
                model_opti =  [self.model.metric, self.optimizer]
                self.model_saver.save_model("SNTrain.pt", model_opti, epoch_i, mean_loss)

        save(self.model.metric.state_dict(), PATH_MODEL+"SN.pt")

        return losses



class Tester(ModelUtils):
    """
    Class implementing the testing part of the pipeline after the cleaning process

    This class implement both the testing part and stand-alone testing
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    
    def get_first_class_based_on_representation(self, predictions : List[Tuple[Tensor, int]]) -> int:
        """
        get the class that is the most represented in a list of predictions

        Parameters
        ----------
        predictions :
            a list of prediction with (pred, label) format

        Return
        ------
        the label of the most represented class
        """

        representation = [0]*N_WAY
        for pred in predictions:
            representation[pred[0]] += 1

        return representation.index(max(representation))


    def is_model_correct(self, predictions : List[Tuple[Tensor, int]], query_class : int) -> Tuple[int, int, float]:

        pred_sorted = sorted(predictions, key=lambda tup: tup[-1], reverse=True)
        first_five = pred_sorted[0:5]

        predicted_class_label = self.get_first_class_based_on_representation(first_five)
        similarity = int(query_class == predicted_class_label)
        prediction_score = first_five[0][1]

        return similarity, predicted_class_label, prediction_score


    def model_prediction(self, query_info : Tuple[Tensor, int], image_info : Tuple[Tensor, int], triplets : List[Tuple[Tensor, int]]) -> Tuple[List[Tuple[Tensor, int]], int, int, float]:

        image, image_class = image_info
        query, query_class = query_info

        query, image = super().get_images(query, image)
        w, u, v = self.model.create_combined_vector(query, image)
        p = self.model(w)
        triplets.append((image_class, p))

        are_really_similar, image_class, prediction = self.is_model_correct(triplets, query_class)

        return triplets, are_really_similar, image_class, prediction


    def evaluate_with_metric(self, support_set : Tensor, query_set : Tensor):

        triplets = []
        pred_labels = []
        query_labels = []
        correct_predictions, incorrect_predictions, index_incorrect_query = [], [], []
        correct = 0

        self.model.eval()
        with no_grad():

            for index_query, query_info in enumerate(list(itertools.chain(*query_set))):

                _, queryClass = query_info
                for image_info in list(itertools.chain(*support_set)):
                    triplets, are_really_similar, image_class, prediction = self.model_prediction(query_info, image_info, triplets)
                    
                if are_really_similar == 1: 
                    correct+=1
                    correct_predictions.append(prediction)
                else:
                    incorrect_predictions.append(prediction)
                    index_incorrect_query.append(index_query)

                triplets = []
                pred_labels.append(image_class)
                query_labels.append(queryClass)

        return "\n accuracy :"+str(100.0*correct/(N_WAY*N_QUERY)), pred_labels, query_labels, correct_predictions, incorrect_predictions, index_incorrect_query
        

    def query_evaluation(self, support_set : Tensor, query : Tensor) -> int:

        triplets = []
        self.model.eval()
        with no_grad():

            for imageInfo in list(itertools.chain(*support_set)):

                image, imageClass = imageInfo
                query, image = super().get_images(query, image)
                w, _, _ = self.model.create_combined_vector(query, image)
                p = self.model(w)
                triplets.append((imageClass, p))
            
        pred_sorted = sorted(triplets, key=lambda tup: tup[-1], reverse=True)
        first_five = pred_sorted[0:5]
        predicted_class_label = self.get_first_class_based_on_representation(first_five)

        return predicted_class_label



class Cleaner(ModelUtils):
    """
    class implementing the cleaning part of the pipeline

    It uses the MetaSet class to create all of the cleaning set possible by extracting one image at a time
    """

    def __init__(self, path_model : str, model : Siamese, meta_set : MetaSet, cuda_ : bool):
        super().__init__()

        self.meta_set = meta_set
        self.threshold = 0.5

        self.model_t = Trainer(path_model, model, cuda_, True)
        self.model_saver = ModelSaver(path_model)

        self.false_vector = []
        self.true_vector = []

        self.cuda_ = cuda_


    def is_query_outsider(self, Eij : Tuple[Tensor, Tensor, int]) -> float:

        model = self.model_t.model
        support_set, query_set, lenght = Eij[0], Eij[1], Eij[2]-1
        predictions = []

        model.eval()
        with no_grad():

            query_raw = query_set[0]
            for image_raw, image_class in support_set[:lenght]:

                query, image = self.model_t.get_images(query_raw, image_raw)
                w = model.create_combined_vector(query, image)
                p = model(w)
                res = round(p.cpu().item(), 2)
                predictions.append(res)
            
        return predictions


    #TODO CAN CHANGE THE RELOADING MODEL HERE(cleaning context)
    def get_predictions_for_one_query(self, Eij : Tuple[Tensor, Tensor, int], index_set : int) -> float:

        model_name = super().get_model_name("clean", self.cuda_)
        _, model_, opti = self.model_saver.load_model(model_name, self.model_t.model.metric, self.model_t.optimizer)
        self.model_t.model.metric, self.model_t.optimizer = model_, opti

        self.model_t.training(Eij[0], set_i=index_set) 
        preds = self.is_query_outsider(Eij)

        return preds


    def decide_on_image_type(self, indices : int, Qij : Tensor, r : float, image_to_clean : List[str]) -> List[str]:
        """
        decide if the query is to be kept or not based on the r parameter

        Parameters
        ----------
        indices :
            the position of the image in the matrix
        Qij :
            the query
        r :
            the r parameter has calculated by get_r()
        imageToClean :
            a list of images

        Return
        ------
        the updated list of images to remove
        """

        tmp = image_to_clean
        if 0 <= r < 1:
            self.false_vector.append(Qij)
            tmp.append(self.meta_set.imageName_set_matrix[indices[0]][indices[1]]) 
        else:
            self.true_vector.append(Qij)

        return tmp


    def get_true_and_false_vectors(self, index : int, threshold : float, similiarity_matrix : List[List[float]]) -> Tuple[List[str], List[float]]:

        rList = []
        pathToRemove = []
        for j in range(0, len(self.meta_set[0])):
            Eij = [*self.meta_set(index, j)]
            false_distribution, true_distribution = get_similarity_distributions(similiarity_matrix[j], threshold)
            r = get_r(false_distribution, true_distribution)
            rList.append(r)
            pathToRemove = self.decide_on_image_type((index, j), Eij[1][0], r, pathToRemove)

        return pathToRemove, rList


    def show_separation(self):

        if self.false_vector != []:
            false_tensor = stack(self.false_vector)
            plot_images(false_tensor, "Images To Clean", images_per_row=len(self.false_vector))

        if self.true_vector != []:
            true_tensor = stack(self.true_vector)
            plot_images(true_tensor, "Images to keep", images_per_row=len(self.true_vector))


    def clean_sets(self):

        print("\n note that the cleaning process stops at the cumulative similarity matrix stage for now")

        rows, columns = self.meta_set.lenght()    
        for i in range(0, rows):

            similarity_matrix = []
            self.false_vector, self.true_vector = [], []
            for j in range(0, columns):

                Eij = [*self.meta_set(i, j)]
                preds = self.get_predictions_for_one_query(Eij, i)
                similarity_matrix.append(preds[:j]+[1]+preds[j:])
                print_similarity_matrix(similarity_matrix)
                
                if False:
                    flattened_matrix = list(itertools.chain(*similarity_matrix))
                    m, v = stats.norm.fit(flattened_matrix)

                    print("threshold is", m-v)
                    path_to_remove, rList = self.get_true_and_false_vectors(i, m-v, similarity_matrix)
                    self.show_separation()

                    remove(path_to_remove) # function dosen't exist

        return similarity_matrix
