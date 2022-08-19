from .Util import Util
from .imports import *

class SuperClassBalancer:
    """
    This class allows to balance a given dataset and equally distributing data on each class.
    It is used specifically for the super classes.

    Attributes
    ----------
    path :
        the path for the dataset. This depends on how the dataset is saved
    super_classes:
        A list of the super classes choosed
    """
    def __init__(self, path : str, super_classes : List[str]):
        self.util = Util(path)
        self.super_classes = super_classes
    
    def balance(self, samples_number : int, labels):
        """Return indexes equaly distributed between each label"""
        self.balanced_indexes = dict()
        for super_class in tqdm(self.super_classes):
            self.balanced_indexes[super_class] = random.Random(0).sample(self.util.get_indexes(super_class, labels), samples_number)
        
        return list(itertools.chain.from_iterable(list(self.balanced_indexes.values())))
            
    def add_value_label(self, x_list, y_list):
        """allows to write text on a given plot"""
        for i in range(len(x_list)):
            plt.text(i, y_list[i], y_list[i], ha="center", fontweight='bold', fontsize="medium")
    
    def plot_distribution(self):
        """Plot a histogram presenting the number of samples on each class"""
        super_class_distribution = dict()
        for super_class in tqdm(self.super_classes):
            super_class_distribution[super_class] = len(self.balanced_indexes[super_class])
        
        plt.figure(figsize=(20,8))
        plt.bar(super_class_distribution.keys(), super_class_distribution.values(), align='center', width=0.5, color='g')
        self.add_value_label(list(super_class_distribution.keys()), list(super_class_distribution.values()))