from .imports import np, pd, List

class SuperClassHandler:
    """
    This class presents different functions that are useful when working with super classes
    
    Attributes
    ----------
    super_class_csv :
        A csv file containing the super classes and their inner classes
    """
    def __init__(self, super_class_csv : pd.DataFrame):
        self.super_class_df = pd.read_csv(super_class_csv)
        self.super_classes = self.super_class_df.iloc[:, 0].tolist()
    
    def get_classes(self, super_class : str) -> List[str]:
        """Take a super class as parameter and returns its correspondent classes"""
        try:
            return [i.replace("_", " ").lower() for i in self.super_class_df[self.super_class_df.iloc[:, 0] == super_class].to_numpy().tolist()[0][1:] if not(pd.isnull(i)) == True]
        except:
            print('Failed to find the super class')
        
    def get_super_class(self, class_name : str) -> str:
        """return the super class of the class given in the parameter"""
        for super_class in self.super_classes:
            if class_name.replace("_", " ").lower() in self.get_classes(super_class):
                return super_class
        return None
    
    def get_super_classes(self) -> List[str]:
        """returns a list containing all super classes"""
        return self.super_classes