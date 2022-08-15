from .imports import *

class Util:
    def __init__(self, path):
        self.path = path
        self.directory_classes = [os.listdir(self.get_path(os.listdir(self.path)[i])) for i in range(len(os.listdir(self.path)))]
          
    def get_path(self, _type : str):
        return '../input/996-imagenet/'+_type+'/data1/'+_type
    
    def get_data(self) -> Tuple[List[str], List[str]]:
        """
        Read images and labels from a folder
        """
        images_path = []
        labels = []
        absolute_path = os.listdir(self.path)
        for i in range(len(self.directory_classes)):
            for current_class in tqdm(self.directory_classes[i]):
                pth = os.path.join(self.get_path(absolute_path[i]), current_class)
                for dirname, _, filenames in os.walk(pth):
                    for file in filenames:
                        images_path.append(os.path.join(dirname, file))
                        labels.append(current_class)
        
        return images_path, current_class
    
    def get_all_classes(self) -> List[str]:
        """
        Returns the different classes of the images
        """
        classes = list(itertools.chain.from_iterable(list(self.directory_classes)))
        return [x.lower() for x in classes]
    
    def get_indexes(self, class_name : str, labels : List[str]) -> List[int]:
        """return indexes of class_name in labels"""
        try:
            return [index for index in range(len(labels)) if labels[index] == class_name]
        except ValueError:
            print("That item does not exist")
    
    def get_label_from_fname(self, fname:str):
        """
        Returns the name of an image's class given its path
        """
        fname = fname.split('/')

        index = fname.index('data1') + 2
        word = fname[index]
        return word
    
    def get_mean_visual_embeddings(self, model) -> dict:
        """computes the mean of the visual embeddigns of each class in the dataset"""
        features = dict()
        absolute_path = os.listdir(self.path)
        
        for i in range(len(self.directory_classes)):
            for clse in tqdm(self.directory_classes[i]):
                mean_list = []
                pth = os.path.join(self.get_path(absolute_path[i]), clse)
                for dirname, _, filenames in os.walk(pth):
                    for file in filenames:
                        image = preprocessing.image.load_img(os.path.join(dirname, file), target_size=(224, 224))
                        image = preprocessing.image.img_to_array(image)
                        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                        image = preprocess_input(image)
                        mean_list.append(model.predict(image, verbose=0))

                features[clse] = np.mean(mean_list, axis=0) 
        
        return features
                                               
    def get_mean_textual_embeddings(self, model) -> dict:
        """computes the mean of the textual embeddigns generated
           by the mapping model given in the parameter of each class in the dataset"""
        features = dict()
        absolute_path = os.listdir(self.path)

        for i in range(len(self.directory_classes)):
            for clse in tqdm(self.directory_classes[i]):
                tensor_list = []
                pth = os.path.join(self.get_path(absolute_path[i]), clse)
                for dirname, _, filenames in os.walk(pth):
                    for file in filenames:
                        img = preprocessing.image.load_img(os.path.join(dirname, file), target_size=(224, 224))
                        img_data = preprocessing.image.img_to_array(img)
                        img_data = preprocess_input(img_data)
                        vec = model.predict(img_data[None])
                        totest = torch.FloatTensor(vec.reshape(-1))
                        tensor_list.append(totest)
                features[clse] = torch.mean(torch.stack(tensor_list), dim=0)

        return features
    
    def get_visual_embeddings(self, img_path : str) -> np.ndarray:
        """
        Returns the visual embeddings of an image
        given its path using the ResNet50 model
        """
        model = ResNet50(weights="imagenet", include_top=True)
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        
        image = preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        return model.predict(image, verbose=0)