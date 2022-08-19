from .imports import *
from .DataLoader import ImageGenerator
from .Solver import Solver

class Learner():
    """Base learner object"""
    def __init__(self):
        pass
            
    @classmethod
    def freeze(cls, model, limit=None):
        """freeze all layers of the model (from left to right)"""
        # handle negative indices
        if limit != None and limit < -1:
            limit += len(model.layers) 
        # loop for all valid indices and mark the corresponding layer
        for index, layer in enumerate(model.layers):
            if limit != None and index > limit:
                break
            layer.trainable = False

    @classmethod
    def unfreeze(cls, model, limit=None):
        """unfreeze all layers of the model up to the given layer index (from right to left)"""
        # handle negative indices
        if limit != None and limit < -1:
            limit += len(model.layers)
        for index, layer in enumerate(model.layers):
            if limit != None and index < limit:
                continue
            layer.trainable = True


class ZeroShotLearner(Learner):
    """
    Mapping model
    
    Attributes
    ----------
    data :
        Data generated by The ImageGenerator class, It can train, validation or test data
    loss :
        The loss function to train the neural networks
    metrics :
        List of metrics to observe the performance of the model while training
    """
    def __init__(self, data : ImageGenerator, loss=tf.keras.losses.CosineSimilarity(axis=1), metrics=['accuracy']):
        self.data = data
        self.model = self.create_model()
        adam = Adam(learning_rate=0.001, epsilon=0.01, decay=0.0001)
        self.model.compile(adam, loss, metrics)
        
    
    def create_model(self):
        base_model = ResNet50(weights='imagenet')
        Learner.freeze(base_model, -3)
        
        x = base_model.layers[-3].output          # shape = (bs=None, 7, 7, 2048)
        x = Dropout(rate=0.3)(x)                  # shape = (bs=None, 7, 7, 2048)
        x = GlobalAveragePooling2D()(x)           # shape = (bs=None, 2048)
        x = Dense(1024, activation='relu')(x)     # shape = (bs=None, 1024)
        x = BatchNormalization()(x)
        y = Dense(self.data.classes_size, activation='linear')(x)
         
        return Model(inputs=base_model.input, outputs=y)
        
    def fit(self, epochs=10):
        history = self.model.fit(self.data.train_data, validation_data=self.data.validation_data, epochs=epochs)
        return history
    
    def predict_on_one_sample(self, image_path : str, solver : Solver) -> List[str]:
        """
        Predict the label of the image given its path and using the solver
        to get the nearest labels
        """
        img = preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_data = preprocessing.image.img_to_array(img)
        img_data = preprocess_input(img_data)
        vec = self.model.predict(img_data[None])
        totest = torch.FloatTensor(vec.reshape(-1))
        return solver.get_nearest_embedding_of(totest)[0][0]
    
    def predict_on_samples(self, image_paths : List[str], solver : Solver) -> List[List[str]]:
        """
        Predict the label of the images given their path and using the solver
        to get the nearest labels
        """
        y_pred = []
        for i in tqdm(range(len(image_paths))):
            img = preprocessing.image.load_img(image_paths[i], target_size=(224, 224))
            img_data = preprocessing.image.img_to_array(img)
            img_data = preprocess_input(img_data)
            vec = self.model.predict(img_data[None])
            totest = torch.FloatTensor(vec.reshape(-1))
            y_pred.append(solver.get_nearest_embedding_of(totest)[0][0])
        return y_pred
    
    def save_model(self, model_name : str):
        self.model.save(model_name+".model", save_format="h5")