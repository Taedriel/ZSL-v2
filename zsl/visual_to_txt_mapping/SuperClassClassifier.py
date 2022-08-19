from .imports import tqdm, tf, preprocess_input, preprocessing, ResNet50
from .SuperClassHandler import SuperClassHandler

class SuperClassClassifier:
    """
    his class presents a classifier that classify super classes. 
    It uses the second approach explained in the report
    """
    def __init__(self, super_class_handler : SuperClassHandler):
        self.model = ResNet50(weights='imagenet')
        self.super_class_handler = super_class_handler
    
    def predict(self, image_path):
        """
        predicts the label of the image given its path
        """
        img = preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_data = preprocessing.image.img_to_array(img)
        img_data = preprocess_input(img_data)
        prediction = self.model.predict(img_data[None])
        label = tf.keras.applications.imagenet_utils.decode_predictions(prediction) 
        return self.super_class_handler.get_super_class(label[0][0][1])
    
    def predict_on_samples(self, image_paths):
        """
        predicts the labels of the images given their path
        """
        predictions = []
        for i in tqdm(range(len(image_paths))):
            img = preprocessing.image.load_img(image_paths[i], target_size=(224, 224))
            img_data = preprocessing.image.img_to_array(img)
            img_data = preprocess_input(img_data)
            prediction = self.model.predict(img_data[None])
            label = tf.keras.applications.imagenet_utils.decode_predictions(prediction) 
            predictions.append(self.super_class_handler.get_super_class(label[0][0][1]))
        return predictions