from .imports import os, np, tf, load_model, preprocessing, preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_textual_features(img_path:str, model_path:str) -> np.ndarray:
    """
    Returns the textual embeddings of the image using the mapping model
    """
    model = load_model(model_path)

    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_data = preprocessing.image.img_to_array(img)
    img_data = tf.keras.applications.resnet50.preprocess_input(img_data)

    return model.predict(img_data[None])

if __name__ == '__main__':
    print(generate_textual_features("./images/009.jpeg", "./models/general_mapping_model.model"))


