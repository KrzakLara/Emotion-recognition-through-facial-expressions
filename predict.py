import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Emotion labels corresponding to your 7 classes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def prepare_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_emotion(model, img_path):
    img = prepare_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return emotion_labels[predicted_class], prediction

if __name__ == "__main__":
    model_paths = [
        'emotion_recognition_model.keras',
        'emotion_recognition_model_enriched.keras',
        'emotion_recognition_model_vgg16.keras'
    ]

    img_path = 'test_image.jpg'

    for model_path in model_paths:
        print(f"Loading model from: {model_path}")
        try:
            model = load_model(model_path)
            emotion, prediction = predict_emotion(model, img_path)
            print(f"Predicted emotion: {emotion}")
            print(f"Prediction confidence: {prediction}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
