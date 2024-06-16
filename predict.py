import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Define paths
new_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/new_predict_dataset'
model_path_resnet = 'C:/Users/larak/Desktop/ORV/CVProject/best_model.keras'
model_path_vgg16 = 'C:/Users/larak/Desktop/ORV/CVProject/best_model_vgg16.keras'
model_path_train = 'C:/Users/larak/Desktop/ORV/CVProject/emotion_recognition_model.keras'
model_path_train_enriched = 'C:/Users/larak/Desktop/ORV/CVProject/emotion_recognition_model_enriched.h5'

# Load the models
model_resnet = tf.keras.models.load_model(model_path_resnet)
model_vgg16 = tf.keras.models.load_model(model_path_vgg16)
model_train = tf.keras.models.load_model(model_path_train)
model_train_enriched = tf.keras.models.load_model(model_path_train_enriched)

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load all images and labels
images = []
true_labels = []
class_names = sorted(os.listdir(new_data_path))
for class_name in class_names:
    class_dir = os.path.join(new_data_path, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        images.append(load_and_preprocess_image(img_path))
        true_labels.append(class_names.index(class_name))

images = np.vstack(images)
true_labels = np.array(true_labels)

# Predict using ResNet model
print("Predicting with ResNet model...")
preds_resnet = model_resnet.predict(images)
pred_classes_resnet = np.argmax(preds_resnet, axis=1)

# Predict using VGG16 model
print("Predicting with VGG16 model...")
preds_vgg16 = model_vgg16.predict(images)
pred_classes_vgg16 = np.argmax(preds_vgg16, axis=1)

# Predict using Train model
print("Predicting with Train model...")
preds_train = model_train.predict(images)
pred_classes_train = np.argmax(preds_train, axis=1)

# Predict using Train Enriched model
print("Predicting with Train Enriched model...")
preds_train_enriched = model_train_enriched.predict(images)
pred_classes_train_enriched = np.argmax(preds_train_enriched, axis=1)

# Function to plot images with predictions
def plot_predictions(images, true_labels, pred_labels_resnet, pred_labels_vgg16, pred_labels_train, pred_labels_train_enriched, class_labels):
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(2, len(images) // 2, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_labels[true_labels[i]]}\n"
                  f"ResNet: {class_labels[pred_labels_resnet[i]]}\n"
                  f"VGG16: {class_labels[pred_labels_vgg16[i]]}\n"
                  f"Train: {class_labels[pred_labels_train[i]]}\n"
                  f"Train Enriched: {class_labels[pred_labels_train_enriched[i]]}")
        plt.axis('off')
    plt.show()

# Plot predictions
plot_predictions(images, true_labels, pred_classes_resnet, pred_classes_vgg16, pred_classes_train, pred_classes_train_enriched, class_names)

print("Prediction complete.")
