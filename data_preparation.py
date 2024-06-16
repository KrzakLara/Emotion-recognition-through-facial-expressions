import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress all TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Additional logging suppression
import logging
logger = tf.get_logger()
logger.disabled = True
logger.propagate = False

def load_and_preprocess_data(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset

def display_sample_images(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[tf.argmax(labels[i])])
            plt.axis("off")
    plt.show()

if __name__ == "__main__":
    train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_train'
    validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_validation'

    # Load and preprocess datasets
    train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path)

    # Display sample images from the training dataset
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    display_sample_images(train_dataset, class_names)
