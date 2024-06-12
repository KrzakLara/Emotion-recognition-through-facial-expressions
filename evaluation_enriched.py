import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load the model saved in .keras format without the optimizer
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

# Define paths
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'
train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

# Define train and validation datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Apply augmentations to the training dataset
train_dataset = train_dataset.map(augment)

# Load and compile the model
model_path = 'C:/Users/larak/Desktop/ORV/CVProject/emotion_recognition_model_enriched.keras'
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model with data augmentation
model.fit(train_dataset, epochs=2, validation_data=validation_dataset)

# Evaluate the model
y_true = []
y_pred = []
for images, labels in validation_dataset:
    predictions = model.predict(images)
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(predictions, axis=1).numpy())

# Print classification report
print(classification_report(y_true, y_pred, target_names=validation_dataset.class_names))
