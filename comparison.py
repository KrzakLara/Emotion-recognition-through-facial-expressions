import tensorflow as tf
from sklearn.metrics import classification_report # type: ignore

def evaluate_model(model_path, validation_data_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_data_path,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )

    y_true = []
    y_pred = []
    for images, labels in validation_dataset:
        predictions = model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    return classification_report(y_true, y_pred, target_names=validation_dataset.class_names)

# Paths to your models and validation data
model_paths = ['emotion_recognition_model.keras', 'emotion_recognition_model_vgg16.keras']
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

for model_path in model_paths:
    print(f"Evaluating model: {model_path}")
    report = evaluate_model(model_path, validation_data_path)
    print(report)
