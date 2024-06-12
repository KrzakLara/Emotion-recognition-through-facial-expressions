import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set logging to WARNING level
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
import tensorflow as tf
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning


def evaluate_model(model_path, validation_data_path):
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)

        # Recompile the model
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

        # Suppress warnings for undefined metrics
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        report = classification_report(y_true, y_pred, target_names=validation_dataset.class_names, zero_division=0)
        return report

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")


if __name__ == "__main__":
    try:
        validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'
        model_paths = [
            'C:/Users/larak/Desktop/ORV/CVProject/emotion_recognition_model.keras']  # Only the existing model

        for model_path in model_paths:
            print(f"Evaluating model: {model_path}")
            report = evaluate_model(model_path, validation_data_path)
            print(report)
    except Exception as e:
        print(f"An error occurred: {e}")
