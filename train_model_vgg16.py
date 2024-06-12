import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
from model_vgg16 import create_vgg16_model
from TrainingMetricsVisualizer import TrainingMetricsVisualizer
import warnings
from sklearn.metrics import f1_score
import numpy as np

# Suppress TensorFlow logging warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Set environment variables to suppress warnings and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Only show INFO level logs and higher
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

# Disable device placement logging
tf.debugging.set_log_device_placement(False)


def calculate_f1_score(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='weighted')

def main():
    try:
        print("Loading and preprocessing data...")

        # Define paths to training and validation datasets
        train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
        validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

        # Load and preprocess datasets
        train_dataset, validation_dataset = load_and_preprocess_data(
            train_data_path, validation_data_path, batch_size=8
        )
        print("Data loaded successfully.")

        # Create the model with the specified number of classes
        model = create_vgg16_model(7)
        print("Model created successfully.")

        # Define callbacks to save the best model
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('best_model_vgg16.keras', save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

        # Start model training
        print("Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=2,  # Train for 2 epochs
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        print("Model training completed.")

        # Initialize history dictionary
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }

        # Iterate over epochs to collect metrics
        for epoch in range(2):
            train_acc = history.history['accuracy'][epoch]
            train_loss = history.history['loss'][epoch]
            val_acc = history.history['val_accuracy'][epoch]
            val_loss = history.history['val_loss'][epoch]

            # Calculate F1 scores
            train_f1 = calculate_f1_score(model.predict(train_dataset), train_dataset)
            val_f1 = calculate_f1_score(model.predict(validation_dataset), validation_dataset)

            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_acc'].append(val_acc)
            metrics_history['train_f1'].append(train_f1)
            metrics_history['val_f1'].append(val_f1)

            print(f"Epoch {epoch + 1}/2:")
            print(f" - Accuracy: {train_acc}")
            print(f" - Loss: {train_loss}")
            print(f" - Validation Accuracy: {val_acc}")
            print(f" - Validation Loss: {val_loss}")
            print(f" - Train F1: {train_f1}")
            print(f" - Validation F1: {val_f1}")

        # Save the trained model
        model.save('emotion_recognition_model_vgg16.keras')
        print("Model saved successfully.")

        # Visualize metrics
        visualizer = TrainingMetricsVisualizer(metrics_history)
        visualizer.plot_metrics()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
