import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
from model import create_model

# Set environment variables to suppress warnings and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Only show INFO level logs and higher
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

# Disable device placement logging
tf.debugging.set_log_device_placement(False)


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
        model = create_model(7)
        print("Model created successfully.")

        # Start model training
        print("Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=2  # Train for 2 epochs
        )
        print("Model training completed.")

        # Print the accuracy and loss for each epoch
        for epoch in range(2):
            print(f"Epoch {epoch + 1}/{2}:")
            print(f" - Accuracy: {history.history['accuracy'][epoch]}")
            print(f" - Loss: {history.history['loss'][epoch]}")
            print(f" - Validation Accuracy: {history.history['val_accuracy'][epoch]}")
            print(f" - Validation Loss: {history.history['val_loss'][epoch]}")

        # Save the trained model
        model.save('emotion_recognition_model.keras')
        print("Model saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
