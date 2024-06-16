import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
from model_vgg16 import create_vgg16_model
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO level logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Set environment variables to disable GPU and MKL optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

# Disable device placement logging
tf.debugging.set_log_device_placement(False)


def calculate_f1_score(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='weighted')


def plot_metrics(metrics_history):
    epochs = range(1, len(metrics_history['train_acc']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics_history['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics_history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics_history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics_history['train_f1'], label='Train F1-Score')
    plt.plot(epochs, metrics_history['val_f1'], label='Validation F1-Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_precision_recall(y_true, y_pred_probs, class_names):
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, marker='.', label=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset, validation_dataset, class_names):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.class_names = class_names
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }

    def on_epoch_end(self, epoch, logs=None):
        train_pred = np.concatenate([self.model.predict(x) for x, _ in self.train_dataset])
        train_true = np.concatenate([y for _, y in self.train_dataset])
        val_pred = np.concatenate([self.model.predict(x) for x, _ in self.validation_dataset])
        val_true = np.concatenate([y for _, y in self.validation_dataset])

        train_f1 = calculate_f1_score(train_true, train_pred)
        val_f1 = calculate_f1_score(val_true, val_pred)

        self.metrics_history['train_loss'].append(logs['loss'])
        self.metrics_history['val_loss'].append(logs['val_loss'])
        self.metrics_history['train_acc'].append(logs['accuracy'])
        self.metrics_history['val_acc'].append(logs['val_accuracy'])
        self.metrics_history['train_f1'].append(train_f1)
        self.metrics_history['val_f1'].append(val_f1)

        print(f"Epoch {epoch + 1}: Train F1: {train_f1}, Val F1: {val_f1}")

    def on_train_end(self, logs=None):
        np.save('metrics_history.npy', self.metrics_history)
        plot_metrics(self.metrics_history)

        # Plot Precision-Recall Curve
        y_true = np.concatenate([y for _, y in self.validation_dataset])
        y_pred_probs = np.concatenate([self.model.predict(x) for x, _ in self.validation_dataset])
        plot_precision_recall(y_true, y_pred_probs, self.class_names)

        # Plot Confusion Matrix
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        plot_confusion_matrix(y_true_labels, y_pred, self.class_names)


def main():
    try:
        print("Loading and preprocessing data...")

        # Define paths to training and validation datasets
        train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_train'
        validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_validation'

        # Load and preprocess datasets
        train_dataset, validation_dataset = load_and_preprocess_data(
            train_data_path, validation_data_path, batch_size=8
        )
        print("Data loaded successfully.")

        # Create the model with the specified number of classes
        model = create_vgg16_model(7)
        print("Model created successfully.")

        # Define callbacks
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('best_model_vgg16.keras', save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        f1_score_cb = F1ScoreCallback(train_dataset, validation_dataset, class_names)

        # Start model training
        print("Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=2,  # Train for 2 epochs
            callbacks=[checkpoint_cb, early_stopping_cb, f1_score_cb]
        )
        print("Model training completed.")

        # Save the trained model
        model.save('emotion_recognition_model_vgg16.keras')
        print("Model saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
