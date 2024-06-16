import os
import sys
import subprocess

# Ensure seaborn is installed
try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

import tensorflow as tf
from data_preparation import load_and_preprocess_data
from model import create_model
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow oneDNN and CPU instructions warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO level logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Set environment variables to disable GPU and MKL optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

# Disable device placement logging
tf.debugging.set_log_device_placement(False)

# Define class names
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_train'
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_validation'

train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path)
model = create_model(7)

def plot_metrics(history):
    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1'], label='Train F1-Score')
    plt.plot(epochs, history['val_f1'], label='Validation F1-Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
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

        train_f1 = f1_score(train_true, train_pred, average='weighted')
        val_f1 = f1_score(val_true, val_pred, average='weighted')

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

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=2,
    callbacks=[F1ScoreCallback(train_dataset, validation_dataset, class_names)]
)

model.save('emotion_recognition_model_enriched.h5')

# Save history
np.save('history.npy', history.history)

# Plot metrics
plot_metrics(history.history)
