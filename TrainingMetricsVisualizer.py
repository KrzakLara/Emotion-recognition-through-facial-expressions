import matplotlib.pyplot as plt

class TrainingMetricsVisualizer:
    def __init__(self, history):
        self.history = history

    def plot_metrics(self):
        epochs = range(len(self.history['train_loss']))

        plt.figure(figsize=(18, 6))

        # Plot training & validation loss values
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training & validation accuracy values
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training & validation F1-score values
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['train_f1'], label='Train F1-Score')
        plt.plot(epochs, self.history['val_f1'], label='Validation F1-Score')
        plt.title('Model F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.legend()

        plt.tight_layout()
        plt.show()
