import matplotlib.pyplot as plt
import numpy as np
import os

# Define the path to the history file
history_path = 'history.npy'

# Check if the file exists
if not os.path.exists(history_path):
    print(f"Error: The file '{history_path}' does not exist. Please check the path and try again.")
else:
    # Load history
    history = np.load(history_path, allow_pickle=True).item()

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot F1-score values
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1-Score')
    plt.plot(history['val_f1'], label='Validation F1-Score')
    plt.title('Model F1-Score')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
