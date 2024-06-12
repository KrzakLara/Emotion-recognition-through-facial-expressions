import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
import warnings

# Suppress TensorFlow logging warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Set logging to INFO level
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use only CPU
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

tf.debugging.set_log_device_placement(False)

def print_dataset_info(dataset, dataset_name):
    print(f"{dataset_name} dataset:")
    batch_count = 0
    for images, labels in dataset:
        try:
            # Ensure the batch has the expected number of samples
            if images.shape[0] != 8:  # Your batch size is 8
                print(f"Skipping incomplete batch with {images.shape[0]} samples")
                continue
            print(f"Batch {batch_count + 1} - Images shape: {images.shape}, Labels shape: {labels.shape}")
            batch_count += 1
        except Exception as e:
            print(f"An error occurred in batch {batch_count + 1}: {e}")
            continue
    print(f"Total batches in {dataset_name}: {batch_count}")

if __name__ == "__main__":
    try:
        print("Loading and preprocessing data...")
        train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
        validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

        train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path, batch_size=8)
        print("Data loaded and preprocessed successfully.")

        # Print dataset information
        print_dataset_info(train_dataset, "Training")
        print_dataset_info(validation_dataset, "Validation")

        print("Datasets fully consumed.")

    except Exception as e:
        print(f"An error occurred: {e}")
