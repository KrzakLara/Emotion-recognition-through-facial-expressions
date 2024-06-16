import os
import random
import shutil

# Define the data paths
train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'
rebalanced_train_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_train'
rebalanced_validation_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/rebalanced_validation'

# Function to rebalance data
def rebalance_data(source_dir, target_dir, exact_count):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for class_dir in os.listdir(source_dir):
        source_class_dir = os.path.join(source_dir, class_dir)
        target_class_dir = os.path.join(target_dir, class_dir)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        files = os.listdir(source_class_dir)
        if len(files) >= exact_count:
            files = random.sample(files, exact_count)
        else:
            raise ValueError(f"Not enough images in {class_dir} to meet the exact count of {exact_count}")

        for file in files:
            source_file = os.path.join(source_class_dir, file)
            target_file = os.path.join(target_class_dir, file)
            shutil.copy(source_file, target_file)

# Exact count for train and validation datasets
train_exact_count = min([len(os.listdir(os.path.join(train_data_path, class_dir))) for class_dir in os.listdir(train_data_path)])
validation_exact_count = min([len(os.listdir(os.path.join(validation_data_path, class_dir))) for class_dir in os.listdir(validation_data_path)])

# Rebalance train and validation datasets
rebalance_data(train_data_path, rebalanced_train_path, train_exact_count)
rebalance_data(validation_data_path, rebalanced_validation_path, validation_exact_count)

# Display the number of images in each class after rebalancing
def display_counts(data_path):
    print(f"Image counts in {data_path}:")
    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        num_images = len(os.listdir(class_path))
        print(f"{class_dir}: {num_images} images")

print("Rebalanced Training Data Counts:")
display_counts(rebalanced_train_path)

print("\nRebalanced Validation Data Counts:")
display_counts(rebalanced_validation_path)
