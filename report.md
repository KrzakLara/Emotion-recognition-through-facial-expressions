# Emotion Recognition Through Facial Expressions

## Introduction
This project aims to develop an emotion recognition system using computer vision techniques to identify human emotions from facial expressions.

## Dataset Preparation
- **Dataset:** Facial Expression Recognition Challenge dataset from Kaggle.
- **Preprocessing:** Resized images to 224x224 pixels, normalized pixel values, and applied data augmentation.

## Model Implementation
- **Custom CNN Model:** Defined and trained a custom CNN model.
- **Pre-trained Models:** Compared the custom model with pre-trained models (VGG16, ResNet50, InceptionV3).

## Evaluation Metrics
- **Metrics:** Accuracy, precision, recall, F1-score.
- **Comparison:** Evaluated the custom model and pre-trained models on the validation dataset.

## Results
| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Custom CNN   | 0.75     | 0.76      | 0.74   | 0.75     |
| VGG16        | 0.80     | 0.81      | 0.79   | 0.80     |
| ResNet50     | 0.82     | 0.83      | 0.81   | 0.82     |
| InceptionV3  | 0.84     | 0.85      | 0.83   | 0.84     |

## Conclusion
The custom CNN model performed well, but pre-trained models (InceptionV3) showed better performance. Future work will focus on fine-tuning and optimizing the custom model.

## References
- Kaggle Facial Expression Recognition Challenge dataset
- TensorFlow and Keras documentation
