import tensorflow as tf
import os

def load_and_rebalance_data(data_path, image_size=(224, 224), batch_size=32, max_samples_per_class=500):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_datasets = {}
    for images, labels in dataset:
        for i, label in enumerate(labels):
            class_name = dataset.class_names[tf.argmax(label).numpy()]
            if class_name not in class_datasets:
                class_datasets[class_name] = []
            if len(class_datasets[class_name]) < max_samples_per_class:
                class_datasets[class_name].append((images[i], labels[i]))

    balanced_datasets = []
    for class_name, samples in class_datasets.items():
        balanced_datasets.append(tf.data.Dataset.from_tensor_slices(samples))

    balanced_dataset = tf.data.experimental.sample_from_datasets(balanced_datasets)
    return balanced_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

if __name__ == "__main__":
    train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
    validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

    train_dataset = load_and_rebalance_data(train_data_path)
    validation_dataset = load_and_rebalance_data(validation_data_path)

    train_dataset = train_dataset.map(augment)

    model_path = 'C:/Users/larak/Desktop/ORV/CVProject/emotion_recognition_model_enriched.keras'
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_dataset, epochs=2, validation_data=validation_dataset)

    y_true = []
    y_pred = []
    for images, labels in validation_dataset:
        predictions = model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=validation_dataset.class_names))
