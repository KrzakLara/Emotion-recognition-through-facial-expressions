import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model(model_path, validation_data_path):
    model = tf.keras.models.load_model(model_path)
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_data_path,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )

    y_true = []
    y_pred = []
    for images, labels in validation_dataset:
        predictions = model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    return classification_report(y_true, y_pred, target_names=validation_dataset.class_names)

def create_model(num_classes):
    input_layer = tf.keras.Input(shape=(224, 224, 3))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    x = data_augmentation(input_layer)

    x = tf.keras.layers.Rescaling(1. / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
