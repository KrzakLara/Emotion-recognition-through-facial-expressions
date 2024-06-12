import tensorflow as tf
from sklearn.metrics import classification_report # type: ignore
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3 # type: ignore

from data_preparation import load_and_preprocess_data

def create_pretrained_model(model_name, num_classes):
    if model_name == 'VGG16':
        base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_name == 'ResNet50':
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError("Model not recognized")

    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, validation_dataset):
    y_true = []
    y_pred = []
    for images, labels in validation_dataset:
        predictions = model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    return classification_report(y_true, y_pred, target_names=validation_dataset.class_names)

train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'
train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path)

# List of state-of-the-art models to compare
model_names = ['VGG16', 'ResNet50', 'InceptionV3']
num_classes = 7

for model_name in model_names:
    print(f"Evaluating model: {model_name}")
    model = create_pretrained_model(model_name, num_classes)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=2)
    report = evaluate_model(model, validation_dataset)
    print(report)
