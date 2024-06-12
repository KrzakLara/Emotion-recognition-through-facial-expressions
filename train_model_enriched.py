from data_preparation import load_and_preprocess_data
from model import create_model
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np

train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path)
model = create_model(7)

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        val_true = []
        for images, labels in validation_dataset:
            predictions = model.predict(images)
            val_pred.extend(tf.argmax(predictions, axis=1).numpy())
            val_true.extend(tf.argmax(labels, axis=1).numpy())

        train_pred = []
        train_true = []
        for images, labels in train_dataset:
            predictions = model.predict(images)
            train_pred.extend(tf.argmax(predictions, axis=1).numpy())
            train_true.extend(tf.argmax(labels, axis=1).numpy())

        val_f1 = f1_score(val_true, val_pred, average='weighted')
        train_f1 = f1_score(train_true, train_pred, average='weighted')

        logs['val_f1'] = val_f1
        logs['train_f1'] = train_f1

        print(f"Epoch {epoch + 1}: Train F1: {train_f1}, Val F1: {val_f1}")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=2,
    callbacks=[F1ScoreCallback()]
)

model.save('emotion_recognition_model_enriched.h5')

# Save history
np.save('history.npy', history.history)
