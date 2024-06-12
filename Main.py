from data_preparation import load_and_preprocess_data
from model import create_model

if __name__ == "__main__":
    train_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/train'
    validation_data_path = 'C:/Users/larak/Desktop/ORV/CVProject/dataset/validation'

    train_dataset, validation_dataset = load_and_preprocess_data(train_data_path, validation_data_path)
    model = create_model(7)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10
    )

    model.save('emotion_recognition_model.h5')
