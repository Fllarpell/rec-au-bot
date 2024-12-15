import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Reshape, GRU, Bidirectional, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# Пути
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "cv-corpus-20.0-2024-12-06", "ru", "preprocessed"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "speech_model.keras"))
HDF5_PATH = os.path.join(BASE_PATH, "preprocessed.h5")

# Параметры
FIXED_LEN = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
PATIENCE = 5


# Улучшенная функция активации
def swish(x):
    return x * tf.keras.activations.sigmoid(x)


# Функция для обрезки/дополнения данных
def pad_or_trim_features(features, target_len=FIXED_LEN):
    if features.shape[1] > target_len:
        return features[:, :target_len]
    else:
        pad_width = target_len - features.shape[1]
        return np.pad(features, ((0, 0), (0, pad_width)), mode="constant")


# Генератор данных
def hdf5_generator(hdf5_path, batch_size, label_encoder):
    with h5py.File(hdf5_path, "r") as hf:
        data = hf["data"]
        labels = hf["labels"]
        num_samples = len(data)

        # Преобразование меток в индексированный формат
        encoded_labels = label_encoder.transform([label.decode("utf-8") for label in labels])

        while True:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_data = []
                batch_labels = []

                for i in range(start, end):
                    features_reshaped = data[i].reshape((128, -1))
                    batch_data.append(pad_or_trim_features(features_reshaped))
                    batch_labels.append(encoded_labels[i])

                batch_data = np.expand_dims(np.array(batch_data), axis=-1)
                batch_labels = np.array(batch_labels)

                yield batch_data, batch_labels


# Построение модели
def build_model(input_shape, output_units):
    inputs = Input(name="inputs", shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation(swish)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(swish)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Reshape((-1, x.shape[-1] * x.shape[-2]))(x)

    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    outputs = Dense(output_units, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Основной блок
if __name__ == "__main__":
    print("Определение количества уникальных классов...")
    with h5py.File(HDF5_PATH, "r") as hf:
        labels = [label.decode("utf-8") for label in hf["labels"]]

    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    output_units = len(np.unique(y_encoded))
    print(f"Количество уникальных классов: {output_units}")

    print("Построение модели...")
    input_shape = (128, FIXED_LEN, 1)
    model = build_model(input_shape, output_units)

    # Генератор данных
    print("Создание генератора данных...")
    train_gen = hdf5_generator(HDF5_PATH, BATCH_SIZE, label_encoder)
    steps_per_epoch = len(y_encoded) // BATCH_SIZE

    # Настройка обратных вызовов
    callbacks = [
        EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="loss", save_best_only=True, verbose=1)
    ]

    # Обучение модели
    print("Начало обучения...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"Модель сохранена в {MODEL_PATH}")

    # График обучения
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Точность модели во время обучения")
    plt.grid()
    plt.show()

