import os
import numpy as np
import pandas as pd
import librosa
import multiprocessing as mp
from tqdm import tqdm
import h5py


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "cv-corpus-20.0-2024-12-06", "ru"))
CLIPS_PATH = os.path.join(BASE_PATH, "clips")
TRANSCRIPTIONS_PATH = os.path.join(BASE_PATH, "validated.tsv")
OUTPUT_PATH = os.path.join(BASE_PATH, "preprocessed")

SAMPLE_RATE = 16000
N_MELS = 128
os.makedirs(OUTPUT_PATH, exist_ok=True)


def process_audio(row):
    audio_path, transcription = row
    file_path = os.path.join(CLIPS_PATH, audio_path)
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=5.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec, transcription
    except Exception as e:
        print(f"Ошибка обработки {file_path}: {e}")
        return None


def preprocess_data():
    print("Загрузка данных...")
    df = pd.read_csv(TRANSCRIPTIONS_PATH, sep="\t", usecols=["path", "sentence"])
    total_files = len(df)

    print(f"Начало обработки {total_files} файлов...")

    pool = mp.Pool(mp.cpu_count())
    results = []
    processed_files = 0

    for result in tqdm(pool.imap_unordered(process_audio, df.itertuples(index=False, name=None)), total=total_files):
        if result:
            results.append(result)
            processed_files += 1

    pool.close()
    pool.join()

    print(f"Обработано файлов: {processed_files} из {total_files}")

    if results:
        data, labels = zip(*results)

        # Фильтрация некорректных данных
        filtered_data = []
        filtered_labels = []
        for mel_spec, label in zip(data, labels):
            if mel_spec.size > 0:  # Проверка на пустые спектрограммы
                filtered_data.append(mel_spec)
                filtered_labels.append(label)

        print(f"Корректных данных для сохранения: {len(filtered_data)}")

        print("Сохранение данных...")
        with h5py.File("preprocessed.h5", "w") as hf:
            # Используем vlen для данных переменной длины
            data_dtype = h5py.special_dtype(vlen=np.float32)
            string_dtype = h5py.string_dtype(encoding="utf-8")

            hf.create_dataset("data", (len(filtered_data),), dtype=data_dtype)
            hf.create_dataset("labels", data=np.array(filtered_labels, dtype=string_dtype))

            for i, mel_spec in enumerate(tqdm(filtered_data, desc="Сохранение данных")):
                hf["data"][i] = mel_spec.flatten()

        print(f"Данные успешно сохранены в {OUTPUT_PATH}/preprocessed.h5")


if __name__ == "__main__":
    preprocess_data()
