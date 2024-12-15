import wave
import numpy as np
from scipy.signal import spectrogram
from pydub import AudioSegment
from models.model_loader import load_model
from sklearn.preprocessing import StandardScaler

model = load_model()


def convert_to_wav(input_file: str, output_file: str):
    audio = AudioSegment.from_file(input_file, format="ogg")
    audio.export(output_file, format="wav")


def extract_features(file_path: str):
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_frames = wf.readframes(n_frames)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)

        _, _, Sxx = spectrogram(audio_data, fs=sample_rate)

        scaler = StandardScaler()
        return scaler.fit_transform(Sxx)[np.newaxis, ..., np.newaxis]


def transcribe_audio(file_path: str):
    features = extract_features(file_path)
    predictions = model.predict(features)
    return decode_predictions(predictions)


def decode_predictions(predictions):
    return "".join([chr(np.argmax(pred)) for pred in predictions])
