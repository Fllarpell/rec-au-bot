from keras.models import load_model as keras_load_model

MODEL_PATH = "models/speech_model.h5"


def load_model():
    try:
        model = keras_load_model(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Error with uploading model: {e}")
