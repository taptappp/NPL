import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = "/content/drive/MyDrive/NCKH"
MODEL_DIR = f"{BASE_DIR}/saved_models/lstm"
MAX_LEN = 100


label_map = {
    0: "trung_tính",
    1: "vui",
    2: "buồn",
    3: "sợ_hãi",
    4: "tức_giận",
    5: "ghê_tởm",
    6: "ngạc_nhiên"
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


model = load_model(f"{MODEL_DIR}/model.keras")
tokenizer = joblib.load(f"{MODEL_DIR}/tokenizer.pkl")


def predict(text):
    text = clean(text)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(pad, verbose=0)

    label_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return {
        "label_id": label_id,
        "label": label_map[label_id],
        "confidence": round(confidence, 4)
    }