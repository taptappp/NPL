import pandas as pd
import numpy as np
import os
import joblib
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =====================
# SEED (ổn định kết quả)
# =====================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# CONFIG
MAX_WORDS = 20000
MAX_LEN = 100
EMBED_DIM = 128
#MODEL_DIR = "saved_models/lstm"
MODEL_DIR = "/content/drive/MyDrive/NCKH/saved_models/lstm"

os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv("/content/drive/MyDrive/NCKH/data/split/train.csv")
df = df.dropna(subset=["lstm_text", "label"])


print(df["label"].value_counts())  # check data balance


texts = df["lstm_text"].astype(str).values
labels = df["label"].values


le = LabelEncoder()
y = le.fit_transform(labels)


X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts, y, test_size=0.2, random_state=42
)


tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
)

tokenizer.fit_on_texts(X_train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=MAX_LEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_texts), maxlen=MAX_LEN)


model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM),
    LSTM(128),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print(">>> Training LSTM...")
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val)
)


loss, acc = model.evaluate(X_val, y_val)
print("Validation accuracy:", acc)


print(">>> Saving model...")
model.save(f"{MODEL_DIR}/model.keras")
joblib.dump(tokenizer, f"{MODEL_DIR}/tokenizer.pkl")
joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

print("DONE ✔")