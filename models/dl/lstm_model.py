import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def build_lstm_model(vocab_size, embed_dim, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim),
        LSTM(128),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model