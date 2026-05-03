import pandas as pd
import os
import joblib

from sklearn.metrics import classification_report, accuracy_score
from models.ml.tfidf_vectorizer import TFIDFVectorizer
from models.ml.naive_bayes_model import NaiveBayesModel


label_names = {
    0: "trung_tính",
    1: "vui",
    2: "buồn",
    3: "sợ_hãi",
    4: "tức_giận",
    5: "ghê_tởm",
    6: "ngạc_nhiên"
}


import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_nb():

    TRAIN_PATH = 'data/split/train.csv'
    TEST_PATH = 'data/split/test.csv'
    MODEL_DIR = 'saved_models/nb'

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.dropna(subset=['nb_text', 'label'])
    test_df = test_df.dropna(subset=['nb_text', 'label'])

    print(">>> Cleaning text...")
    train_df['nb_text'] = train_df['nb_text'].apply(clean_text)
    test_df['nb_text'] = test_df['nb_text'].apply(clean_text)

    y_train = train_df['label'].values
    y_test = test_df['label'].values


    print(">>> Vectorizing text...")
    vectorizer = TFIDFVectorizer(max_features=10000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_df['nb_text'])
    X_test = vectorizer.transform(test_df['nb_text'])

  
    print(">>> Training Naive Bayes...")
    model = NaiveBayesModel(alpha=0.1)
    model.train(X_train, y_train)


    print("Saving model...")

    model.save(f"{MODEL_DIR}/model.pkl")
    vectorizer.save(f"{MODEL_DIR}/tfidf_vectorizer.pkl")

    print(f"Saved to {MODEL_DIR}")

    print("\n>>> Evaluation...")

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    target_names = [label_names[i] for i in sorted(label_names.keys())]

    print("\nClassification Report:\n")

    print(classification_report(
        y_test,
        y_pred,
        target_names=target_names
    ))


if __name__ == "__main__":
    train_nb()