import joblib
import os

class NBInference:
    def __init__(self):

        MODEL_DIR = 'saved_models/nb'

        self.model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    def predict(self, text):

        if not text or text.strip() == "":
            return "invalid input"

        X = self.vectorizer.transform([text])

        label_id = self.model.predict(X)[0]

        label = self.label_encoder.inverse_transform([label_id])[0]

        return label


if __name__ == "__main__":
    infer = NBInference()

    print(infer.predict("Tôi đang rất vui hôm nay"))
    print(infer.predict("Tôi cảm thấy rất tệ và buồn"))