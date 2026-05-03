# models/ml/naive_bayes_model.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

class NaiveBayesModel:
    def __init__(self, alpha=1.0):
        # alpha là tham số smoothing (Laplace smoothing)
        self.model = MultinomialNB(alpha=alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)