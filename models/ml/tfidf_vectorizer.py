import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf

class TFIDFVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        max_features: Giới hạn số lượng từ vựng quan trọng nhất để tránh quá tải.
        ngram_range: (1, 2) có nghĩa là lấy cả từ đơn ("vui") và cụm từ đôi ("rất_vui").
        """
        self.vectorizer = SklearnTfidf(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True # Giúp mượt hoá tần suất xuất hiện của từ
        )

    def fit_transform(self, raw_documents):
        """Học từ vựng và chuyển đổi văn bản sang ma trận số"""
        return self.vectorizer.fit_transform(raw_documents)

    def transform(self, raw_documents):
        """Chuyển đổi văn bản dựa trên từ vựng đã học (dùng khi inference)"""
        return self.vectorizer.transform(raw_documents)

    def save(self, path):
        joblib.dump(self.vectorizer, path)

    def load(self, path):
        self.vectorizer = joblib.load(path)