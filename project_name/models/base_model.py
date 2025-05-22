import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class BaseModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:

        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
