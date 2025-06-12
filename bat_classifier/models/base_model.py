import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class BaseModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.train_data = [None, None]
        self.test_data = [None, None]

    def fit(self):
        self.model.fit(self.train_data[0], self.train_data[1])

    def predict(self):
        probabilities = self.model.predict_proba(self.test_data[0])
        return self.model.predict(self.test_data[0]), probabilities
    
    def prepare_data(self, train_images, train_labels, test_images, test_labels):
        # Flatten images for BaseModel: (N, 281, 1000, 1) -> (N, 281000)
        X_train_flat = train_images.reshape(train_images.shape[0], -1)
        X_test_flat = test_images.reshape(test_images.shape[0], -1)
        self.train_data[0], self.train_data[1] = X_train_flat, train_labels
        self.test_data[0], self.test_data[1] = X_test_flat, test_labels

