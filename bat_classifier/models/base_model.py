import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple


class BaseModel:
    def __init__(self) -> None:
        self.model = LogisticRegression()
        self.train_data = [None, None]
        self.test_data = [None, None]

    def fit(self) -> None:
        """
        Fits the logistic regression model to the training data.
        """
        self.model.fit(self.train_data[0], self.train_data[1])

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the class labels and probabilities.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple with the predicted labels and their probabilities.
        """
        probabilities = self.model.predict_proba(self.test_data[0])
        return self.model.predict(self.test_data[0]), probabilities

    def prepare_data(self, train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray) -> None:
        """
        Prepares the data for training and testing by flattening the images.

        Args:
            train_images (np.ndarray): Training images of shape (n_samples, 512, 1024, 1).
            train_labels (np.ndarray): Labels for training images.
            test_images (np.ndarray): Testing images of shape (n_samples, 512, 1024, 1).
            test_labels (np.ndarray): Labels for testing images.

        After flattening, images will have shape (n_samples, 512*1024) = (n_samples, 524288).
        """
        X_train_flat = train_images.reshape(train_images.shape[0], -1)
        X_test_flat = test_images.reshape(test_images.shape[0], -1)
        self.train_data[0], self.train_data[1] = X_train_flat, train_labels
        self.test_data[0], self.test_data[1] = X_test_flat, test_labels

