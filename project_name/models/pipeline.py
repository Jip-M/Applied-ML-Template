import numpy as np
from sklearn.model_selection import KFold
from base_model import BaseModel


def kfold_validation(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:

    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = BaseModel()
        model.train(X_train, y_train)
        accuracy = model.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {average_accuracy}")
    return average_accuracy
