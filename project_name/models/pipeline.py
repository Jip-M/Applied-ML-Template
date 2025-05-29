from sklearn.model_selection import KFold
from base_model import BaseModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
import os
import pandas as pd
from metrics import MultiClassMetrics
from CNN import AudioCNN
from base_model import BaseModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def kfold_validation_CNN(X: np.ndarray, y: np.ndarray, k: int = 5, num_classes: int = 2) -> (float, np.ndarray):
    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    accuracies = []
    cm = np.zeros((num_classes, num_classes))

    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        num_epochs, model = initialize_CNN(num_classes)
        train_data, train_labels_tensor = numpy_to_tensor(train_images, train_labels)
        test_data, test_labels_tensor = numpy_to_tensor(test_images, test_labels)
        batch_size = 32
        train_dataset = TensorDataset(train_data, train_labels_tensor)
        test_dataset = TensorDataset(test_data, test_labels_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        train_losses, train_accuracies = model.train_model(num_epochs, train_loader)
        predictions, labels = model.predict(test_loader)
        accuracy = model.evaluate(predictions, labels)
        cm += confusion_matrix(labels, predictions, labels=np.arange(0, num_classes))
        accuracies.append(accuracy)
        print(f"CNN Test Accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)
        # model.plot_train_loss(num_epochs, train_losses, train_accuracies)

    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {average_accuracy}")
    return average_accuracy, cm


def prepare_data():
    print("Preparing data")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cleaned")
    label_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "labels.csv")
    label_array = pd.read_csv(label_file, header=None, names=["file_id", "species"], dtype={"str": "int"})["species"].to_numpy()
    csv_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')])

    images = []
    for i, file in enumerate(csv_files):
        img = pd.read_csv(file, header=None).values  # shape: (281, 1000)
        images.append(img)

    images = np.array(images)  # shape: (n_samples, 281, 1000)
    images = images[..., np.newaxis]  # Add channel dimension: (n_samples, 281, 1000, 1)
    labels = label_array
    print(f"Loaded {len(csv_files)} sonograms")
    return images, labels # train_images, test_images, train_labels, test_labels (before)


def numpy_to_tensor(data, labels):
        data_tensor = torch.tensor(data).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        labels_tensor = torch.tensor(labels).long()
        return data_tensor, labels_tensor
    

def initialize_CNN(num_classes: int):
    torch.manual_seed(2222)
    np.random.seed(2222)

    learning_rate = 0.001
    num_epochs = 10

    # CNN Model
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate)

    return num_epochs, model


def run_basemodel(train_images, test_images, train_labels, test_labels):
    print("Evaluating BaseModel")
    # Flatten images for BaseModel: (N, 281, 1000, 1) -> (N, 281000)
    X_train_flat = train_images.reshape(train_images.shape[0], -1)
    X_test_flat = test_images.reshape(test_images.shape[0], -1)

    base_model = BaseModel()
    base_model.train(X_train_flat, train_labels)
    base_test_accuracy = base_model.evaluate(X_test_flat, test_labels)
    print(f"BaseModel Test Accuracy: {base_test_accuracy:.4f}")
    

def run_pipeline():
    #train_images, test_images, train_labels, test_labels = prepare_data()
    images, labels = prepare_data()
    #initialize_CNN()
    (acc, cm) = kfold_validation_CNN(images, labels, k=5, num_classes=4)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    # run_CNN(train_images, test_images, train_labels, test_labels)
    # run_basemodel(train_images, test_images, train_labels, test_labels)


run_pipeline()