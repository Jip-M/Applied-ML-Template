from sklearn.model_selection import KFold
from base_model import BaseModel
import torch
import numpy as np
import os
import pandas as pd
from metrics import MultiClassMetrics
from CNN import AudioCNN
from base_model import BaseModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from project_name.data.download_data import download_metadata, load_metadata, download_files
from project_name.data.preprocess import preprocess_all_data

def kfold_validation(X: np.ndarray, y: np.ndarray, model, k: int = 5, num_classes: int = 2):
    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    accuracies = []
    cm = np.zeros((num_classes, num_classes))
    metric = MultiClassMetrics()

    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        model.prepare_data(train_images, train_labels, test_images, test_labels)
        model.fit()
        predictions = model.predict()
        confusion_matrix = metric.confusion(test_labels, predictions)
        # auroc = metric.auroc(test_labels, predictions)
        accuracy = metric.accuracy(test_labels, predictions)
        accuracies.append(accuracy)

        cm += confusion_matrix(test_labels, predictions, labels=np.arange(0, num_classes))
        accuracies.append(accuracy)
        print(f"CNN Test Accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)
        # model.plot_train_loss(num_epochs, train_losses, train_accuracies)

    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {average_accuracy}")
    return average_accuracy, cm


def initialize_CNN():
    torch.manual_seed(2222)
    np.random.seed(2222)

    num_classes = 2 
    learning_rate = 0.001
    num_epochs = 1

    # CNN Model
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate, number_of_epochs=num_epochs)

    return model


def initialize_basemodel():
    base_model = BaseModel()
    return base_model


def retrieve_data():
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
    

def initialize_CNN(num_classes: int):
    torch.manual_seed(2222)
    np.random.seed(2222)

    learning_rate = 0.001
    num_epochs = 1

    # CNN Model
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate, number_of_epochs=num_epochs)

    return model

def download_and_preprocess():
    selected_species = ["pipistrellus", "noctula", "auritus", "albescens"]
    download_metadata()
    data = load_metadata(selected_species)
    download_files(data)
    preprocess_all_data(data, selected_species)

def run_pipeline():
    images, labels = retrieve_data()
    print("Starting the kfold cross validation!")
    print("CNN:")
    (acc, cm) = kfold_validation(images, labels, model=initialize_CNN(num_classes=4), k=5)
    print("Basemodel:")
    kfold_validation(images, labels, model=initialize_basemodel(), k=5)

    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    # run_CNN(train_images, test_images, train_labels, test_labels)
    # run_basemodel(train_images, test_images, train_labels, test_labels)

download_and_preprocess()
run_pipeline()