from sklearn.model_selection import KFold, train_test_split
from bat_classifier.models.base_model import BaseModel
import torch
import numpy as np
import os
import pandas as pd
from bat_classifier.models.metrics import MultiClassMetrics
from bat_classifier.models.CNN import AudioCNN
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


from bat_classifier.data.download_data import create_folders, download_metadata, load_metadata, download_files
from bat_classifier.data.preprocess import preprocess_all_data

def kfold_validation(X: np.ndarray, y: np.ndarray, k: int = 5):
    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    count = 1
    validation_images, validation_labels = None, None
    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        accuracies_CNN = []
        accuracies_base = []
        
        print(f"Training fold {count} of {k}...")
        print("CNN:")
        model, batch_size = best_model()  # this initializes the model with the best parameters, which are defined in the function
        
        # model = initialize_CNN(num_classes=4, learning_rate=0.0005, num_epochs=50, patience=6)
    
        start_of_training = time.perf_counter()
        
        model.prepare_data(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, batch_size, device)
        train_losses, train_accuracies, losses, accuracies_fit = model.fit(verbose=False)
        # "losses" and "accuracies" are the losses and accuracies for the test set, or, if we have a validation set, the validation set.
        end_of_training = time.perf_counter()
        elapsed = end_of_training - start_of_training
        print(f"Total time to train the CNN: {elapsed / 60} minutes")

        predictions, probabilities = model.predict()
        
        is_last_fold = (test_index[-1] == X.shape[0] - 1)
        
        accuracy = measurements(test_labels, predictions, probabilities, model_type="cnn", save_metrics=is_last_fold)
        accuracies_CNN.append(accuracy)
        # model.plot_loss(train_losses, train_accuracies, title="Training")
        
        ############################################################################################################
        print("Base model:")
        start_of_training = time.perf_counter()

        base_model = initialize_basemodel()
        base_model.prepare_data(train_images, train_labels, test_images, test_labels)
        base_model.fit()

        end_of_training = time.perf_counter()
        predictions_base, probabilities_base = base_model.predict()

        elapsed = end_of_training - start_of_training
        print(f"Total time to train the base model: {elapsed / 60} minutes")
        accuracy = measurements(test_labels, predictions_base, probabilities_base, model_type="lr", save_metrics=is_last_fold)
        accuracies_base.append(accuracy)

        # model.plot_loss(losses, accuracies_fit, title="Test/Validation")
        
        count += 1
        # break  # uncomment this line to only run one fold for testing purposes

    accuracies_CNN = np.array(accuracies_CNN)
    avg_accuracy_CNN = np.mean(accuracies_CNN)
    print("CNN test accuracy mean:", round(avg_accuracy_CNN, 4))
    
    accuracies_base = np.array(accuracies_base)
    avg_accuracy_base = np.mean(accuracies_base)
    print("Base model test accuracy mean:", round(avg_accuracy_base, 4))


def initialize_basemodel():
    base_model = BaseModel()
    return base_model


def retrieve_data():
    print("Preparing data")
    data_dir = os.path.join(os.path.dirname(__file__), "data", "cleaned")
    label_file = os.path.join(os.path.dirname(__file__), "data", "labels.csv")
    label_array = pd.read_csv(label_file, header=None, names=["file_id", "species"], dtype={"str": "int"})["species"].to_numpy()
    csv_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')])

    images = []
    for i, file in enumerate(csv_files):
        img = pd.read_csv(file, header=None).values  # shape: (512, 1024)
        images.append(img)

    images = np.array(images)  # shape: (n_samples, 512, 1024)
    images = images[..., np.newaxis]  # Add channel dimension: (n_samples, 512, 1024, 1)
    labels = label_array
    print(f"Loaded {len(csv_files)} sonograms")

    return images, labels
    

def initialize_CNN(num_classes: int, learning_rate, num_epochs, patience):
    torch.manual_seed(2222)
    np.random.seed(2222)

    # CNN Model
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate, number_of_epochs=num_epochs, patience=patience)

    return model

def download_and_preprocess():
    selected_species = ["pipistrellus", "noctula", "auritus", "albescens"]
    download_metadata()
    data = load_metadata(selected_species)
    download_files(data)
    preprocess_all_data(data, selected_species)


def measurements(test_labels, predictions, probabilities, model_type="CNN", accuracies=None, save_metrics=False):
    metric = MultiClassMetrics()
    if accuracies is not None:
        average_accuracy = np.mean(accuracies)
        print(f"Average accuracy: {average_accuracy}")

    accuracy = metric.accuracy(test_labels, predictions)
    roc = metric.auroc(test_labels, probabilities)
    cm = metric.confusion(test_labels, predictions)

    if model_type.lower() == "cnn":
        print(f"CNN Test ROC AUC: {roc:.4f}")
        print(f"CNN Test Accuracy: {accuracy:.4f}")
    else:
        print(f"Logistic Regression Test ROC AUC: {roc:.4f}")
        print(f"Logistic Regression Test Accuracy: {accuracy:.4f}")
    print(cm)

    if save_metrics:
        import pandas as pd
        metrics_dict = {
            "Accuracy": [accuracy],
            "ROC AUC": [roc],
            "Confusion Matrix": [cm.tolist()]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        if model_type.lower() == "cnn":
            metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './trained_model/cnn_metrics.csv'))
        else:
            metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './trained_model/lr_metrics.csv'))
        metrics_dir = os.path.dirname(metrics_path)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        metrics_df.to_csv(metrics_path, index=False)
    return accuracy

def best_model():
    model = initialize_CNN(num_classes=4, learning_rate=0.0005, num_epochs=50, patience=6)
    batch_size = 32
    return model, batch_size


def run_pipeline():
    # if this is the first time running the pipeline, you need to uncomment the next two lines of code:
    #create_folders()
    #download_and_preprocess()

    images, labels = retrieve_data()
    print("Starting the kfold cross validation!")

    print("Kfold cross validation started!")
    kfold_validation(images, labels, k=5)

    # training_base(images, labels)
    
    # this line of code saves the model's parameters after training
    # torch.save(state_dict, os.path.join(os.path.dirname(__file__), "trained_model", "CNN.pt"))


run_pipeline()


# if you run the next two lines of code, you will "retrieve" the data and save it in a .npz file
# this way, when you plan to run the model multiple times, instead of waiting to get the data again, you can just load the .npz file
#images, labels = retrieve_data()
# np.savez("mydata.npz", images=images, labels=labels)

# the next three lines of code are for loading the data from the .npz file
# data = np.load("mydata.npz")
# images = data["images"]
# labels = data["labels"]
