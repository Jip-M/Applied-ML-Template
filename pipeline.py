from sklearn.model_selection import KFold, train_test_split
from project_name.models.base_model import BaseModel
import torch
import numpy as np
import os
import pandas as pd
from project_name.models.metrics import MultiClassMetrics
from project_name.models.CNN import AudioCNN
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


from project_name.data.download_data import create_folders, download_metadata, load_metadata, download_files
from project_name.data.preprocess import preprocess_all_data

def kfold_validation(X: np.ndarray, y: np.ndarray, k: int = 5):
    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    validation_images, validation_labels = None, None
    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        accuracies = []
        
        model, batch_size = best_model()  # this initializes the model with the best parameters, which are defined in the function
        
        # model = initialize_CNN(num_classes=4, learning_rate=0.0005, num_epochs=50, patience=6, save=False) # if you set "save=True", it saves the model after each epoch in a separate folder
    
        start_of_training = time.perf_counter()
        
        model.prepare_data(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, batch_size, device)
        train_losses, train_accuracies, losses, accuracies_train = model.fit() # the two _ _ are substitutes for the test loss
        # "losses" and "accuracies" are the losses and accuracies for the test set, or, if we have a validation set, the validation set.
        end_of_training = time.perf_counter()
        elapsed = end_of_training - start_of_training
        print(f"Total time to train the model: {elapsed / 60} minutes")

        predictions, probabilities = model.predict()
        
        is_last_fold = (test_index[-1] == X.shape[0] - 1)
        accuracy = measurements(test_labels, predictions, probabilities, model_type="cnn", save_metrics=is_last_fold)
        accuracies.append(accuracy)
        model.plot_loss(train_losses, train_accuracies, title="Triaining")

        # this one does not work yet
        # model.plot_loss(losses, accuracies_train, title="Test/Validation")
        # break  # uncomment this line to only run one fold for testing purposes
    accuracies = np.array(accuracies)
    avg_accuracy = np.mean(accuracies)
    print("CNN test accuracy mean:", round(avg_accuracy, 4))



    # state_dict = model.state_dict()
    # average_accuracy = np.mean(accuracies)
    # print(f"Average accuracy: {average_accuracy}")
    # return average_accuracy, cm, state_dict

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
    # Assuming images and labels are already defined as numpy arrays
    return images, labels # train_images, test_images, train_labels, test_labels (before)
    

def initialize_CNN(num_classes: int, learning_rate, num_epochs, patience,  save):
    torch.manual_seed(2222)
    np.random.seed(2222)

    # CNN Model
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate, number_of_epochs=num_epochs, patience=patience, save=save)

    return model

def download_and_preprocess():
    selected_species = ["pipistrellus", "noctula", "auritus", "albescens"]
    download_metadata()
    data = load_metadata(selected_species)
    download_files(data)
    preprocess_all_data(data, selected_species)


def select_epoch(model, epoch: int):
    # this lets you load the paramters from the previous run to see how the model performs
    print(f"the model has loaded the state dict from {epoch}")
    model.load_model(f"project_name/models/trained_model/epoch{epoch}/CNN_{epoch}.pt")

def training_CNN():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    metric = MultiClassMetrics()
    data = np.load("mydata.npz")
    images = data["images"]
    labels = data["labels"]
    validation_images, validation_labels = None, None
    # First, split into train+val and test
    kf = KFold(n_splits=5, shuffle=True, random_state=22)
    accuracies = []
    cm = np.zeros((4, 4))
    
    X = images
    y = labels
    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        break
    # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=22, stratify=labels)

    # # Then, split train+val into train and validation
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=22, stratify=train_labels)
    # different number of epochs to test
    list_of_accuracies = []
    details = {}
    # for epoch in [10, 20, 30, 40, 50]:
    #     print(f"Training epochs: {epoch + 1}")
    #     for learning_rate in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    #         print(f"Training with learning rate {learning_rate}")
    #         # Initialize the model with the specified parameters
    #         for batch_size in [16, 32, 64]:
    #             print(f"Training with batch size {batch_size}")
    # details["amount of epochs"] = epoch
    # details["learning rate"] = learning_rate
    # details["batch size"] = batch_size
    model = initialize_CNN(num_classes=4, learning_rate=0.01, num_epochs=50, patience=7, save=False) # if you set "save=True", it saves the model after each epoch in a separate folder
    
    start_of_training = time.perf_counter()
    
    model.prepare_data(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, 64, device)
    train_losses, train_accuracies, losses, accuracies = model.fit()
    # "losses" and "accuracies" are the losses and accuracies for the test set, or, if we have a validation set, the validation set.
    end_of_training = time.perf_counter()
    elapsed = end_of_training - start_of_training
    print(f"Total time to train the model: {elapsed / 60} minutes")

    predictions, probabilities = model.predict()
    
    accuracy = metric.accuracy(test_labels, predictions)

    
    measurements(test_labels, predictions, probabilities, model_type="cnn")
    # model.plot_train_loss(train_losses, train_accuracies)
    # print("train losses:", len(train_losses),"train accuracies", len(train_accuracies))
    # print("losses shape:", len(losses), "accuracies shape:", len(accuracies))
    # model.plot_train_loss(losses[:len(losses) - 1], accuracies[:len(accuracies) - 1])

    
    # list_of_accuracies = sorted(list_of_accuracies, key=lambda x: x[0], reverse=True)
    # print("The best accuracy was achieved with the following parameters:")
    # print(list_of_accuracies)
    

def training_base(images, labels):
    kf = KFold(n_splits=5, shuffle=True, random_state=22)
    accuracies = []
    
    X = images
    y = labels


    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        print("Base model started training!")
        base_model = initialize_basemodel()
        base_model.prepare_data(train_images, train_labels, test_images, test_labels)
        base_model.fit()

        predictions_base, probabilities_base = base_model.predict()
        is_last_fold = (test_index[-1] == X.shape[0] - 1)
        accuracy = measurements(test_labels, predictions_base, probabilities_base, model_type="lr", save_metrics=is_last_fold)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    print("Base model mean accuracy after the 5 folds", np.mean(accuracies))


def evaluate_model(choose_epoch: int):
    """
    This function is for evaluating the model after training, from a state dict saved from each epoch
    to check its performance.
    """
    data = np.load("mydata.npz")
    images = data["images"]
    labels = data["labels"]
    print("got the data")
    # First, split into train+val and test
    kf = KFold(n_splits=5, shuffle=True, random_state=22)
    accuracies = []
    # cm = np.zeros((4, 4))
    X = images
    y = labels
    for train_index, test_index in kf.split(X):
        train_images, test_images = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        break
    

    # # Then, split train+val into train and validation
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=22, stratify=train_labels)

    model = initialize_CNN(num_classes=4, learning_rate=0.005, num_epochs=20, save=True) # if you set "save=True", it saves the model after each epoch in a separate folder

    batch_size = 32
    model.prepare_data(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, batch_size)


    # select_epoch(model, choose_epoch)
    # 
    # we need the probabilities for the ROC AUC and the predictions for the other metrics
    # predictions, probabilities = model.predict()
    # measurements(test_labels, predictions, probabilities, model_type="cnn")


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
            metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../trained_model/cnn_metrics.csv'))
        else:
            metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../trained_model/lr_metrics.csv'))
        metrics_dir = os.path.dirname(metrics_path)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        metrics_df.to_csv(metrics_path, index=False)
    return accuracy

def best_model():
    model = initialize_CNN(num_classes=4, learning_rate=0.0005, num_epochs=50, patience=7, save=False) # if you set "save=True", it saves the model after each epoch in a separate folder
    batch_size = 32
    return model, batch_size


def run_pipeline():
    images, labels = retrieve_data()
    print("Starting the kfold cross validation!")

    print("CNN:")
    kfold_validation(images, labels, k=5)

    print("Base model:")
    training_base(images, labels)
    # torch.save(state_dict, os.path.join(os.path.dirname(__file__), "trained_model", "CNN.pt"))

    # ConfusionMatrixDisplay(cm).plot()
    # plt.show()

#create_folders()
#download_and_preprocess()
run_pipeline()
# now the model saves all the state dict from each epoch in a separate folder

# if you run the next two lines of code, you will "retrieve" the data and save it in a .npz file
# this way, when you plan to run the model multiple times, instead of waiting to get the data again, you can just load the .npz file
#images, labels = retrieve_data()
# np.savez("mydata.npz", images=images, labels=labels)

# the next three lines of code are for loading the data from the .npz file
# data = np.load("mydata.npz")
# images = data["images"]
# labels = data["labels"]
# kfold_validation(images, labels, k=5)



# training_CNN()
# training_base()
# evaluate_model(18)
