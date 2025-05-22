import numpy as np
from sklearn.model_selection import KFold
from base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pandas as pd


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


torch.manual_seed(2222)
np.random.seed(2222)


data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cleaned")
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
images = []
labels = []


for i, file in enumerate(csv_files):
    img = pd.read_csv(file, header=None).values  # shape: (281, 1000)
    images.append(img)

    labels.append(0 if i < len(csv_files) // 2 else 1)


images = np.array(images)  # shape: (n_samples, 281, 1000)
images = images[..., np.newaxis]  # Add channel dimension: (n_samples, 281, 1000, 1)
labels = np.array(labels)


train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=2222, stratify=labels
)


def numpy_to_tensor(data, labels):
    data_tensor = torch.tensor(data).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    labels_tensor = torch.tensor(labels).long()
    return data_tensor, labels_tensor

train_data, train_labels_tensor = numpy_to_tensor(train_images, train_labels)
test_data, test_labels_tensor = numpy_to_tensor(test_images, test_labels)


batch_size = 32
train_dataset = TensorDataset(train_data, train_labels_tensor)
test_dataset = TensorDataset(test_data, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from CNN import AudioCNN
from base_model import BaseModel

num_classes = 2 
learning_rate = 0.001
num_epochs = 10

# CNN Model
model = AudioCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train_model():
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return train_losses, train_accuracies

train_losses, train_accuracies = train_model()


model.eval()
test_accuracy = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_accuracy += calculate_accuracy(outputs, labels)

test_accuracy /= len(test_loader)
print(f"CNN Test Accuracy: {test_accuracy:.4f}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.show()


print("Evaluating BaseModel")
# Flatten images for BaseModel: (N, 281, 1000, 1) -> (N, 281000)
X_train_flat = train_images.reshape(train_images.shape[0], -1)
X_test_flat = test_images.reshape(test_images.shape[0], -1)

base_model = BaseModel()
base_model.train(X_train_flat, train_labels)
base_test_accuracy = base_model.evaluate(X_test_flat, test_labels)
print(f"BaseModel Test Accuracy: {base_test_accuracy:.4f}")