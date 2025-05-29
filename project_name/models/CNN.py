import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, learning_rate: float, number_of_epochs: int):
        super(AudioCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1_input_features = None
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.num_epochs = 10
        
        self.train_data = None
        self.test_data = None
        self.num_epochs = number_of_epochs
        self.num_classes = num_classes
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        X: shape (batch_size, 1, height, width).

        Returns: shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        
        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1_input_features = x.size(1)
            self.fc1 = nn.Linear(self.fc1_input_features, 128)
            self.fc1.to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self):
        train_losses = []
        train_accuracies = []

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0
            running_accuracy = 0.0
            for inputs, labels in self.train_data:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, outputs = torch.max(outputs, dim=1)
                running_accuracy += accuracy_score(outputs, labels)

            epoch_loss = running_loss / len(self.train_data)
            epoch_accuracy = running_accuracy / len(self.train_data)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # model.plot_train_loss(num_epochs, train_losses, train_accuracies)
        return train_losses, train_accuracies # it returns these but they are not that important, just for the plots
        
    def predict(self):
        """
        Make predictions from raw model outputs
        Args:
            inputs: Tensor of shape (batch_size, ...)
        Returns:
            dict: Contains both class indices and probabilities
        """
        all_outputs = torch.empty(0, self.num_classes)
        all_labels = torch.empty(0)
        with torch.no_grad():
            for inputs, labels in self.test_data:
                outputs = self(inputs)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

        # Get predicted class indices
        _, predicted_classes = torch.max(all_outputs, dim=1)
        
        return predicted_classes
    
    # def evaluate(self, predictions, metric):
    #     # predictions, labels = self.predict(self.test_data)
    #     return accuracy_score(labels, predictions)
    
    def prepare_data(self, train_data, train_labels, test_data, test_labels):
        train_data = torch.tensor(train_data).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        train_labels_tensor = torch.tensor(train_labels).long()
        
        test_data = torch.tensor(test_data).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        test_labels_tensor = torch.tensor(test_labels).long()
        batch_size = 32
        train_dataset = TensorDataset(train_data, train_labels_tensor)
        test_dataset = TensorDataset(test_data, test_labels_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.train_data = train_loader
        self.test_data = test_loader


    def plot_train_loss(self, num_epochs, train_losses, train_accuracies):
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
