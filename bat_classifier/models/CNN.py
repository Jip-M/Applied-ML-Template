import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import time
from typing import Optional, Tuple
import numpy as np

class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, learning_rate: float, number_of_epochs: int, patience: int) -> None:
        """
        Initialize the CNN model with convolutional, pooling, and fully connected layers.
        Args:
            num_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            number_of_epochs (int): Number of training epochs.
            patience (int): Patience for early stopping.
        """
        super().__init__()
        # convolutional layers.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=3, padding=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=3, padding=4)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # dummy input 
        dummy_input = torch.zeros(1, 1, 512, 1024)

        # pooling layers
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        fc1_input_features = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(fc1_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.num_epochs = number_of_epochs
        self.num_classes = num_classes
        self.patience = patience
        self.device: Optional[torch.device] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 512, 1024).
        Returns:
            torch.Tensor: Output logits for each class.
        """
        # Apply 3 convolutional + pooling blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def early_stopping(self, loss: float, best_loss: float, epoch: int, epochs_no_improve: int, best_model_state: Optional[Tuple[dict, int]] = None, verbose: bool = True) -> Tuple[float, int, bool, Optional[Tuple[dict, int]]]:
        """
        Implements early stopping based on validation or test loss.
        Args:
            loss (float): Current loss.
            best_loss (float): Best loss so far.
            epoch (int): Current epoch.
            epochs_no_improve (int): Number of epochs without improvement.
            best_model_state (Optional[Tuple[dict, int]]): State dict and epoch of best model.
            verbose (bool): Whether to print messages.
        Returns:
            Tuple containing updated best_loss, epochs_no_improve, stop flag, and best_model_state.
        """
        stop = False
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
            best_model_state = (self.state_dict(), epoch + 1)
        else:
            epochs_no_improve += 1
        # Stop if no improvement for 'patience' epochs
        if epochs_no_improve >= self.patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch+1}")
            if best_model_state is not None:
                self.load_state_dict(best_model_state[0])
                if verbose:
                    print("Loaded best model state from epoch", best_model_state[1])
            self.num_epochs = epoch
            stop = True
        return best_loss, epochs_no_improve, stop, best_model_state

    def fit(self, verbose: bool = True):
        """
        Train the CNN model.
        Args:
            verbose (bool): Whether to print progress messages.
        Returns:
            Tuple of lists: (train_losses, train_accuracies, val/test_losses, val/test_accuracies)
        """
        train_losses = []
        train_accuracies = []
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        if self.validation_data is not None:
            val_losses = []
            val_accuracies = []
            test_losses = None
            test_accuracies = None
        else:
            test_losses = []
            test_accuracies = []
            val_accuracies = None
            val_losses = None
        for epoch in range(self.num_epochs):
            start = time.perf_counter()
            self.train()
            running_loss = 0.0
            running_accuracy = 0.0
            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # Use torch.max to get predicted class indices
                _, outputs_idx = torch.max(outputs, dim=1)
                running_accuracy += accuracy_score(labels.cpu().numpy(), outputs_idx.cpu().numpy())
            epoch_loss = running_loss / len(self.train_data)
            epoch_accuracy = running_accuracy / len(self.train_data)
            if self.validation_data is not None:
                if verbose:
                    print("the validation started")
                val_loss, val_accuracy = self.evaluate(self.validation_data)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                best_loss, epochs_no_improve, stop, best_model_state = self.early_stopping(loss, best_loss, epoch, epochs_no_improve, best_model_state, verbose=verbose)
                if stop:
                    if verbose:
                        print("Early stopping triggered, training stopped.")
                    break
            else:
                test_loss, test_accuracy = self.evaluate(self.test_data)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                best_loss, epochs_no_improve, stop, best_model_state = self.early_stopping(loss, best_loss, epoch, epochs_no_improve, best_model_state, verbose=verbose)
                if stop:
                    if verbose:
                        print("Early stopping triggered, training stopped.")
                    break
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            end = time.perf_counter()
            if self.validation_data is not None and verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Time: {(end - start):.2f} seconds")
            elif verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {(end - start):.2f} seconds")
        if self.validation_data is None:
            return train_losses, train_accuracies, test_losses, test_accuracies 
        else:
            return train_losses, train_accuracies, val_losses, val_accuracies

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the classes.
        Returns:
            Tuple of (predicted_classes, raw_outputs)
        """
        all_outputs = torch.empty((0, self.num_classes), device=self.device)
        all_labels = torch.empty((0,), device=self.device)
        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
        _, predicted_classes = torch.max(all_outputs, dim=1)
        return predicted_classes, all_outputs

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.
        Args:
            loader (DataLoader): DataLoader for evaluation.
        Returns:
            Tuple of (loss, accuracy)
        """
        self.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                # Use argmax to get predicted class indices for accuracy
                running_accuracy += accuracy_score(
                    labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy()
                )
        loss = running_loss / len(loader)
        accuracy = running_accuracy / len(loader)
        return loss, accuracy

    def prepare_data(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, val_data: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None, bs: int = 32, device: Optional[torch.device] = None) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Prepare and load data into DataLoaders for training, validation, and testing.
        Args:
            train_data (np.ndarray): Training data, shape (N, 512, 1024, 1)
            train_labels (np.ndarray): Training labels
            test_data (np.ndarray): Test data, shape (N, 512, 1024, 1)
            test_labels (np.ndarray): Test labels
            val_data (Optional[np.ndarray]): Optional validation data, shape (N, 512, 1024, 1)
            val_labels (Optional[np.ndarray]): Optional validation labels
            bs (int): Batch size
            device (Optional[torch.device]): Device to use
        Returns:
            Tuple of (train_loader, test_loader, val_loader)
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        # Convert data to torch tensors 
        train_data = torch.tensor(train_data).permute(0, 3, 1, 2).float().to(self.device)  # (N, C, H, W)
        train_labels_tensor = torch.tensor(train_labels).long().to(self.device)
        test_data = torch.tensor(test_data).permute(0, 3, 1, 2).float().to(self.device) 
        test_labels_tensor = torch.tensor(test_labels).long().to(self.device)
        batch_size = bs
        train_dataset = TensorDataset(train_data, train_labels_tensor)
        test_dataset = TensorDataset(test_data, test_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.train_data = train_loader
        self.test_data = test_loader
        if val_data is not None and val_labels is not None:
            val_data = torch.tensor(val_data).permute(0, 3, 1, 2).float().to(self.device)
            val_labels_tensor = torch.tensor(val_labels).long().to(self.device)
            val_dataset = TensorDataset(val_data, val_labels_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            self.validation_data = val_loader
        else:
            self.validation_data = None
        self.to(self.device)
        return self.train_data, self.test_data, self.validation_data

    def plot_loss(self, losses: list[float], accuracies: list[float], title: str) -> None:
        """
        Plot training/validation/test loss and accuracy curves.
        Args:
            losses (list[float]): List of loss values per epoch.
            accuracies (list[float]): List of accuracy values.
            title (str): Title for the plots.
        """
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs + 1), losses, label="CNN Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CNN " + title + " Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs + 1), accuracies, label="CNN Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("CNN " + title + " Accuracy")
        plt.legend()
        plt.show()

    def load_model(self, model_path: str) -> None:
        """
        Load model weights from a saved state dict.
        Args:
            model_path (str): Path to the saved model file.
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)