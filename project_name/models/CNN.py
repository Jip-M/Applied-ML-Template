import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import os


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, learning_rate: float, number_of_epochs: int, patience: int,  save: bool = False):
        super(AudioCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=3, padding=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=3, padding=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=3)


        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        dummy_input = torch.zeros(1, 1, 512, 1024)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        fc1_input_features = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(fc1_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.fc_task1 = nn.Linear(128, num_classes_task1)
        # self.fc_task2 = nn.Linear(128, num_classes_task2)


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        
        self.train_data = None
        self.test_data = None
        self.validation_data = None

        self.num_epochs = number_of_epochs
        self.num_classes = num_classes
        # self.num_classes_task2 = num_classes_task2
        
        self.patience = patience
        
        self.save = save
        
    # def forward(self, x): # for multi-task learning, we return two outputs
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc1(x))
    #     out1 = self.fc_task1(x)
    #     out2 = self.fc_task2(x)
    #     return out1, out2

    def forward(self, x: torch.tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def early_stopping(self, loss, best_loss, epoch, epochs_no_improve, best_model_state=None):
        stop = False
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
            best_model_state = self.state_dict()
            print("we keep model state of epoch {}".format(epoch + 1))
        else:
            epochs_no_improve += 1
        
        # if the epochs do not improve after "patience" consecutive epochs, we stop training
        if epochs_no_improve >= self.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            if best_model_state is not None:
                self.load_state_dict(best_model_state)
            # we change the number of epochs to the current epoch in order to plot the graph after
            self.num_epochs = epoch
            stop = True
        return best_loss, epochs_no_improve, stop

    def fit(self):
        train_losses = []
        train_accuracies = []
        best_loss = float('inf')
        epochs_no_improve = 0
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
                _, outputs = torch.max(outputs, dim=1)
                running_accuracy += accuracy_score(labels.cpu().numpy(), outputs.cpu().numpy())


            epoch_loss = running_loss / len(self.train_data)
            epoch_accuracy = running_accuracy / len(self.train_data)


            if self.validation_data is not None:
                print("the validation started")
                val_loss, val_accuracy = self.evaluate(self.validation_data)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                best_loss, epochs_no_improve, stop = self.early_stopping(loss, best_loss, epoch, epochs_no_improve)
                if stop is True:
                    print("Early stopping triggered, training stopped.")
                    break
            else:
                test_loss, test_accuracy = self.evaluate(self.test_data)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

                best_loss, epochs_no_improve, stop = self.early_stopping(loss, best_loss, epoch, epochs_no_improve)
                if stop is True:
                    print("Early stopping triggered, training stopped.")
                    break


            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            end = time.perf_counter()
            
            if self.validation_data is not None:
                print(f"Epoch {epoch+1}/{self.num_epochs}, "
        f"Train Loss: {epoch_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy:.4f}", 
        f"Time: {(end - start):.2f} seconds")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs}, "
        f"Train Loss: {epoch_loss:.4f}, "
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}", 
        f"Time: {(end - start):.2f} seconds")

            if self.save is not None:
                state_dict = self.state_dict()
                
                # Define the directory for this epoch
                epoch_dir = os.path.join(os.path.dirname(__file__), "trained_model", f"epoch{epoch+1}")

                # Create the directory if it doesn't exist
                os.makedirs(epoch_dir, exist_ok=True)  # exist_ok=True avoids error if it already exists
                
                # Define the path to save the model
                model_path = os.path.join(epoch_dir, "CNN" + f"_{epoch+1}" + ".pt")

                # Save the state_dict
                torch.save(state_dict, model_path)

            # torch.save(state_dict, os.path.join(os.path.dirname(_file_), "trained_model", "CNN.pt"))

        # it returns these but they are not that important, just for the plots
        if self.validation_data is None:
            return train_losses, train_accuracies, test_losses, test_accuracies 
        else:
            return train_losses, train_accuracies, val_losses, val_accuracies
        
    def predict(self):
        all_outputs = torch.empty((0, self.num_classes), device=self.device)
        all_labels = torch.empty((0,), device=self.device)
        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

        # Get predicted class indices
        _, predicted_classes = torch.max(all_outputs, dim=1)
        
        return predicted_classes, all_outputs

    def evaluate(self, loader):
        self.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                running_accuracy += accuracy_score(
                    labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy()
                )
        loss = running_loss / len(loader)
        accuracy = running_accuracy / len(loader)
        return loss, accuracy
        
        

    def prepare_data(self, train_data, train_labels, test_data, test_labels, val_data=None, val_labels=None, bs=32, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
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

        # Add validation data if provided
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



    def plot_loss(self, losses, accuracies, title: str):
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

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)