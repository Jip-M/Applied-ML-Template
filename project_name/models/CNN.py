import torch.nn as nn
import torch.nn.functional as F
import torch

class AudioCNN(nn.Module):
    def __init__(self, num_classes: int):
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