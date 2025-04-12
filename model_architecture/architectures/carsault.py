import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:  # Only add noise during training
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class ChordExtractionCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChordExtractionCNN, self).__init__()
        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1)

        # Convolutional layers with Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability

        # Fully connected layers
        self.fc1 = nn.Linear(36 * 3 * 18, 128)  # Adjusted for output of conv3
        self.fc2 = nn.Linear(128, num_classes)   # Output layer with num_classes neurons

    def forward(self, x):
        x = self.noise(x)       # Apply Gaussian noise
        x = x.unsqueeze(1)      # Add channel dimension
        x = self.batchnorm(x)   # Apply batch normalization to the input

        # Convolutional layers with ReLU activation and Dropout
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)    # Apply dropout after conv1
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)    # Apply dropout after conv2
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)    # Apply dropout after conv3

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)          # Output layer
        return x 