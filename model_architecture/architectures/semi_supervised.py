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

class SemiSupervisedChordExtractionCNN(nn.Module):
    def __init__(self, num_classes):
        super(SemiSupervisedChordExtractionCNN, self).__init__()
        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1)

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        # Shared fully connected layer
        self.fc1 = nn.Linear(36 * 3 * 18, 128)

        # Task-specific heads
        self.sequence_head = nn.Linear(128, 9)  # For sequence ordering (9 positions)
        self.classification_head = nn.Linear(128, num_classes)  # For chord classification

    def forward(self, x, task='classification'):
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm(x)

        # Shared feature extraction
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Task-specific output
        if task == 'sequence':
            return self.sequence_head(x)
        else:
            return self.classification_head(x) 