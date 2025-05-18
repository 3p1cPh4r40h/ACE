import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.guassian_noise import GaussianNoise

class SmallDilationFirstLastModel(nn.Module):
    """
    A CNN model for chord extraction using dilated convolutions only in the first and last layers.
    The middle layer uses standard convolution.
    """
    def __init__(self, num_classes, input_height=9, input_width=24):
        super(SmallDilationFirstLastModel, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1) # Normalize across the single input channel

        # Layer 1: Standard kernel size, dilation rate 2
        kernel_size1 = 3
        dilation1 = 2
        padding1 = math.floor(((kernel_size1 - 1) * dilation1) / 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12,
                               kernel_size=kernel_size1,
                               padding=padding1,
                               dilation=dilation1)
        self.dropout1 = nn.Dropout2d(p=0.25)

        # Layer 2: Standard convolution (no dilation)
        kernel_size2 = 3
        padding2 = 1  # Standard padding for 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24,
                               kernel_size=kernel_size2,
                               padding=padding2)
        self.dropout2 = nn.Dropout2d(p=0.25)

        # Layer 3: Standard kernel size, dilation rate 4
        kernel_size3 = 3
        dilation3 = 4
        padding3 = math.floor(((kernel_size3 - 1) * dilation3) / 2)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36,
                               kernel_size=kernel_size3,
                               padding=padding3,
                               dilation=dilation3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        # Fully Connected Layers
        conv_output_feature_size = 36 * self.input_height * self.input_width
        self.fc1 = nn.Linear(conv_output_feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)   # Output layer

    def forward(self, x):
        # Input x expected shape: (batch_size, height, width)

        # 1. Preprocessing
        x = self.noise(x)       # Apply Gaussian noise (only during training)
        x = x.unsqueeze(1)      # Add channel dimension: (batch, 1, height, width)
        x = self.batchnorm(x)   # Apply batch normalization

        # 2. First Dilated Convolutional Layer
        x = self.conv1(x)       # Shape: (batch, 12, height, width)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 3. Standard Convolutional Layer
        x = self.conv2(x)       # Shape: (batch, 24, height, width)
        x = torch.relu(x)
        x = self.dropout2(x)

        # 4. Last Dilated Convolutional Layer
        x = self.conv3(x)       # Shape: (batch, 36, height, width)
        x = torch.relu(x)
        x = self.dropout3(x)

        # 5. Flatten
        x = x.view(x.size(0), -1) # Shape: (batch, 36 * height * width)

        # 6. Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)          # Output layer (logits)

        return x 