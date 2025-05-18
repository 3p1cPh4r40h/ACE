import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.guassian_noise import GaussianNoise

class SmallDilationSecondModel(nn.Module):
    """
    A CNN model for chord extraction using dilated convolution only in the second layer.
    This model uses a standard convolution in the first layer, a dilated convolution (rate=2)
    in the second layer, and a standard convolution in the third layer.
    """
    def __init__(self, num_classes, input_height=9, input_width=24):
        super(SmallDilationSecondModel, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1)

        # Layer 1: Standard convolution
        kernel_size1 = 3
        dilation1 = 1
        padding1 = math.floor(((kernel_size1 - 1) * dilation1) / 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12,
                               kernel_size=kernel_size1,
                               padding=padding1,
                               dilation=dilation1)
        self.dropout1 = nn.Dropout2d(p=0.25)

        # Layer 2: Dilated convolution (rate=2)
        kernel_size2 = 3
        dilation2 = 2
        padding2 = math.floor(((kernel_size2 - 1) * dilation2) / 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24,
                               kernel_size=kernel_size2,
                               padding=padding2,
                               dilation=dilation2)
        self.dropout2 = nn.Dropout2d(p=0.25)

        # Layer 3: Standard convolution
        kernel_size3 = 3
        dilation3 = 1
        padding3 = math.floor(((kernel_size3 - 1) * dilation3) / 2)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36,
                               kernel_size=kernel_size3,
                               padding=padding3,
                               dilation=dilation3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        # Fully Connected Layers
        conv_output_feature_size = 36 * self.input_height * self.input_width
        self.fc1 = nn.Linear(conv_output_feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input preprocessing
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm(x)

        # Convolutional layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x 