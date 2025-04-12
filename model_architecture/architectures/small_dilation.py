import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Needed for calculating padding

class GaussianNoise(nn.Module):
    """Applies additive Gaussian noise."""
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:  # Only add noise during training
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class SmallDilationModel(nn.Module):
    """
    A CNN model for chord extraction using dilated convolutions.

    This model replaces standard convolutions with dilated ones to increase
    the receptive field without reducing spatial resolution via pooling.
    'Same' padding is used in convolutional layers to maintain spatial dimensions.
    """
    # We need input dimensions to calculate the size of the first fully connected layer
    def __init__(self, num_classes, input_height=9, input_width=24):
        super(SmallDilationModel, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1) # Normalize across the single input channel

        # --- Dilated Convolutional Layers ---
        # We use padding to maintain the height and width dimensions ('same' padding).
        # Padding calculation for 'same' padding with stride=1:
        # padding = floor(((kernel_size - 1) * dilation) / 2)
        # Note: PyTorch Conv2d padding can be a single int (applied to all sides)
        # or a tuple (pad_h, pad_w). We'll use a single int assuming symmetric kernels/padding.

        # Layer 1: Standard kernel size, standard dilation (1)
        kernel_size1 = 3
        dilation1 = 1
        padding1 = math.floor(((kernel_size1 - 1) * dilation1) / 2) # (3-1)*1 / 2 = 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12,
                               kernel_size=kernel_size1,
                               padding=padding1,
                               dilation=dilation1)
        self.dropout1 = nn.Dropout2d(p=0.25)

        # Layer 2: Standard kernel size, increased dilation (e.g., 2)
        kernel_size2 = 3
        dilation2 = 2
        # Effective kernel size = k + (k-1)*(d-1) = 3 + 2*1 = 5
        padding2 = math.floor(((kernel_size2 - 1) * dilation2) / 2) # (3-1)*2 / 2 = 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24,
                               kernel_size=kernel_size2,
                               padding=padding2,
                               dilation=dilation2)
        self.dropout2 = nn.Dropout2d(p=0.25)

        # Layer 3: Standard kernel size, further increased dilation (e.g., 4)
        kernel_size3 = 3
        dilation3 = 4
        # Effective kernel size = k + (k-1)*(d-1) = 3 + 2*3 = 9
        padding3 = math.floor(((kernel_size3 - 1) * dilation3) / 2) # (3-1)*4 / 2 = 4
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36,
                               kernel_size=kernel_size3,
                               padding=padding3,
                               dilation=dilation3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        # --- Fully Connected Layers ---
        # Calculate the flattened size after the conv layers.
        # Since we used 'same' padding, the height and width remain unchanged.
        # The number of channels is the output channels of the last conv layer (36).
        conv_output_feature_size = 36 * self.input_height * self.input_width

        self.fc1 = nn.Linear(conv_output_feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)   # Output layer

    def forward(self, x):
        # Input x expected shape: (batch_size, height, width)
        # Example: (batch_size, num_bins, num_frames) for a spectrogram/chromagram

        # 0. Input Checks (optional but good practice)
        # assert x.ndim == 3, f"Expected input ndim=3 (batch, H, W), got {x.ndim}"
        # assert x.shape[1] == self.input_height, f"Expected input height {self.input_height}, got {x.shape[1]}"
        # assert x.shape[2] == self.input_width, f"Expected input width {self.input_width}, got {x.shape[2]}"

        # 1. Preprocessing
        x = self.noise(x)       # Apply Gaussian noise (only during training)
        x = x.unsqueeze(1)      # Add channel dimension: (batch, 1, height, width)
        x = self.batchnorm(x)   # Apply batch normalization

        # 2. Dilated Convolutional Layers
        # Activation -> Convolution -> Dropout sequence
        x = self.conv1(x)       # Shape: (batch, 12, height, width)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)       # Shape: (batch, 24, height, width)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)       # Shape: (batch, 36, height, width)
        x = torch.relu(x)
        x = self.dropout3(x)

        # 3. Flatten
        # Flatten the output of the conv layers for the fully connected layers
        # x.shape[0] is the batch size
        x = x.view(x.size(0), -1) # Shape: (batch, 36 * height * width)

        # 4. Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        # Note: Dropout could optionally be added after fc1 as well
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)          # Output layer (logits)

        return x