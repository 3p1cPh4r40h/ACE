import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.guassian_noise import GaussianNoise
from model_architecture.utils.multi_dilation_block import MultiDilationBlock

class HybridMultiDilationEarlySqueezeSoftmaxChordCNN(nn.Module):
    """
    A hybrid CNN model for chord extraction that combines the best performing architecture
    (Multi Dilation Early Squeeze Softmax) with features from the second best model
    (Multi Dilation 4816), including larger dilation rates and enhanced feature extraction.
    """
    def __init__(self, num_classes, input_height=9, input_width=24,
                 branch_out_channels=16, # Increased channels per dilation branch
                 dilations_small=[2, 4, 8],     # Small dilation rates (from best model)
                 dilations_large=[4, 8, 16],    # Large dilation rates (from second best model)
                 kernel_size=3,           # Kernel size for dilated convs
                 fc_size=256,             # Increased size of the first FC layer
                 dropout_rate=0.3):       # Slightly increased dropout
        super(HybridMultiDilationEarlySqueezeSoftmaxChordCNN, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm_in = nn.BatchNorm2d(1)

        # --- First Multi-Dilation Block (Small Dilations) ---
        self.multi_dilation_block_small = MultiDilationBlock(
            in_channels=1,
            branch_out_channels=branch_out_channels,
            dilations=dilations_small,
            kernel_size=kernel_size
        )
        small_block_output_channels = self.multi_dilation_block_small.total_out_channels

        # --- Second Multi-Dilation Block (Large Dilations) ---
        self.multi_dilation_block_large = MultiDilationBlock(
            in_channels=1,
            branch_out_channels=branch_out_channels,
            dilations=dilations_large,
            kernel_size=kernel_size
        )
        large_block_output_channels = self.multi_dilation_block_large.total_out_channels

        # --- Feature Fusion Layer ---
        total_channels = small_block_output_channels + large_block_output_channels
        self.fusion_conv = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1)
        
        # --- Additional Convolutional Layers ---
        self.conv1 = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.batchnorm_small = nn.BatchNorm2d(small_block_output_channels)
        self.batchnorm_large = nn.BatchNorm2d(large_block_output_channels)
        self.batchnorm_fusion = nn.BatchNorm2d(total_channels)
        self.batchnorm1 = nn.BatchNorm2d(total_channels)
        self.batchnorm2 = nn.BatchNorm2d(total_channels)
        self.batchnorm3 = nn.BatchNorm2d(total_channels)
        self.batchnorm4 = nn.BatchNorm2d(total_channels)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

        # --- Deeper Fully Connected Layers ---
        conv_output_feature_size = total_channels * self.input_height * self.input_width
        
        self.fc1 = nn.Linear(conv_output_feature_size, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size // 2)
        self.fc3 = nn.Linear(fc_size // 2, fc_size // 4)
        self.fc4 = nn.Linear(fc_size // 4, num_classes)

        # Additional dropout for FC layers
        self.fc_dropout1 = nn.Dropout(p=0.4)
        self.fc_dropout2 = nn.Dropout(p=0.4)

    def forward(self, x):
        # Input x expected shape: (batch_size, height, width)

        # 1. Preprocessing
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm_in(x)

        # 2. Parallel Multi-Dilation Blocks
        x_small = self.multi_dilation_block_small(x)
        x_small = self.batchnorm_small(x_small)
        x_small = torch.relu(x_small)
        x_small = self.dropout1(x_small)

        x_large = self.multi_dilation_block_large(x)
        x_large = self.batchnorm_large(x_large)
        x_large = torch.relu(x_large)
        x_large = self.dropout2(x_large)

        # 3. Feature Fusion
        x = torch.cat([x_small, x_large], dim=1)  # Concatenate along channel dimension
        x = self.fusion_conv(x)
        x = self.batchnorm_fusion(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # 4. Additional Convolutional Layers with Residual Connections
        identity = x
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x + identity)  # Residual connection
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)

        # 5. Early Squeeze with Softmax
        x = F.softmax(x, dim=1)

        # 6. Flatten
        x = x.view(x.size(0), -1)

        # 7. Deeper Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_dropout2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        
        x = self.fc4(x)

        return x 