import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.guassian_noise import GaussianNoise
from model_architecture.utils.squeeze_excitation import SqueezeExcitationBlock

class MultiDilationBlock(nn.Module):
    """
    Applies multiple dilated convolutions in parallel to the input
    and concatenates the results. Uses 'same' padding.
    """
    def __init__(self, in_channels, branch_out_channels, dilations, kernel_size=3):
        """
        Args:
            in_channels (int): Number of input channels.
            branch_out_channels (int): Number of output channels *per branch*.
            dilations (list or tuple of int): Dilation rates for each parallel branch.
            kernel_size (int): Kernel size for the convolutions.
        """
        super(MultiDilationBlock, self).__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()

        for dilation in dilations:
            # Calculate 'same' padding for this branch
            padding = math.floor(((kernel_size - 1) * dilation) / 2)
            self.branches.append(
                nn.Conv2d(in_channels,
                          branch_out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation)
            )

        # The total number of output channels after concatenation
        self.total_out_channels = len(dilations) * branch_out_channels

    def forward(self, x):
        branch_outputs = []
        for branch_conv in self.branches:
            branch_outputs.append(branch_conv(x))

        # Concatenate along the channel dimension (dim=1)
        # Input x shape: (batch, in_channels, H, W)
        # Output shape: (batch, total_out_channels, H, W)
        return torch.cat(branch_outputs, dim=1)

class MidSqueezeChordCNN(nn.Module):
    """
    A CNN model for chord extraction using a multi-dilation block
    with mid attention applied between the multi-dilation block and additional convolution.
    """
    def __init__(self, num_classes, input_height=9, input_width=24,
                 branch_out_channels=12, # Channels per dilation branch
                 dilations=[1, 2, 4],     # Dilation rates for the block
                 kernel_size=3,           # Kernel size for dilated convs
                 fc_size=128,             # Size of the first FC layer
                 dropout_rate=0.25):      # Dropout probability
        super(MidSqueezeChordCNN, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm_in = nn.BatchNorm2d(1) # Normalize across the single input channel

        # --- Multi-Dilation Convolutional Block ---
        self.multi_dilation_block = MultiDilationBlock(
            in_channels=1, # Takes the single channel input after batchnorm
            branch_out_channels=branch_out_channels,
            dilations=dilations,
            kernel_size=kernel_size
        )
        # Total channels output by the block
        block_output_channels = self.multi_dilation_block.total_out_channels

        # --- Mid Attention Block ---
        self.mid_attention = SqueezeExcitationBlock(
            channels=block_output_channels,
            reduction=16,  # Standard reduction ratio
            activation="sigmoid"  # Use sigmoid for attention weights
        )

        # --- Additional Convolution Layer ---
        self.conv_after_dilation = nn.Conv2d(
            in_channels=block_output_channels,  # Output channels from the multi-dilation block
            out_channels=block_output_channels,  # You can adjust this as needed
            kernel_size=3,                       # Kernel size for the convolution
            padding=1                            # Padding to maintain spatial dimensions
        )

        # Optional: Batchnorm after concatenating branches
        self.batchnorm_block_out = nn.BatchNorm2d(block_output_channels)

        # Dropout after the multi-dilation block and activation
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # --- Fully Connected Layers ---
        # Calculate the flattened size after the multi-dilation block.
        # Spatial dimensions (H, W) are maintained by 'same' padding.
        conv_output_feature_size = block_output_channels * self.input_height * self.input_width

        self.fc1 = nn.Linear(conv_output_feature_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)   # Output layer

    def forward(self, x):
        # Input x expected shape: (batch_size, height, width)

        # 1. Preprocessing
        x = self.noise(x)       # Apply Gaussian noise (only during training)
        x = x.unsqueeze(1)      # Add channel dimension: (batch, 1, H, W)
        x = self.batchnorm_in(x) # Apply batch normalization

        # 2. Multi-Dilation Convolutional Block
        x = self.multi_dilation_block(x) # Shape: (batch, block_output_channels, H, W)
        
        # 3. Mid Attention
        x = self.mid_attention(x)  # Apply channel-wise attention after multi-dilation
        
        # 4. Additional Convolution and Processing
        x = self.conv_after_dilation(x)  # Apply the additional convolution layer
        x = self.batchnorm_block_out(x)
        x = torch.relu(x)       # Apply activation *after* the block (or within branches)
        x = self.dropout(x)     # Apply dropout

        # 5. Flatten
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape: (batch, block_output_channels * H * W)

        # 6. Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        # Optional: Add dropout after fc1
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)          # Output layer (logits)

        return x 