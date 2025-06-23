import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.guassian_noise import GaussianNoise
from model_architecture.utils.multi_dilation_block import MultiDilationBlock

class DeepSemiSupervisedMultiDilationChordCNN(nn.Module):
    """
    A deeper semi-supervised CNN model for chord extraction using multiple multi-dilation blocks
    with early squeeze and softmax activation, combined with sequence ordering pre-training.
    """
    def __init__(self, num_classes, input_height=9, input_width=24,
                 branch_out_channels=16, # Increased channels per dilation branch
                 dilations=[2, 4, 8],     # Dilation rates for the block
                 kernel_size=3,           # Kernel size for dilated convs
                 fc_size=256,             # Increased size of the first FC layer
                 dropout_rate=0.3):       # Slightly increased dropout
        super(DeepSemiSupervisedMultiDilationChordCNN, self).__init__()

        if input_height <= 0 or input_width <= 0:
             raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm_in = nn.BatchNorm2d(1)

        # --- First Multi-Dilation Convolutional Block ---
        self.multi_dilation_block1 = MultiDilationBlock(
            in_channels=1,
            branch_out_channels=branch_out_channels,
            dilations=dilations,
            kernel_size=kernel_size
        )
        block1_output_channels = self.multi_dilation_block1.total_out_channels

        # --- Second Multi-Dilation Block (deeper) ---
        self.multi_dilation_block2 = MultiDilationBlock(
            in_channels=block1_output_channels,
            branch_out_channels=branch_out_channels,
            dilations=dilations,
            kernel_size=kernel_size
        )
        block2_output_channels = self.multi_dilation_block2.total_out_channels

        # --- Additional Convolutional Layers ---
        self.conv1 = nn.Conv2d(block2_output_channels, block2_output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(block2_output_channels, block2_output_channels, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(block1_output_channels)
        self.batchnorm2 = nn.BatchNorm2d(block2_output_channels)
        self.batchnorm3 = nn.BatchNorm2d(block2_output_channels)
        self.batchnorm4 = nn.BatchNorm2d(block2_output_channels)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # --- Shared Feature Extraction ---
        conv_output_feature_size = block2_output_channels * self.input_height * self.input_width
        
        self.shared_fc1 = nn.Linear(conv_output_feature_size, fc_size)
        self.shared_fc2 = nn.Linear(fc_size, fc_size // 2)

        # --- Task-Specific Heads ---
        # Sequence ordering head (for semi-supervised pre-training)
        self.sequence_head = nn.Linear(fc_size // 2, input_height * input_width)
        
        # Classification head (for chord classification)
        self.classification_head1 = nn.Linear(fc_size // 2, fc_size // 4)
        self.classification_head2 = nn.Linear(fc_size // 4, num_classes)

        # Additional dropout for FC layers
        self.fc_dropout1 = nn.Dropout(p=0.4)
        self.fc_dropout2 = nn.Dropout(p=0.4)

    def forward(self, x, task='classification'):
        # Input x expected shape: (batch_size, height, width)

        # 1. Preprocessing
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm_in(x)

        # 2. First Multi-Dilation Block
        x = self.multi_dilation_block1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 3. Second Multi-Dilation Block
        x = self.multi_dilation_block2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # 4. Additional Convolutional Layers
        x = self.conv1(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.conv2(x)
        x = self.batchnorm4(x)
        x = torch.relu(x)

        # 5. Early Squeeze with Softmax
        x = F.softmax(x, dim=1)

        # 6. Flatten
        x = x.view(x.size(0), -1)

        # 7. Shared Feature Extraction
        x = self.shared_fc1(x)
        x = torch.relu(x)
        x = self.fc_dropout1(x)
        
        x = self.shared_fc2(x)
        x = torch.relu(x)
        x = self.fc_dropout2(x)

        # 8. Task-Specific Output
        if task == 'sequence':
            # For sequence ordering pre-training
            return self.sequence_head(x).view(-1, self.input_height, self.input_width)
        else:
            # For chord classification
            x = self.classification_head1(x)
            x = torch.relu(x)
            return self.classification_head2(x) 