import torch
import torch.nn as nn
import torch.nn.functional as F
from model_architecture.utils.guassian_noise import GaussianNoise
from model_architecture.utils.multi_dilation_block import MultiDilationBlock

class LSTMMultiDilationChordCNNE(nn.Module):
    """
    LSTM Multi-Dilation Architecture E: Dilations [1, 3, 6] with softmax attention
    """
    def __init__(self, num_classes, input_height=9, input_width=24,
                 lstm_hidden_size=64, lstm_num_layers=2,
                 branch_out_channels=12, dilations=[1, 3, 6],
                 kernel_size=3, fc_size=128, dropout_rate=0.25):
        super(LSTMMultiDilationChordCNNE, self).__init__()

        if input_height <= 0 or input_width <= 0:
            raise ValueError("input_height and input_width must be positive integers.")

        self.input_height = input_height
        self.input_width = input_width
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.noise = GaussianNoise(std=0.01)
        self.batchnorm_in = nn.BatchNorm2d(1)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_width,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )

        # First Multi-Dilation Block - Softmax attention
        self.multi_dilation_block1 = MultiDilationBlock(
            in_channels=1,
            branch_out_channels=branch_out_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            attention_type='softmax'
        )
        block1_output_channels = self.multi_dilation_block1.total_out_channels

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Multi-Dilation Block - Softmax attention
        self.multi_dilation_block2 = MultiDilationBlock(
            in_channels=block1_output_channels,
            branch_out_channels=branch_out_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            attention_type='softmax'
        )
        block2_output_channels = self.multi_dilation_block2.total_out_channels

        # Additional Convolutional Layers
        self.conv1 = nn.Conv2d(block2_output_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)

        # Dropout layers
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Calculate the size after pooling and convolutions
        pooled_height = input_height // 2
        pooled_width = input_width // 2
        conv_output_feature_size = 256 * pooled_height * pooled_width

        # Fully Connected Layers
        self.fc1 = nn.Linear(conv_output_feature_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        # Input x expected shape: (batch_size, height, width)
        batch_size = x.size(0)

        # 1. Preprocessing
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm_in(x)

        # 2. LSTM processing
        x_lstm = x.squeeze(1)
        x_lstm = x_lstm.transpose(1, 2)
        
        # Initialize LSTM hidden state
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x_lstm, (h0, c0))
        
        # Reshape back to spatial format
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = lstm_out.unsqueeze(1)
        
        # Resize to match original spatial dimensions
        lstm_out = F.interpolate(lstm_out, size=(self.input_height, self.input_width), 
                                mode='bilinear', align_corners=False)

        # 3. First Multi-Dilation Block
        x = self.multi_dilation_block1(lstm_out)
        x = torch.relu(x)
        x = self.dropout(x)

        # 4. Pooling
        x = self.pool(x)

        # 5. Second Multi-Dilation Block
        x = self.multi_dilation_block2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # 6. Additional Convolutional Layers
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # 7. Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # 8. Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x 