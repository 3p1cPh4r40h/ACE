import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16, activation="sigmoid"):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        else:
            print("[WARNING] Invalid Activiation Function in Squeeze Excitation Block, Defaulting to Sigmoid.")
            self.activation = nn.Sigmoid

    def forward(self, x):
        # Squeeze: Global Average Pooling
        batch, channels, height, width = x.size()
        y = x.mean(dim=(2, 3))  # Squeeze height and width -> (batch, channels)
        
        # Excitation: Fully connected layers
        y = F.relu(self.fc1(y))
        y = self.activation(self.fc2(y))
        
        # Scale: Reshape and multiply
        y = y.view(batch, channels, 1, 1)  # Reshape to (batch, channels, 1, 1)
        return x * y  # Scale input by attention weights