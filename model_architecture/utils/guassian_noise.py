import torch
import torch.nn as nn

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