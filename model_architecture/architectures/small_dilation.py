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

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        batch, channels, height, width = x.size()
        y = x.mean(dim=(2, 3))  # Squeeze height and width -> (batch, channels)
        
        # Excitation: Fully connected layers
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        
        # Scale: Reshape and multiply
        y = y.view(batch, channels, 1, 1)  # Reshape to (batch, channels, 1, 1)
        return x * y  # Scale input by attention weights

class MaxPooling(nn.Module):
    def __init__(self, factor=2):
        super(MaxPooling, self).__init__()
        self.pool = nn.MaxPool2d(factor)

    def forward(self, x):
        return self.pool(x)

class Part1BatchNorm(nn.Module):
    def __init__(self):
        super(Part1BatchNorm, self).__init__()
        self.noise = GaussianNoise(std=0.01)

        # Convolutional layers with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(9)
        self.dropout3 = nn.Dropout2d(p=0.25)
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, dilation=1)
        self.dropout6 = nn.Dropout2d(p=0.25)
        self.batchnorm6 = nn.BatchNorm2d(48)
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1, dilation=1)
        self.batchnorm7 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.noise(x.unsqueeze(1))
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.batchnorm4(F.relu(self.conv4(x)))
        x = self.batchnorm5(F.relu(self.conv5(x)))
        x = self.batchnorm6(F.relu(self.conv6(x)))
        x = self.dropout6(x)
        x = self.batchnorm7(F.relu(self.conv7(x)))
        return x

class Part2Dilation(nn.Module):
    def __init__(self, in_channels=128):
        super(Part2Dilation, self).__init__()
        self.conv8 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1, dilation=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, dilation=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1)
    
    def forward(self, x):
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        return x

class Part3AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Part3AttentionBlock, self).__init__()
        self.attention = SqueezeExcitationBlock(channels=in_channels, reduction=16)
        flattened_size = 1 * 512 * 9 * 24
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SmallDilationModel(nn.Module):
    def __init__(self, num_classes):
        super(SmallDilationModel, self).__init__()
        self.pool = MaxPooling()
        self.part1 = Part1BatchNorm()
        self.part2 = Part2Dilation(128//2)
        self.part3 = Part3AttentionBlock(512//2, num_classes)
        
    def forward(self, x):
        x = self.part1(x)
        x = self.pool(x)
        x = self.part2(x)
        x = self.pool(x)
        x = self.part3(x)
        return x 