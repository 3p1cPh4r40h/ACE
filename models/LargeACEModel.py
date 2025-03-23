import time

import numpy as np
import pickle
from custom_dataset import CustomDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from thop import profile

from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    return device

device = get_device()
print(f"Using device: {device}")

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

# Define the parts of the model
class Part1BatchNorm(nn.Module):
    def __init__(self):
        super(Part1BatchNorm, self).__init__()
        self.noise = GaussianNoise(std=0.01)

        # Convolutional layers with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)  # Output: (3, 9, 24)
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)  # Output: (6, 9, 24)
        self.batchnorm2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, padding=1)  # Output: (9, 9, 24)
        self.batchnorm3 = nn.BatchNorm2d(9)
        self.dropout3 = nn.Dropout2d(p=0.25)
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, padding=1)  # Output: (12, 9, 24)
        self.batchnorm4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)  # Output: (24, 9, 24)
        self.batchnorm5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, dilation=1)  # Output: (48, 9, 24)
        self.dropout6 = nn.Dropout2d(p=0.25)
        self.batchnorm6 = nn.BatchNorm2d(48)
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1, dilation=1)  # Output: (128, 9, 24)
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
    def __init__(self):
        super(Part2Dilation, self).__init__()
        # Starting dilation
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, dilation=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, dilation=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1)
    
    def forward(self, x):
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        return x

class Part3Dilation(nn.Module):
    def __init__(self):
        super(Part3Dilation, self).__init__()
        # Continuing dilation
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        return x

class Part4Dilation(nn.Module):
    def __init__(self):
        super(Part4Dilation, self).__init__()
        # Continuing dilation
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1)
    
    def forward(self, x):
        x = F.relu(self.conv12(x))
        return x

class Part5Dilation(nn.Module):
    def __init__(self):
        super(Part5Dilation, self).__init__()
        # Continuing convolution layes with dilation
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, dilation=1) 

    def forward(self, x):
        x = F.relu(self.conv13(x))
        return x

class Part6Dilation(nn.Module):
    def __init__(self):
        super(Part6Dilation, self).__init__()
        # Continuing convolution layes with dilation
        self.conv14 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, dilation=1) 

    def forward(self, x):
        x = F.relu(self.conv14(x))
        return x

class Part7Dilation(nn.Module):
    def __init__(self):
        super(Part7Dilation, self).__init__()
        # Continuing convolution layes with dilation
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, dilation=1) 

    def forward(self, x):
        x = F.relu(self.conv15(x))
        return x

class Part8Pooling(nn.Module):
    def __init__(self):
        super(Part8Pooling, self).__init__()
        # Max pooling layer to reduce size while maintaining information
        self.maxpool = nn.MaxPool2d(2)

        self.conv16 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, dilation=1) 

    def forward(self, x):
        x = self.maxpool(x)

        x = F.relu(self.conv16(x))
        return x

class Part9Dilation(nn.Module):
    def __init__(self):
        super(Part9Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv17 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv17(x))
        return x

class Part10Dilation(nn.Module):
    def __init__(self):
        super(Part10Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv18(x))
        return x

class Part11Dilation(nn.Module):
    def __init__(self):
        super(Part11Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv19 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv19(x))

        return x

class Part12Dilation(nn.Module):
    def __init__(self):
        super(Part12Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv20(x))

        return x

class Part13Dilation(nn.Module):
    def __init__(self):
        super(Part13Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv21(x))

        return x

class Part14Dilation(nn.Module):
    def __init__(self):
        super(Part14Dilation, self).__init__()
        # Stacked conv layers of the same size with dilation
        self.conv22 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, padding=1, dilation=1) 
        

    def forward(self, x):
        x = F.relu(self.conv22(x))

        return x

class Part15Dilation(nn.Module):
    def __init__(self):
        super(Part15Dilation, self).__init__()
        # Dropout to prevent overfitting
        self.dropout9 = nn.Dropout2d(p=0.25)
        # Stacked conv layers of the same size with dilation
        self.conv23 = nn.Conv2d(in_channels=4096, out_channels=8192, kernel_size=3, padding=1, dilation=1) 


    def forward(self, x):
        x = self.dropout9(x)
        x = F.relu(self.conv23(x))

        return x

class Part16AttentionBlock(nn.Module):
    def __init__(self, num_classes):
        super(Part16AttentionBlock, self).__init__()
        # Attention block: Squeeze-and-Excitation
        self.attention = SqueezeExcitationBlock(channels=8192, reduction=16)

        # Fully connected layers
        # Shape before flattening: torch.Size([8, 8192, 9, 24])
        flattened_size = 1 * 8192 * 4 * 12  # Based on the final conv output
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Output layer with num_classes neurons
        
    def forward(self, x):
        # Apply attention block
        x = self.attention(x)
        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LargeACEModel(nn.Module):
    def __init__(self, num_classes):
        super(LargeACEModel, self).__init__()
        
        # Part 1: Initial Convolution Layers with BatchNorm
        self.part1 = Part1BatchNorm()
        
        # Part 2-7: Convolution Layers with Dilation
        self.part2 = Part2Dilation()
        self.part3 = Part3Dilation()
        self.part4 = Part4Dilation()
        self.part5 = Part5Dilation()
        self.part6 = Part6Dilation()
        self.part7 = Part7Dilation()
        
        # Part 8: Pooling Layer
        self.part8 = Part8Pooling()
        
        # Part 9-15: More Dilation Layers
        self.part9 = Part9Dilation()
        self.part10 = Part10Dilation()
        self.part11 = Part11Dilation()
        self.part12 = Part12Dilation()
        self.part13 = Part13Dilation()
        self.part14 = Part14Dilation()
        self.part15 = Part15Dilation()
        
        # Part 16: Attention Block + Fully Connected Layers
        self.part16 = Part16AttentionBlock(num_classes)
        
    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        x = self.part5(x)
        x = self.part6(x)
        x = self.part7(x)
        x = self.part8(x)
        x = self.part9(x)
        x = self.part10(x)
        x = self.part11(x)
        x = self.part12(x)
        x = self.part13(x)
        x = self.part14(x)
        x = self.part15(x)
        x = self.part16(x)
        return x

# Load the data and labels            
print("Setting up data")

with open('data/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('data/val_dataset.pkl', 'rb') as f:
    val_dataset = pickle.load(f)
with open('data/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# Create DataLoaders for training, testing and validation
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=False)

# Set up the network
print("Setting up network")

model = LargeACEModel(num_classes=39).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2.1e-5)

inj=torch.randn(1,9,24).to(device)
# Get MACs and parameters
macs, params = profile(model, inputs=(inj, ))

# Convert MACs to GFLOPs (1e9 FLOPs = 1 GFLOP)
flops = 2 * macs
gflops = flops / 1e9

# Print results
print(f"MACs: {macs:,}")
print(f"FLOPs: {flops:,}")
print(f"GFLOPs: {gflops:.4f}")
print(f"Parameters: {params:.2f}")

summary(model)


num_epochs = 1  # Set number of epochs (1000 recommended by Carsault et. al.)

def train(model, data, epochs=10, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):    
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use DataLoader for batching
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update model parameters

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()


        current_loss = running_loss / len(train_dataloader)
        current_val_loss = val_loss / len(val_dataloader)
        losses.append(current_loss)
        val_losses.append(current_val_loss)
        print(f"Epoch {epoch+1}, Training Loss: {current_loss}, Validation Loss: {current_val_loss}")

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_model_state = model.state_dict()
            worse_loss = 0
            early_stop = 0
        else:
            worse_loss += 1
            early_stop += 1

        if worse_loss >= loss_hit_epochs-1:
            print('Weight Optimization Hit')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            worse_loss = 0

        if early_stop >= early_stop_epochs-1:
            print('Ending Training Early')
            break
    
    return [model.state_dict(), losses, val_losses]

model_state, losses, val_losses = train(model, train_dataloader, num_epochs, device=device)