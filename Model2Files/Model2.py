import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import argparse
import os
import time

from torch.distributed.pipeline.sync import Pipe

parser = argparse.ArgumentParser(description="ACE Task Test")
parser.add_argument('--lr', default=2.1e-5, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():
    
    args = parser.parse_args()

    class CustomDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform
            self.label_to_index = {label:idx for idx , label in enumerate(set(labels))}

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = torch.tensor(self.data[idx], dtype=torch.float32)
            label = self.label_to_index[self.labels[idx]]
            if self.transform:
                sample = self.transform(sample)
            return sample, label

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
    class ConvPart1(nn.Module):
        def __init__(self):
            super(ConvPart1, self).__init__()
            self.noise = GaussianNoise(std=0.01)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(3)
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(6)
            self.conv3 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, padding=1)
            self.batchnorm3 = nn.BatchNorm2d(9)
            self.dropout3 = nn.Dropout2d(p=0.25)
        
        def forward(self, x):
            x = self.noise(x.unsqueeze(1))
            x = self.batchnorm1(F.relu(self.conv1(x)))
            x = self.batchnorm2(F.relu(self.conv2(x)))
            x = self.batchnorm3(F.relu(self.conv3(x)))
            x = self.dropout3(x)
            return x

    class ConvPart2(nn.Module):
        def __init__(self):
            super(ConvPart2, self).__init__()
            self.conv4 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, padding=1)
            self.batchnorm4 = nn.BatchNorm2d(12)
            # Continue adding other layers up to the attention block
            self.conv14 = nn.Conv2d(in_channels=8192, out_channels=16384, kernel_size=3, padding=1)
            self.attention = SqueezeExcitationBlock(channels=8192, reduction=16)
        
        def forward(self, x):
            # Apply convolutional layers and the attention block
            x = self.batchnorm4(F.relu(self.conv4(x)))
            x = F.relu(self.conv14(x))
            x = self.attention(x)
            return x

    class MLPPart(nn.Module):
        def __init__(self, num_classes):
            super(MLPPart, self).__init__()
            flattened_size = 8192 * 4 * 12  # Adjust based on actual output dimensions
            self.fc1 = nn.Linear(flattened_size, 256)
            self.fc2 = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use GPU if available
        else:
            device = torch.device("cpu")  # Fallback to CPU
        return device

    device = get_device()
    print(f"Using device: {device}")


    print("Setting up data")

    # Load the data and labels
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    # Create DataLoaders for training, testing and validation
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=False)

    print("Making model")
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    
    part1 = ConvPart1().to('cuda:0')
    part2 = ConvPart2().to('cuda:1')
    part3 = MLPPart(num_classes=39).to('cuda:2')

    # Sequential pipeline
    model = nn.Sequential(part1, part2, part3)
    model = Pipe(model, chunks=8)

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification (implied softmax)
    optimizer = optim.Adam(model.parameters(), lr=2.1e-5)
    print(summary(model, (1,9,24)))


    num_epochs = 1000  # Set number of epochs (1000 recommended by Carsault et. al.)
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
            torch.save(model.state_dict(), 'bestmodel2.pth.tar')
            worse_loss = 0
            early_stop = 0
        else:
            worse_loss += 1
            early_stop += 1

        if worse_loss >= 49:
            print('Weight Optimization Hit')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            worse_loss = 0

        if early_stop >= 199:
            print('Ending Training Early')
            break

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
