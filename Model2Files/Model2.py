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

parser = argparse.ArgumentParser(description="ACE Task Test")
parser.add_argument('--lr', default=2.1e-5, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():

    args = parser.parse_args()
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    
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

    class ChordExtractionCNN(nn.Module):
        def __init__(self, num_classes):
            super(ChordExtractionCNN, self).__init__()
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

            self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv10 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv11 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv12 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv13 = nn.Conv2d(in_channels=4096, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv14 = nn.Conv2d(in_channels=8192, out_channels=16384, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv15 = nn.Conv2d(in_channels=16384, out_channels=16384, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)

            self.maxpool = nn.MaxPool2d(2)

            self.conv16 = nn.Conv2d(in_channels=16384, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv17 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv18 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv19 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv20 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv21 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.conv22 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            self.dropout9 = nn.Dropout2d(p=0.25)
            self.conv23 = nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=3, padding=1, dilation=1)  # Output: (256, 9, 24)
            
            # Attention block: Squeeze-and-Excitation
            self.attention = SqueezeExcitationBlock(channels=8192, reduction=16)
            
            # Fully connected layers
            flattened_size = 8192 * 4 * 12  # Based on the final conv output
            self.fc1 = nn.Linear(flattened_size, 256)
            self.fc2 = nn.Linear(256, num_classes)  # Output layer with num_classes neurons
        
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
            x = F.relu(self.conv8(x))
            x = F.relu(self.conv9(x))
            x = F.relu(self.conv10(x))
            x = F.relu(self.conv11(x))
            x = F.relu(self.conv12(x))
            x = F.relu(self.conv13(x))
            
            x = F.relu(self.conv14(x))
            x = F.relu(self.conv15(x))
            x = self.maxpool(x)
            x = F.relu(self.conv16(x))
            x = F.relu(self.conv17(x))
            x = F.relu(self.conv18(x))
            x = F.relu(self.conv19(x))
            x = F.relu(self.conv20(x))
            x = F.relu(self.conv21(x))
            x = F.relu(self.conv22(x))
            
            
            x = self.dropout9(x)

            x = F.relu(self.conv23(x))
            
            # Apply attention block
            x = self.attention(x)
            
            # Flatten for the fully connected layers
            x = x.view(x.size(0), -1)
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

    num_classes = 39  # Number of unique labels
    model = ChordExtractionCNN(num_classes=num_classes).to(device)
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
