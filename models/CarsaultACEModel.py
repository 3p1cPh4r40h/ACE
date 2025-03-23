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

from sklearn.metrics import accuracy_score, f1_score

import sys

# Add this before calling summary() to force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')


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

class ChordExtractionCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChordExtractionCNN, self).__init__()
        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1)

        # Convolutional layers with Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.25)  # Dropout with 25% probability

        # Fully connected layers
        self.fc1 = nn.Linear(36 * 3 * 18, 128)  # Adjusted for output of conv3
        self.fc2 = nn.Linear(128, num_classes)   # Output layer with num_classes neurons

    def forward(self, x):
        x = self.noise(x)       # Apply Gaussian noise
        x = x.unsqueeze(1)      # Add channel dimension
        x = self.batchnorm(x)   # Apply batch normalization to the input

        # Convolutional layers with ReLU activation and Dropout
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)    # Apply dropout after conv1
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)    # Apply dropout after conv2
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)    # Apply dropout after conv3

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)          # Output layer
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

model = ChordExtractionCNN(num_classes=39).to(device)
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


num_epochs = 1000  # Set number of epochs (1000 recommended by Carsault et. al.)

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

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():  # No gradients needed
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)  # Get predicted class

            y_true.extend(labels.cpu().numpy())  # Move labels to CPU
            y_pred.extend(predicted.cpu().numpy())  # Move predictions to CPU

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')  # 'macro' for multi-class

    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1

evaluate_model(model, test_dataloader, device)
