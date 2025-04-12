import time
import argparse
import os
import matplotlib.pyplot as plt

import numpy as np
import pickle
from custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split

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

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dilation ACE Model')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train (default: 1000)')
    parser.add_argument('--num_classes', type=int, default=28,
                      help='Number of classes (default: 28)')
    parser.add_argument('--model_name', type=str, default='Dilation',
                      help='Name of the model folder in ModelResults (default: Dilation)')
    parser.add_argument('--data_type', type=str, default='majmin',
                      help='Type of data to use (e.g., majmin, majmin7, majmininv, majmin7inv) (default: majmin)')
    parser.add_argument('--loss_hit_epochs', type=int, default=50,
                      help='Number of epochs without improvement before reducing learning rate (default: 50)')
    parser.add_argument('--early_stop_epochs', type=int, default=200,
                      help='Number of epochs without improvement before early stopping (default: 200)')
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    return device

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
    # Fill in to see about reducing network size
    def __init__(self, factor=2):
        self.pool = nn.MaxPooling(factor)

    def forward(self, x):
        return self.pool(x)
        

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
    def __init__(self, in_channels=128):
        super(Part2Dilation, self).__init__()
        # Starting dilation
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
        # Attention block: Squeeze-and-Excitation
        self.attention = SqueezeExcitationBlock(channels=in_channels, reduction=16)

        # Fully connected layers
        # Shape before flattening: torch.Size([8, 8192, 9, 24])
        flattened_size = 1 * 512 * 9 * 24  # Based on the final conv output
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

class SmallDilationModel(nn.Module):
    def __init__(self, num_classes):
        super(SmallDilationModel, self).__init__()

        # Max pooling layer
        self.pool = MaxPooling()
                
        # Part 1: Initial Convolution Layers with BatchNorm
        self.part1 = Part1BatchNorm()

        # Part 2: Convolution Layers with Dilation
        self.part2 = Part2Dilation(128/2) # Default without pooling is 128
        
        # Part 3: Attention Block + Fully Connected Layers
        self.part3 = Part3AttentionBlock(512/2, num_classes) # Default without pooling is 512
        
    def forward(self, x):
        x = self.part1(x)
        x = self.pool(x)
        x = self.part2(x)
        x = self.pool(x)
        x = self.part3(x)
        return x

def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=1000, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):    
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(epochs):
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

def save_loss_graphs(train_losses, val_losses, model_name, data_type, epochs):
    # Create ModelResults directory structure
    results_dir = os.path.join('ModelResults', model_name, data_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses\nModel: {model_name}, Dataset: {data_type}, Epochs: {epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, f'loss_plot_{data_type}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Loss plot saved to: {plot_path}")

def save_model_stats(model, macs, params, flops, gflops, accuracy, f1, model_name, data_type, args):
    # Create ModelResults directory structure
    results_dir = os.path.join('ModelResults', model_name, data_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create stats file path
    stats_path = os.path.join(results_dir, f'model_stats_{data_type}.txt')
    
    # Redirect stdout to capture model summary
    original_stdout = sys.stdout
    with open(stats_path, 'w') as f:
        sys.stdout = f
        
        print(f"Model Statistics for {model_name} - {data_type}")
        print("=" * 50)
        print("\nModel Configuration:")
        print(f"Number of classes: {args.num_classes}")
        print(f"Epochs: {args.epochs}")
        print(f"Loss hit epochs: {args.loss_hit_epochs}")
        print(f"Early stop epochs: {args.early_stop_epochs}")
        
        print("\nModel Performance Metrics:")
        print(f"MACs: {macs:,}")
        print(f"FLOPs: {flops:,}")
        print(f"GFLOPs: {gflops:.4f}")
        print(f"Parameters: {params:.2f}")
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nModel Summary:")
        summary(model)
        
        # Reset stdout
        sys.stdout = original_stdout
    
    print(f"Model statistics saved to: {stats_path}")

def main():
    start_time = time.time()
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training for {args.epochs} epochs with {args.num_classes} classes")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_type}")
    print(f"Loss hit epochs: {args.loss_hit_epochs}")
    print(f"Early stop epochs: {args.early_stop_epochs}")

    # Load the data and labels            
    print("Setting up data")
    
    data_path = os.path.join('data', args.data_type)
    data_file = os.path.join(data_path, f'{args.data_type}_data.pkl')
    labels_file = os.path.join(data_path, f'{args.data_type}_labels.pkl')

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Data files not found in {data_path}. Please ensure the data files exist.")

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    data = np.array(data)
    labels = np.array(labels)

    # Split into 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    # Convert the split data into PyTorch datasets
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    # Create DataLoaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # Set up the network
    print("Setting up network")

    model = SmallDilationModel(num_classes=args.num_classes).to(device)
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

    model_state, losses, val_losses = train(model, criterion, optimizer, train_dataloader, val_dataloader, 
                                          epochs=args.epochs, 
                                          loss_hit_epochs=args.loss_hit_epochs,
                                          early_stop_epochs=args.early_stop_epochs,
                                          device=device)
    
    # Evaluate model and get metrics
    accuracy, f1 = evaluate_model(model, test_dataloader, device)
    
    # Save loss graphs
    save_loss_graphs(losses, val_losses, args.model_name, args.data_type, args.epochs)
    
    # Save model statistics
    save_model_stats(model, macs, params, flops, gflops, accuracy, f1, args.model_name, args.data_type, args)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()