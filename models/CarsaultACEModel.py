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
    parser = argparse.ArgumentParser(description='Train Carsault ACE Model')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train (default: 1000)')
    parser.add_argument('--num_classes', type=int, default=28,
                      help='Number of classes (default: 28)')
    parser.add_argument('--model_name', type=str, default='Baseline',
                      help='Name of the model folder in ModelResults (default: Baseline)')
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

    model = ChordExtractionCNN(num_classes=args.num_classes).to(device)
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