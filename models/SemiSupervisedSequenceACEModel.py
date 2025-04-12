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
    parser = argparse.ArgumentParser(description='Train Semi-Supervised Sequence ACE Model')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train (default: 1000)')
    parser.add_argument('--num_classes', type=int, default=28,
                      help='Number of classes (default: 28)')
    parser.add_argument('--model_name', type=str, default='SemiSupervised',
                      help='Name of the model folder in ModelResults (default: SemiSupervised)')
    parser.add_argument('--data_type', type=str, default='majmin',
                      help='Type of data to use (e.g., majmin, majmin7, majmininv, majmin7inv) (default: majmin)')
    parser.add_argument('--loss_hit_epochs', type=int, default=50,
                      help='Number of epochs without improvement before reducing learning rate (default: 50)')
    parser.add_argument('--early_stop_epochs', type=int, default=200,
                      help='Number of epochs without improvement before early stopping (default: 200)')
    parser.add_argument('--pretrain_epochs', type=int, default=1000,
                      help='Number of epochs for sequence ordering pre-training (default: 1000)')
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

class SemiSupervisedChordExtractionCNN(nn.Module):
    def __init__(self, num_classes):
        super(SemiSupervisedChordExtractionCNN, self).__init__()
        self.noise = GaussianNoise(std=0.01)
        self.batchnorm = nn.BatchNorm2d(1)

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        # Shared fully connected layer
        self.fc1 = nn.Linear(36 * 3 * 18, 128)

        # Task-specific heads
        self.sequence_head = nn.Linear(128, 9)  # For sequence ordering (9 positions)
        self.classification_head = nn.Linear(128, num_classes)  # For chord classification

    def forward(self, x, task='classification'):
        x = self.noise(x)
        x = x.unsqueeze(1)
        x = self.batchnorm(x)

        # Shared feature extraction
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Task-specific output
        if task == 'sequence':
            return self.sequence_head(x)
        else:
            return self.classification_head(x)

def shuffle_sequence(sequence, device):
    """Shuffle a sequence and return both shuffled sequence and original indices"""
    batch_size = sequence.size(0)
    # Ensure indices are in range [0, batch_size-1]
    indices = torch.randperm(batch_size, device=device)
    return sequence[indices], indices

def train_sequence_ordering(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=1000, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Shuffle sequences and get original indices
            shuffled_inputs, original_indices = shuffle_sequence(inputs, device)
            
            optimizer.zero_grad()
            outputs = model(shuffled_inputs, task='sequence')
            
            # Ensure indices are properly formatted for CrossEntropyLoss
            # The model outputs 9 positions, so we need to ensure indices are in [0,8]
            original_indices = original_indices % 9  # Ensure indices are in range [0,8]
            
            loss = criterion(outputs, original_indices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_dataloader:
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
                shuffled_inputs, original_indices = shuffle_sequence(inputs, device)
                outputs = model(shuffled_inputs, task='sequence')
                
                # Ensure indices are properly formatted for CrossEntropyLoss
                original_indices = original_indices % 9  # Ensure indices are in range [0,8]
                
                loss = criterion(outputs, original_indices)
                val_loss += loss.item()

        current_loss = running_loss / len(train_dataloader)
        current_val_loss = val_loss / len(val_dataloader)
        losses.append(current_loss)
        val_losses.append(current_val_loss)
        print(f"Sequence Ordering Epoch {epoch+1}, Training Loss: {current_loss}, Validation Loss: {current_val_loss}")

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_weights = model.state_dict()
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
            print('Ending Sequence Ordering Training Early')
            break

    return [best_weights, losses, val_losses]

def train_classification(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=1000, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, task='classification')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, task='classification')
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        current_loss = running_loss / len(train_dataloader)
        current_val_loss = val_loss / len(val_dataloader)
        losses.append(current_loss)
        val_losses.append(current_val_loss)
        print(f"Classification Epoch {epoch+1}, Training Loss: {current_loss}, Validation Loss: {current_val_loss}")

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_weights = model.state_dict()
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
            print('Ending Classification Training Early')
            break

    return [best_weights, losses, val_losses]

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, task='classification')
            predicted = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1

def save_loss_graphs(train_losses, val_losses, model_name, data_type, epochs, phase='classification'):
    results_dir = os.path.join('ModelResults', model_name, data_type)
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title(f'{phase.capitalize()} Training and Validation Losses\nModel: {model_name}, Dataset: {data_type}, Epochs: {epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, f'loss_plot_{data_type}_{phase}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Loss plot saved to: {plot_path}")

def save_model_stats(model, macs, params, flops, gflops, accuracy, f1, model_name, data_type, args):
    results_dir = os.path.join('ModelResults', model_name, data_type)
    os.makedirs(results_dir, exist_ok=True)
    
    stats_path = os.path.join(results_dir, f'model_stats_{data_type}.txt')
    
    original_stdout = sys.stdout
    with open(stats_path, 'w') as f:
        sys.stdout = f
        
        print(f"Model Statistics for {model_name} - {data_type}")
        print("=" * 50)
        print("\nModel Configuration:")
        print(f"Number of classes: {args.num_classes}")
        print(f"Pre-training epochs: {args.pretrain_epochs}")
        print(f"Classification epochs: {args.epochs}")
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
        
        sys.stdout = original_stdout
    
    print(f"Model statistics saved to: {stats_path}")

def main():
    start_time = time.time()
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Pre-training for {args.pretrain_epochs} epochs")
    print(f"Classification training for {args.epochs} epochs with {args.num_classes} classes")
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
    model = SemiSupervisedChordExtractionCNN(num_classes=args.num_classes).to(device)

    # Pre-training phase (sequence ordering)
    print("Starting sequence ordering pre-training")
    sequence_criterion = nn.CrossEntropyLoss().to(device)
    sequence_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    pretrain_state, pretrain_losses, pretrain_val_losses = train_sequence_ordering(
        model, sequence_criterion, sequence_optimizer, train_dataloader, val_dataloader,
        epochs=args.pretrain_epochs, loss_hit_epochs=args.loss_hit_epochs,
        early_stop_epochs=args.early_stop_epochs, device=device
    )
    
    # Save pre-training results
    save_loss_graphs(pretrain_losses, pretrain_val_losses, args.model_name, args.data_type, args.pretrain_epochs, phase='sequence')

    # Classification phase
    print("Starting classification training")
    classification_criterion = nn.CrossEntropyLoss().to(device)
    classification_optimizer = optim.Adam(model.parameters(), lr=2.1e-5)
    
    # Load pre-trained weights
    model.load_state_dict(pretrain_state)
    
    classification_state, classification_losses, classification_val_losses = train_classification(
        model, classification_criterion, classification_optimizer, train_dataloader, val_dataloader,
        epochs=args.epochs, loss_hit_epochs=args.loss_hit_epochs,
        early_stop_epochs=args.early_stop_epochs, device=device
    )
    
    # Save classification results
    save_loss_graphs(classification_losses, classification_val_losses, args.model_name, args.data_type, args.epochs, phase='classification')

    # Evaluate final model
    print("Evaluating final model")
    accuracy, f1 = evaluate_model(model, test_dataloader, device)

    # Get model statistics
    inj = torch.randn(1, 9, 24, device=device)
    macs, params = profile(model, inputs=(inj,))
    flops = 2 * macs
    gflops = flops / 1e9

    # Save model statistics
    save_model_stats(model, macs, params, flops, gflops, accuracy, f1, args.model_name, args.data_type, args)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()