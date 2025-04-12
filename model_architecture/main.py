import time
import argparse
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from torch.utils.data import DataLoader
from collections import Counter

from utils.custom_dataset import CustomDataset
from utils.common_utils import get_device, save_loss_graphs, save_model_stats, evaluate_model
from utils.training_utils import train, train_sequence_ordering
from architectures.carsault import ChordExtractionCNN
from architectures.small_dilation import SmallDilationModel
from architectures.semi_supervised import SemiSupervisedChordExtractionCNN
from architectures.multi_dilation import MultiDilationChordCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train ACE Model')
    parser.add_argument('--model_type', type=str, default='small_dilation',
                      choices=['small_dilation', 'carsault', 'semi_supervised', 'multi_dilation'],
                      help='Type of model to train (default: small_dilation)')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train (default: 1000)')
    parser.add_argument('--num_classes', type=int, default=28,
                      help='Number of classes (default: 28)')
    parser.add_argument('--model_name', type=str, default='ACE',
                      help='Name of the model folder in ModelResults (default: ACE)')
    parser.add_argument('--data_type', type=str, default='majmin',
                      help='Type of data to use (e.g., majmin, majmin7, majmininv, majmin7inv) (default: majmin)')
    parser.add_argument('--loss_hit_epochs', type=int, default=50,
                      help='Number of epochs without improvement before reducing learning rate (default: 50)')
    parser.add_argument('--early_stop_epochs', type=int, default=200,
                      help='Number of epochs without improvement before early stopping (default: 200)')
    parser.add_argument('--pretrain_epochs', type=int, default=1000,
                      help='Number of epochs for sequence ordering pre-training (default: 1000)')
    return parser.parse_args()

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

        # Check label distribution
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)

    # Ensure all classes have at least 2 samples
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("One or more classes have fewer than 2 samples.")

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
    
    if args.model_type == 'small_dilation':
        model = SmallDilationModel(num_classes=args.num_classes).to(device)
    elif args.model_type == 'carsault':
        model = ChordExtractionCNN(num_classes=args.num_classes).to(device)
    elif args.model_type == 'multi_dilation':
        model = MultiDilationChordCNN(num_classes=args.num_classes).to(device)
    elif args.model_type == 'semi_supervised':
        model = SemiSupervisedChordExtractionCNN(num_classes=args.num_classes).to(device)
    else:
        raise ValueError("""Model type does not exist. 
                         1. Please make sure it is implemented in the 'architectures' folder.
                         2. Please make sure it is implemented in the 'main.py' file imports and if statement.
                         3. Please make sure it is correctly indexed in 'run_models.py'.""")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2.1e-5)

    # Get MACs and parameters
    inj = torch.randn(1, 9, 24).to(device)
    macs, params = profile(model, inputs=(inj,))
    flops = 2 * macs
    gflops = flops / 1e9

    print(f"MACs: {macs:,}")
    print(f"FLOPs: {flops:,}")
    print(f"GFLOPs: {gflops:.4f}")
    print(f"Parameters: {params:.2f}")

    # Training
    if args.model_type == 'semi_supervised':
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

        # Load pre-trained weights
        model.load_state_dict(pretrain_state)
    
    # Classification training
    model_state, losses, val_losses = train(
        model, criterion, optimizer, train_dataloader, val_dataloader,
        epochs=args.epochs, loss_hit_epochs=args.loss_hit_epochs,
        early_stop_epochs=args.early_stop_epochs, device=device
    )
    
    # Save training results
    save_loss_graphs(losses, val_losses, args.model_name, args.data_type, args.epochs)

    # Evaluate model
    accuracy, f1 = evaluate_model(model, test_dataloader, device)
    
    # Save model statistics
    save_model_stats(model, macs, params, flops, gflops, accuracy, f1, args.model_name, args.data_type, args)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main() 