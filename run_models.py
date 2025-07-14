import time
import argparse
import os
import pickle
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from torch.utils.data import DataLoader
from collections import Counter

from model_architecture.utils.custom_dataset import CustomDataset
from model_architecture.utils.common_utils import get_device, save_loss_graphs, save_model_stats, evaluate_model
from model_architecture.utils.training_utils import train, train_sequence_ordering
from model_architecture.architectures.carsault import ChordExtractionCNN
from model_architecture.architectures.semi_supervised import SemiSupervisedChordExtractionCNN
from model_architecture.architectures.multi_dilation import MultiDilationChordCNN
from model_architecture.architectures.lstm_multi_dilation import LSTMMultiDilationChordCNN
from model_architecture.architectures.lstm_multi_dilation_A import LSTMMultiDilationChordCNNA
from model_architecture.architectures.lstm_multi_dilation_B import LSTMMultiDilationChordCNNB
from model_architecture.architectures.lstm_multi_dilation_C import LSTMMultiDilationChordCNNC
from model_architecture.architectures.lstm_multi_dilation_D import LSTMMultiDilationChordCNND
from model_architecture.architectures.lstm_multi_dilation_E import LSTMMultiDilationChordCNNE

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Available model types and whether they require pretraining
MODEL_TYPES = {
    "carsault": False,
    "semi_supervised": True,
    "multi_dilation": False,
    "lstm_multi_dilation": False,
    "lstm_multi_dilation_A": False,
    "lstm_multi_dilation_B": False,
    "lstm_multi_dilation_C": False,
    "lstm_multi_dilation_D": False,
    "lstm_multi_dilation_E": False
}

# Available datasets and their number of classes
DATASETS = {
    "majmin": 28,
    "majmin7": 54, 
    "majmininv": 73, 
    "majmin7inv": 157 # Requires further processing, currently fails in training
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train ACE Model')
    parser.add_argument('--model_type', type=str, nargs='+', default=['carsault'],
                      choices=['carsault', 'semi_supervised', 'multi_dilation', 'lstm_multi_dilation', 
                              'lstm_multi_dilation_A', 'lstm_multi_dilation_B', 'lstm_multi_dilation_C', 
                              'lstm_multi_dilation_D', 'lstm_multi_dilation_E'],
                      help='Type(s) of model(s) to train (default: carsault). Can specify multiple models.')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train (default: 1000)')
    parser.add_argument('--model_name', type=str, default='carsault',
                      help='Name of the model folder in ModelResults (default: carsault)')
    parser.add_argument('--data_type', type=str, default='majmin',
                      choices=list(DATASETS.keys()),
                      help='Type of data to use (e.g., majmin, majmin7, majmininv, majmin7inv) (default: majmin)')
    parser.add_argument('--loss_hit_epochs', type=int, default=50,
                      help='Number of epochs without improvement before reducing learning rate (default: 50)')
    parser.add_argument('--early_stop_epochs', type=int, default=200,
                      help='Number of epochs without improvement before early stopping (default: 200)')
    parser.add_argument('--pretrain_epochs', type=int, default=1000,
                      help='Number of epochs for sequence ordering pre-training (default: 1000)')
    parser.add_argument('--batch_mode', action='store_true',
                      help='Run all models sequentially on all datasets, note that this will override the model_type, but a data_type can be provided')
    return parser.parse_args()

def train_single_model(args):
    """Train a single model with the given arguments."""
    start_time = time.time()
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training for {args.epochs} epochs")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_type}")
    print(f"Number of classes: {DATASETS[args.data_type]}")
    print(f"Loss hit epochs: {args.loss_hit_epochs}")
    print(f"Early stop epochs: {args.early_stop_epochs}")

    # Load the data and labels            
    print("Setting up data")
    
    data_path = os.path.join('data', args.data_type)
    data_file = os.path.join(data_path, f'{args.data_type}_data.pkl')
    labels_file = os.path.join(data_path, f'{args.data_type}_labels.pkl')

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        print(f"Error: Data files not found in {data_path}. Please ensure the data files exist.")
        return False

    try:
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
            print("Error: One or more classes have fewer than 2 samples.")
            return False

        # Split into 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, stratify=labels, random_state=2025)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=2025)

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
        
        if args.model_type == 'carsault':
            model = ChordExtractionCNN(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'semi_supervised':
            model = SemiSupervisedChordExtractionCNN(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'multi_dilation':
            model = MultiDilationChordCNN(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation':
            model = LSTMMultiDilationChordCNN(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation_A':
            model = LSTMMultiDilationChordCNNA(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation_B':
            model = LSTMMultiDilationChordCNNB(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation_C':
            model = LSTMMultiDilationChordCNNC(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation_D':
            model = LSTMMultiDilationChordCNND(num_classes=DATASETS[args.data_type]).to(device)
        elif args.model_type == 'lstm_multi_dilation_E':
            model = LSTMMultiDilationChordCNNE(num_classes=DATASETS[args.data_type]).to(device)
        else:
            print("Error: Model type does not exist.")
            return False

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
            sequence_criterion = nn.MSELoss().to(device)
            sequence_optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            pretrain_state, pretrain_losses, pretrain_val_losses = train_sequence_ordering(
                model, sequence_criterion, sequence_optimizer, train_dataloader, val_dataloader,
                epochs=args.pretrain_epochs, loss_hit_epochs=args.loss_hit_epochs,
                early_stop_epochs=args.early_stop_epochs, device=device
            )
            
            # Save pre-training results
            save_loss_graphs(pretrain_losses, pretrain_val_losses, args.model_name, args.data_type, args.pretrain_epochs, phase='sequence')

            # Save the pre-trained model
            pretrain_model_save_path = os.path.join('ModelResults', args.model_name, args.data_type, 'pretrained_model.pth')
            os.makedirs(os.path.dirname(pretrain_model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': pretrain_state,
                'model_type': args.model_type,
                'num_classes': DATASETS[args.data_type],
                'data_type': args.data_type,
                'phase': 'pretrain'
            }, pretrain_model_save_path)
            print(f"Pre-trained model saved to {pretrain_model_save_path}")

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
        save_model_stats(model, macs, params, flops, gflops, accuracy, f1, args.model_name, args.data_type, args, num_classes=DATASETS[args.data_type])
        
        # Save the trained model
        model_save_path = os.path.join('ModelResults', args.model_name, args.data_type, 'model.pth')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model_state,
            'model_type': args.model_type,
            'num_classes': DATASETS[args.data_type],
            'data_type': args.data_type,
            'accuracy': accuracy,
            'f1': f1
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return True
    except Exception as e:
        print(f"Error training {args.model_type} on {args.data_type}: {str(e)}")
        return False

def run_batch_models(epochs=1000, data_type=None):
    """Run all models sequentially on the specified dataset.
    If data_type is provided, only run on that dataset.
    If data_type is not provided, run on all datasets."""
    # Get list of models to run - always run all models in batch mode
    models_to_run = list(MODEL_TYPES.keys())
    
    # Get list of datasets to run
    if data_type:
        if data_type not in DATASETS:
            print(f"Error: Dataset '{data_type}' not found in available datasets.")
            return
        datasets_to_run = [data_type]
    else:
        datasets_to_run = list(DATASETS.keys())

    print(f"Running {len(models_to_run)} models on {len(datasets_to_run)} datasets")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    
    results = {}
    for model_type in models_to_run:
        results[model_type] = {}
        for dataset in datasets_to_run:
            print(f"\nRunning {model_type} model on {dataset} dataset...")
            args = argparse.Namespace(
                model_type=model_type,
                epochs=epochs,
                num_classes=DATASETS[dataset],
                model_name=model_type,
                data_type=dataset,
                loss_hit_epochs=50,
                early_stop_epochs=200,
                pretrain_epochs=epochs if MODEL_TYPES[model_type] else 0
            )
            
            success = train_single_model(args)
            results[model_type][dataset] = "Success" if success else "Failed"
            time.sleep(2)  # Small pause between runs
    
    # Print summary
    print("\n\nSummary of Results:")
    print("-" * 80)
    print(f"{'Model Type':<30} {'Dataset':<15} {'Status':<10}")
    print("-" * 80)
    for model_type in results:
        for dataset in results[model_type]:
            print(f"{model_type:<30} {dataset:<15} {results[model_type][dataset]:<10}")
    print("-" * 80)

def main():
    args = parse_args()
    
    if args.batch_mode:
        # You run all models on a single dataset or all datasets in batch mode
        run_batch_models(args.epochs, args.data_type)
    else:
        # Run each specified model
        for model_type in args.model_type:
            print(f"\nTraining model: {model_type}")
            model_args = argparse.Namespace(
                model_type=model_type,
                epochs=args.epochs,
                model_name=model_type,  # Use model_type as the model_name
                data_type=args.data_type,
                loss_hit_epochs=args.loss_hit_epochs,
                early_stop_epochs=args.early_stop_epochs,
                pretrain_epochs=args.pretrain_epochs
            )
            train_single_model(model_args)

if __name__ == "__main__":
    main()