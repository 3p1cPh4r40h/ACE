import torch
import os
import matplotlib.pyplot as plt
import sys
from torchinfo import summary
from sklearn.metrics import accuracy_score, f1_score

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    return device

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

def save_model_stats(model, macs, params, flops, gflops, accuracy, f1, model_name, data_type, args, num_classes):
    results_dir = os.path.join('ModelResults', model_name, data_type)
    os.makedirs(results_dir, exist_ok=True)
    
    stats_path = os.path.join(results_dir, f'model_stats_{data_type}.txt')
    
    original_stdout = sys.stdout
    with open(stats_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print(f"Model Statistics for {model_name} - {data_type}")
        print("=" * 50)
        print("\nModel Configuration:")
        print(f"Number of classes: {num_classes}")
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
        
        sys.stdout = original_stdout
    
    print(f"Model statistics saved to: {stats_path}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1 