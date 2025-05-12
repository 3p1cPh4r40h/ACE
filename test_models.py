import os
import pickle
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from model_architecture.utils.custom_dataset import CustomDataset
from model_architecture.utils.common_utils import get_device
from model_architecture.architectures.carsault import ChordExtractionCNN
from model_architecture.architectures.small_dilation import SmallDilationModel
from model_architecture.architectures.semi_supervised import SemiSupervisedChordExtractionCNN
from model_architecture.architectures.multi_dilation import MultiDilationChordCNN
from model_architecture.architectures.late_squeeze import LateSqueezeChordCNN
from model_architecture.architectures.early_squeeze import EarlySqueezeChordCNN
from model_architecture.architectures.mid_squeeze import MidSqueezeChordCNN
from model_architecture.utils.training_utils import shuffle_sequence

# Available datasets and their number of classes
DATASETS = {
    "majmin": 28,
    "majmin7": 54, 
    "majmininv": 73, 
    "majmin7inv": 157
}

# Label sets for each dataset type
LABEL_SETS = {
    "majmin": ["N", "C:maj", "G:maj", "F:maj", "D:maj", "A:maj", "E:maj", "Bb:maj", "Eb:maj", "Ab:maj", 
               "A:min", "B:maj", "D:min", "E:min", "B:min", "Db:maj", "G:min", "C:min", "F:min", "Gb:maj", 
               "Eb:min", "Bb:min", "Ab:min", "Cb:maj", "Db:min", "Fb:maj", "Gb:min", "Cb:min"],
    "majmin7": ["N", "C:maj", "F:maj", "G:maj", "D:maj", "A:maj", "E:maj", "Bb:maj", "Ab:maj", "Eb:maj", 
                "A:min", "B:maj", "Db:maj", "D:min", "E:min", "B:min", "C:maj7", "G:maj7", "D:maj7", "F:maj7", 
                "A:maj7", "E:maj7", "Eb:maj7", "Ab:maj7", "D:min7", "E:min7", "A:min7", "Gb:maj", "G:min", 
                "Bb:maj7", "B:maj7", "G:min7", "C:min", "B:min7", "Eb:min7", "F:min", "C:min7", "Db:maj7", 
                "F:min7", "Bb:min7", "Eb:min", "Bb:min", "Gb:maj7", "Ab:min7", "Ab:min", "Cb:maj", "Db:min7", 
                "Cb:maj7", "Fb:maj", "Db:min", "Gb:min", "Cb:min", "Gb:min7", "Fb:maj7"],
    "majmininv": ["N", "C:maj", "G:maj", "F:maj", "D:maj", "A:maj", "E:maj", "Bb:maj", "Eb:maj", "Ab:maj", 
                  "A:min", "B:maj", "E:min", "D:min", "B:min", "Db:maj", "G:min", "C:min", "Gb:maj", "F:min", 
                  "Eb:min", "Bb:min", "D:maj/5", "Ab:min", "F:maj/3", "F:maj/5", "D:min/5", "C:maj/3", "Bb:maj/5", 
                  "D:maj/3", "E:maj/3", "C:maj/5", "G:maj/3", "Db:maj/5", "G:maj/5", "Cb:maj", "A:maj/5", "A:maj/3", 
                  "Db:min", "E:maj/5", "F:min/5", "Ab:maj/3", "Eb:maj/5", "Ab:maj/5", "D:min/b3", "Bb:maj/3", 
                  "B:maj/5", "Fb:maj", "E:min/b3", "Gb:maj/5", "Eb:maj/3", "E:min/5", "A:min/b3", "Db:maj/3", 
                  "Gb:min", "G:min/5", "B:min/5", "C:min/b3", "Eb:min/5", "B:min/b3", "B:maj/3", "A:min/5", 
                  "G:min/b3", "F:min/b3", "Ab:min/5", "C:min/5", "Gb:maj/3", "Eb:min/b3", "Cb:maj/5", "Ab:min/b3", 
                  "Db:min/5", "Bb:min/5", "Cb:min"],
    "majmin7inv": ["N", "C:maj", "G:maj", "F:maj", "D:maj", "A:maj", "E:maj", "Bb:maj", "Eb:maj", "Ab:maj", 
                   "A:min", "B:maj", "B:min", "E:min", "Db:maj", "D:min", "G:maj7", "C:maj7", "D:maj7", "F:maj7", 
                   "A:maj7", "E:maj7", "Eb:maj7", "Ab:maj7", "E:min7", "D:min7", "A:min7", "Gb:maj", "G:min", 
                   "G:min7", "B:maj7", "Bb:maj7", "C:min", "Eb:min7", "B:min7", "C:min7", "F:min", "Db:maj7", 
                   "F:min7", "Bb:min7", "Eb:min", "Bb:min", "Gb:maj7", "F:maj/5", "F:maj/3", "Ab:min7", "D:min/5", 
                   "C:maj/5", "Db:maj/5", "D:maj/3", "E:maj/3", "Bb:maj/5", "G:maj/3", "Ab:min", "G:maj/5", 
                   "C:maj/3", "A:maj/5", "A:maj/3", "E:maj/5", "Cb:maj", "Db:min7", "Ab:maj/3", "Ab:maj/5", 
                   "F:min7/5", "Eb:maj/5", "D:min/b3", "C:maj7/3", "Bb:maj/3", "Bb:maj7/7", "Cb:maj7", "Fb:maj", 
                   "F:min/5", "D:min7/5", "Bb:maj7/5", "E:min/b3", "B:maj/5", "Gb:maj/5", "Db:min", "Gb:min", 
                   "A:min/b3", "D:min7/b7", "D:maj7/3", "F:maj7/3", "Eb:maj/3", "E:min7/3", "G:maj7/3", "G:min/5", 
                   "B:min/5", "Db:maj/3", "E:maj7/b7", "B:min7/b7", "G:maj7/3", "F:maj7/5", "D:maj7/5", "Eb:maj7/5", 
                   "Db:maj7/3", "A:maj7/7", "A:maj7/5", "B:maj7/5", "C:maj7/b7", "A:maj7/3", "D:maj7/b7", "A:min7/b7", 
                   "G:min/b3", "E:min7/5", "D:min7/b3", "B:maj7/b7", "G:maj7/b7", "Eb:maj7/3", "B:min7/b3", "G:min7/5", 
                   "B:maj7/3", "A:maj7/b7", "Ab:maj7/b7", "C:min/5", "Gb:maj/3", "B:min/b3", "Ab:min/5", "Gb:maj7/5", 
                   "B:maj/3", "Db:maj7/7", "Eb:maj7/b7", "C:maj7/5", "Bb:maj7/3", "A:min7/5", "Eb:min/b3", "B:min7/5", 
                   "A:min/5", "F:min/b3", "Ab:maj7/5", "Ab:maj7/3", "Db:maj7/b7", "Cb:maj/5", "E:min7/b3", "A:min7/b3", 
                   "Ab:min/b3", "F:min7/b3", "C:min7/b3", "Eb:min7/b7", "Db:min/5", "E:maj7/5", "Ab:maj7/7", "Bb:min7/b7", 
                   "Bb:min/5", "F:maj7/b7", "Cb:min", "Gb:min7", "Fb:maj7", "F:maj7/7", "Db:maj7/5", "Ab:min7/5", 
                   "G:min7/b3", "G:min7/b7"]
}

# Available model types and whether they require pretraining
MODEL_TYPES = {
    "carsault": False,
    "semi_supervised": True,
    "small_dilation": False,
    "multi_dilation": False,
    "late_squeeze": False,
    "early_squeeze": False,
    "mid_squeeze": False
}

def decode_label(label_idx, data_type):
    """Convert a numeric label to its corresponding chord name."""
    # Convert label_idx to integer if it's a string
    if isinstance(label_idx, (str, np.str_)):
        try:
            label_idx = int(label_idx)
        except ValueError:
            return "Unknown"
    
    # Handle numpy types
    if isinstance(label_idx, (np.int32, np.int64)):
        label_idx = int(label_idx)
    
    if not isinstance(label_idx, int):
        return "Unknown"
    
    if label_idx < 0 or label_idx >= len(LABEL_SETS[data_type]):
        return "Unknown"
    return LABEL_SETS[data_type][label_idx]

def load_model(model_path, model_type, num_classes, device):
    """Load a trained model from disk"""
    if model_type == 'carsault':
        model = ChordExtractionCNN(num_classes=num_classes)
    elif model_type == 'small_dilation':
        model = SmallDilationModel(num_classes=num_classes)
    elif model_type == 'semi_supervised':
        model = SemiSupervisedChordExtractionCNN(num_classes=num_classes)
    elif model_type == 'multi_dilation':
        model = MultiDilationChordCNN(num_classes=num_classes)
    elif model_type == 'late_squeeze':
        model = LateSqueezeChordCNN(num_classes=num_classes)
    elif model_type == 'early_squeeze':
        model = EarlySqueezeChordCNN(num_classes=num_classes)
    elif model_type == 'mid_squeeze':
        model = MidSqueezeChordCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out profiling-related keys from state_dict
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                 if not any(x in k for x in ['total_ops', 'total_params'])}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_data(data_type):
    """Load test data for the specified data type."""
    data_path = os.path.join('data', data_type)
    data_file = os.path.join(data_path, f'{data_type}_data.pkl')
    labels_file = os.path.join(data_path, f'{data_type}_labels.pkl')

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    return np.array(data), np.array(labels)

def plot_spectrogram(spectrogram, title, save_path=None):
    """Plot a spectrogram and optionally save it."""
    plt.figure(figsize=(10, 4))
    # Transpose the spectrogram to swap X and Y axes
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_samples(model, data, labels, device, save_dir, model_type, data_type, is_pretrain=False):
    """Save input and output samples for a model."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Randomly select 10 samples
    num_samples = min(10, len(data))  # Ensure we don't try to sample more than available
    random_indices = np.random.choice(len(data), num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        input_data = torch.from_numpy(data[idx]).unsqueeze(0).to(device)
        true_label = labels[idx]
        
        with torch.no_grad():
            if is_pretrain:
                # For pretraining, we need to shuffle the input
                shuffled_input, original = shuffle_sequence(input_data, device)
                output = model(shuffled_input, task='sequence')
                # Plot original, shuffled, and reconstructed
                plot_spectrogram(original[0].cpu().numpy(), 
                               f'Original Spectrogram (Label: {true_label})',
                               os.path.join(save_dir, f'sample_{i+1}_original.png'))
                plot_spectrogram(shuffled_input[0].cpu().numpy(), 
                               'Shuffled Input',
                               os.path.join(save_dir, f'sample_{i+1}_shuffled.png'))
                plot_spectrogram(output[0].cpu().numpy(), 
                               'Reconstructed Output',
                               os.path.join(save_dir, f'sample_{i+1}_reconstructed.png'))
            else:
                # For final model, just show input and prediction
                output = model(input_data)
                predicted_label_idx = torch.argmax(output, dim=1).item()
                predicted_label = decode_label(predicted_label_idx, data_type)
                plot_spectrogram(input_data[0].cpu().numpy(), 
                               f'Input Spectrogram (True: {true_label}, Predicted: {predicted_label})',
                               os.path.join(save_dir, f'sample_{i+1}_input.png'))
                # For classification models, we can also show the probability distribution
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(probs)), probs)
                plt.title(f'Class Probabilities (True: {true_label}, Predicted: {predicted_label})')
                plt.xlabel('Class')
                plt.ylabel('Probability')
                plt.savefig(os.path.join(save_dir, f'sample_{i+1}_probabilities.png'))
                plt.close()

def check_model_exists(model_name, data_type, check_pretrain=False):
    """Check if model files exist."""
    model_path = os.path.join('ModelResults', model_name, data_type, 'model.pth')
    if not os.path.exists(model_path):
        return False
    
    if check_pretrain:
        pretrain_path = os.path.join('ModelResults', model_name, data_type, 'pretrained_model.pth')
        if not os.path.exists(pretrain_path):
            return False
    
    return True

def test_single_model(model_type, model_name, data_type, device):
    """Test a single model and dataset combination."""
    print(f"\n{'='*80}")
    print(f"Testing {model_type} model on {data_type} dataset")
    print(f"{'='*80}")
    
    try:
        # Check if model files exist
        if not check_model_exists(model_name, data_type):
            print(f"Model files not found for {model_type} on {data_type}")
            return False
        
        # Load test data
        data, labels = load_data(data_type)
        
        # Create save directory
        save_dir = os.path.join('ModelTestResults', model_name, data_type)
        os.makedirs(save_dir, exist_ok=True)
        
        # Test final model
        model_path = os.path.join('ModelResults', model_name, data_type, 'model.pth')
        print(f"\nTesting {model_name} model:")
        model = load_model(model_path, model_type, DATASETS[data_type], device)
        save_samples(model, data, labels, device, 
                    os.path.join(save_dir, 'final_model'), 
                    model_type, data_type, is_pretrain=False)
        
        # Test pretrained model if applicable
        if MODEL_TYPES[model_type]:  # If model requires pretraining
            pretrain_path = os.path.join('ModelResults', model_name, data_type, 'pretrained_model.pth')
            if os.path.exists(pretrain_path):
                print("\nTesting pretrained model:")
                pretrain_model = load_model(pretrain_path, model_type, DATASETS[data_type], device)
                save_samples(pretrain_model, data, labels, device, 
                           os.path.join(save_dir, 'pretrained_model'), 
                           model_type, data_type, is_pretrain=True)
            else:
                print(f"Pretrained model not found at {pretrain_path}")
        
        return True
    except Exception as e:
        print(f"Error testing {model_type} on {data_type}: {str(e)}")
        return False

def test_all_models():
    """Test all model and dataset combinations."""
    device = get_device()
    print(f"Using device: {device}")
    
    results = {}
    
    for model_type in MODEL_TYPES:
        results[model_type] = {}
        for data_type in DATASETS:
            print(f"\nTesting {model_type} model on {data_type} dataset...")
            if check_model_exists(model_type, data_type, MODEL_TYPES[model_type]):
                success = test_single_model(model_type, model_type, data_type, device)
                results[model_type][data_type] = "Success" if success else "Failed"
            else:
                print(f"Model files not found for {model_type} on {data_type}")
                results[model_type][data_type] = "Not Found"
    
    # Print summary
    print("\n\nSummary of Results:")
    print("-" * 80)
    print(f"{'Model Type':<20} {'Dataset':<15} {'Status':<10}")
    print("-" * 80)
    for model_type in results:
        for data_type in results[model_type]:
            print(f"{model_type:<20} {data_type:<15} {results[model_type][data_type]:<10}")
    print("-" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Test chord recognition models')
    parser.add_argument('--model_type', type=str, choices=['carsault', 'small_dilation', 'semi_supervised', 'multi_dilation', 'late_squeeze', 'early_squeeze', 'mid_squeeze'],
                        default='carsault', help='Type of model to test')
    parser.add_argument('--model_name', type=str, default=None,
                      help='Name of the model folder in ModelResults (default: same as model_type)')
    parser.add_argument('--data_type', type=str, default=None,
                      choices=list(DATASETS.keys()),
                      help='Type of data to use (default: test all datasets)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    if args.model_type and args.data_type:
        # Test single model and dataset
        model_name = args.model_name if args.model_name else args.model_type
        test_single_model(args.model_type, model_name, args.data_type, device)
    else:
        # Test all models and datasets
        test_all_models()

if __name__ == "__main__":
    main() 