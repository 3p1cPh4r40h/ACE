import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter

def load_data(dataset_name):
    """Load data and labels from the specified dataset."""
    data_path = os.path.join('data', dataset_name)
    data_file = os.path.join(data_path, f'{dataset_name}_data.pkl')
    labels_file = os.path.join(data_path, f'{dataset_name}_labels.pkl')

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Data files not found in {data_path}. Please ensure the data files exist.")

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    return data, labels

def plot_label_distribution(labels, dataset_name):
    """Plot the distribution of class labels and save the figure."""
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.title(f'Class Label Distribution for {dataset_name}')
    plt.xlabel('Class Labels')
    plt.ylabel('Counts')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot to the corresponding dataset folder
    results_dir = os.path.join('data', dataset_name)
    plot_path = os.path.join(results_dir, f'label_distribution_{dataset_name}.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    print(f"Label distribution plot saved to: {plot_path}")

def main():
    datasets = ['majmin', 'majmin7', 'majmininv', 'majmin7inv']  # List of datasets to explore

    for dataset in datasets:
        print(f"Loading data for dataset: {dataset}")
        _, labels = load_data(dataset)
        plot_label_distribution(labels, dataset)

if __name__ == "__main__":
    main()