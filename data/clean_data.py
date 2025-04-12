# Import required libraries for data processing and manipulation
import numpy as np
import pandas as pd
import pickle
import re
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename='logs/clean_data.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Helper function to flatten nested lists
def flatten(xss):
    return [x for xs in xss for x in xs]

# Helper function to ensure chord labels have :maj or :min suffix
def ensure_chord_suffix(label):
    # Match the key and quality (major/minor/seventh)
    if isinstance(label, str):
        # Match the key and quality (major/minor/seventh)
        match = re.match(r"([A-G][b#]?)(?::(maj|min))?(.*)?", label)
        if match:
            key, quality, mod = match.groups()
            # Remove any existing colons from mod
            mod = mod.replace(':', '') if mod else ''
            # Default to 'maj' if no quality is specified
            return f"{key}:{quality or 'maj'}{mod}"
    return "N"

def label_data_split(full_dataset, key):
    try:
        # Extract all unique chord labels from the dataset
        print(f"{key} labels")
        all_labels = []
        i=0
        for dataframe in full_dataset:
            dataframe['chord'] = dataframe['chord'].replace([None, "N"], "X")
            # Apply the chord suffix modification
            dataframe['chord'] = dataframe['chord'].apply(ensure_chord_suffix)
            temp_labels = dataframe['chord'].tolist()
            all_labels.append(temp_labels)
        all_labels = flatten(all_labels)
        all_labels = list(set(all_labels))
        print(all_labels)
        print(len(all_labels))

        # Initialize lists to store processed data and labels
        data = []
        labels = []
        window = 9  # Size of the sliding window
        window_center = 9 // 2  # Center position of the window
        bin_cols = [f'bin_{i}' for i in range(1, 25)]  # Column names for bin data

        # Process the dataset using a sliding window approach
        tracker = 0
        for dataframe in tqdm(full_dataset, desc='Processing DataFrames', unit='df'):
            tracker += 1
            length = len(dataframe)
            # Create chunks of data using the sliding window
            for i in range(0, length - length%window, window):
                chunk = dataframe.iloc[i:i+window]
                data.append(chunk[bin_cols])
                labels.append(chunk.iloc[window_center]['chord'])

        # Convert pandas DataFrames to numpy arrays for better processing
        for i in range(len(data)):
            data[i] = data[i].to_numpy()

        # Remove entries with null values from both data and labels
        temp = 0
        index = []
        for i in range(len(data)):
            if labels[i] != None:
                index.append(i)
        labels = [labels[i] for i in index]
        data = [data[i] for i in index]

        # Create directory if it doesn't exist
        os.makedirs(f"data/{key}", exist_ok=True)

        # Save the processed data and labels to pickle files for future use
        with open(f"data/{key}/{key}_data.pkl", 'wb') as f:
            pickle.dump(data, f)

        with open(f"data/{key}/{key}_labels.pkl", 'wb') as f:
            pickle.dump(labels, f)

    except Exception as e:
        logging.error(f"Error processing {key} dataset: {e}")
        raise

path = "data/"
datasets = {
    "majmin" : None, # 28 labels
    "majmin7" : None, # 54 labels 
    "majmininv" : None, # 73 labels
    "majmin7inv" : None #157 labels
}

for key in datasets:
    try:
        # Open the datasets
        with open(f"data/{key}/{key}_dataset.pkl", 'rb') as f:
            datasets[key] = pickle.load(f)

        label_data_split(datasets[key], key)
    except Exception as e:
        logging.error(f"Error loading or processing {key} dataset: {e}")
        continue