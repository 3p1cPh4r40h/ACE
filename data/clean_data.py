# Import required libraries for data processing and manipulation
import numpy as np
import pandas as pd
import pickle
import re

# Helper function to flatten nested lists
def flatten(xss):
    return [x for xs in xss for x in xs]

def label_data_split(full_dataset, key):
    # Extract all unique chord labels from the dataset
    all_labels = []
    i=0
    for dataframe in full_dataset:
        temp_labels = dataframe['chord'].tolist()
        all_labels.append(temp_labels)
    all_labels = flatten(all_labels)
    all_labels = list(set(all_labels))
    print("All labels:")
    print(all_labels)
    print(len(all_labels))

    # Initialize lists to store processed data and labels
    data = []
    labels = []
    window = 9  # Size of the sliding window
    window_center = 9 // 2  # Center position of the window
    bin_cols = [f'bin N{i}' for i in range(1, 25)]  # Column names for bin data

    # Process the dataset using a sliding window approach
    tracker = 0
    for dataframe in full_dataset:
        tracker += 1
        if tracker%100 == 0:
            print(f'{tracker}/890')
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
    print(index)
    labels = [labels[i] for i in index]
    data = [data[i] for i in index]

    # Save the processed data and labels to pickle files for future use
    with open(f"data/{key}/{key}_data.pkl", 'wb') as f:
        pickle.dump(data, f)

    with open(f"data/{key}/{key}_labels.pkl", 'wb') as f:
        pickle.dump(labels, f)

path = "data/"
datasets = {
    "majmin" : None,
    "majmin7" : None, 
    "majmininv" : None, 
    "majmin7inv" : None
}

for key in datasets:
    # Open the datasets
    with open(f"data/{key}/{key}_dataset.pkl", 'rb') as f:
        datasets[key] = pickle.load(f)

    label_data_split(datasets[key], key)