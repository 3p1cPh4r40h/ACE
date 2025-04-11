"""
This script processes chord and chroma data from the Billboard dataset.
It imports chord labels and chroma features, combines them into a single dataset,
and saves the processed data for later use.
"""

import numpy as np
import pandas as pd
import os
import pickle
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='logs/process_data.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def import_labels(path):
    """
    Import chord labels from a text file.
    
    Args:
        path (str): Path to the label file
        
    Returns:
        pd.DataFrame: DataFrame containing start_time, end_time, and chord labels
    """
    chords = []

    try:
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    chords.append([np.float32(parts[0]), np.float32(parts[1]), parts[2]])
                else:
                    logging.error(f"Skipping line due to unexpected format: {line}")
    except Exception as e:
        logging.error(f"Error processing file {path}: {e}")

    # Create DataFrame with parsed chord data
    time_labels_df = pd.DataFrame(chords, columns=['start_time', 'end_time', 'label'])
    
    # Validate time intervals
    if (time_labels_df['start_time'] > time_labels_df['end_time']).any():
        logging.error(time_labels_df['start_time'])
        logging.error(time_labels_df['end_time'])
    else:
        # Create an IntervalIndex for inclusive time intervals
        time_labels_df['interval'] = pd.IntervalIndex.from_arrays(
            time_labels_df['start_time'],
            time_labels_df['end_time'],
            closed='both'
        )

    return time_labels_df

def import_chromas(path):
    """
    Import chroma features from a CSV file.
    
    Args:
        path (str): Path to the chroma CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing time and chroma bin values
    """

    # Load chroma features from CSV, excluding the first column
    chroma = np.genfromtxt(path, delimiter=',', usecols=range(1,26))

    # Create DataFrame with time and chroma bin columns
    columns = ['time'] + [f'bin N{i}' for i in range(1, 25)]
    timing_df = pd.DataFrame(chroma, columns=columns)

    return timing_df

# Import chord labels from all files in the chords directory
print("Reading chord labels...")

print("Reading majmin.lab files...")
chords_directory = 'data/billboard-2.0.1-mirex/McGill-Billboard'
majmin_labels = []
# Process each folder in the chords directory
for folder in tqdm(os.listdir(chords_directory), desc='Processing chord labels'):
    folder_path = os.path.join(chords_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.basename(file_path) == 'majmin.lab':
                majmin_labels.append(import_labels(file_path))

print("Reading majmin7.lab files...")
majmin7_labels = []
# Process each folder in the chords directory
for folder in tqdm(os.listdir(chords_directory), desc='Processing chord labels'):
    folder_path = os.path.join(chords_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.basename(file_path) == 'majmin7.lab':
                majmin7_labels.append(import_labels(file_path))

print("Reading majmininv.lab files...")
majmininv_labels = []
# Process each folder in the chords directory
for folder in tqdm(os.listdir(chords_directory), desc='Processing chord labels'):
    folder_path = os.path.join(chords_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.basename(file_path) == 'majmininv.lab':
                majmininv_labels.append(import_labels(file_path))

print("Reading majmin7inv.lab files...")
majmin7inv_labels = []
# Process each folder in the chords directory
for folder in tqdm(os.listdir(chords_directory), desc='Processing chord labels'):
    folder_path = os.path.join(chords_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.basename(file_path) == 'majmin7inv.lab':
                majmin7inv_labels.append(import_labels(file_path))

# Import chroma features from all files in the chromas directory
print("Reading chroma features...")
chromas_directory = 'data/billboard-2.0-chordino/McGill-Billboard'
all_chromas = []

# Process each folder in the chromas directory
for folder in tqdm(os.listdir(chromas_directory), desc='Processing chroma features'):
    folder_path = os.path.join(chromas_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.basename(file_path) == 'bothchroma.csv':
                all_chromas.append(import_chromas(file_path))

def get_chord_for_time(time, time_labels_df):
    """
    Get the chord label for a specific time point.
    
    Args:
        time (float): Time point to look up
        time_labels_df (pd.DataFrame): DataFrame containing chord labels and intervals
        
    Returns:
        str: Chord label for the given time point, or None if no match found
    """
    # Handle cases where interval column is missing
    if 'interval' not in time_labels_df.columns:
        return None
    # Find matching interval and return corresponding chord label
    match = time_labels_df[time_labels_df['interval'].apply(lambda x: time in x)]
    return match['label'].values[0] if not match.empty else None

# Combine chroma features with chord labels for each dataset
print("Consolidating datasets...")

print("Consolidating majmin dataset...")
majmin_dataset = []
# Process each song in the dataset
for i in tqdm(range(len(all_chromas)), desc='Consolidating majmin dataset'):
    temp_df = all_chromas[i].copy()  # Create a deep copy of the DataFrame
    temp_labels_df = majmin_labels[i]
    # Map chord labels to each time point
    temp_df['chord'] = temp_df['time'].apply(lambda time: get_chord_for_time(time, temp_labels_df))
    majmin_dataset.append(temp_df)

print("Consolidating majmin7 dataset...")
majmin7_dataset = []
# Process each song in the dataset
for i in tqdm(range(len(all_chromas)), desc='Consolidating majmin7 dataset'):
    temp_df = all_chromas[i].copy()  # Create a deep copy of the DataFrame
    temp_labels_df = majmin7_labels[i]
    # Map chord labels to each time point
    temp_df['chord'] = temp_df['time'].apply(lambda time: get_chord_for_time(time, temp_labels_df))
    majmin7_dataset.append(temp_df)

print("Consolidating majmininv dataset...")
majmininv_dataset = []
# Process each song in the dataset
for i in tqdm(range(len(all_chromas)), desc='Consolidating majmininv dataset'):
    temp_df = all_chromas[i].copy()  # Create a deep copy of the DataFrame
    temp_labels_df = majmininv_labels[i]
    # Map chord labels to each time point
    temp_df['chord'] = temp_df['time'].apply(lambda time: get_chord_for_time(time, temp_labels_df))
    majmininv_dataset.append(temp_df)

print("Consolidating majmin7inv dataset...")
majmin7inv_dataset = []
# Process each song in the dataset
for i in tqdm(range(len(all_chromas)), desc='Consolidating majmin7inv dataset'):
    temp_df = all_chromas[i].copy()  # Create a deep copy of the DataFrame
    temp_labels_df = majmin7inv_labels[i]
    # Map chord labels to each time point
    temp_df['chord'] = temp_df['time'].apply(lambda time: get_chord_for_time(time, temp_labels_df))
    majmin7inv_dataset.append(temp_df)

# Save processed datasets to pickle files
print("Saving full datasets...")

print("Saving majmin dataset...")
with open('data/majmin/majmin_dataset.pkl', 'wb') as f:
    pickle.dump(majmin_dataset, f)

print("Saving majmin7 dataset...")
with open('data/majmin7/majmin7_dataset.pkl', 'wb') as f:
    pickle.dump(majmin7_dataset, f)

print("Saving majmininv dataset...")
with open('data/majmininv/majmininv_dataset.pkl', 'wb') as f:
    pickle.dump(majmininv_dataset, f)

print("Saving majmin7inv dataset...")
with open('data/majmin7inv/majmin7inv_dataset.pkl', 'wb') as f:
    pickle.dump(majmin7inv_dataset, f)

print("Done!")