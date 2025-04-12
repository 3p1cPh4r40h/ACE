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
from multiprocessing import Pool, cpu_count
from functools import partial
import gc

# Set up logging
logging.basicConfig(
    filename='logs/process_data.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def import_labels(path):
    """
    Import chord labels from a text file using numpy for faster parsing.
    
    Args:
        path (str): Path to the label file
        
    Returns:
        pd.DataFrame: DataFrame containing start_time, end_time, and chord labels
    """
    try:
        # Use numpy's loadtxt for faster parsing
        data = np.loadtxt(path, dtype={'names': ('start_time', 'end_time', 'label'),
                                      'formats': ('f4', 'f4', 'U10')})
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate time intervals
        if (df['start_time'] > df['end_time']).any():
            logging.error(f"Invalid time intervals in {path}")
            return None
            
        return df
        
    except Exception as e:
        logging.error(f"Error processing file {path}: {e}")
        return None

def import_chromas(path):
    """
    Import chroma features from a CSV file using numpy for faster loading.
    
    Args:
        path (str): Path to the chroma CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing time and chroma bin values
    """
    try:
        # Load chroma features using numpy for better performance
        chroma = np.loadtxt(path, delimiter=',', usecols=range(1,26), dtype=np.float32)
        
        # Create DataFrame with optimized column names
        columns = ['time'] + [f'bin_{i}' for i in range(1, 25)]
        return pd.DataFrame(chroma, columns=columns)
        
    except Exception as e:
        logging.error(f"Error loading chroma file {path}: {e}")
        return None

def get_chord_for_time(time, time_labels_df):
    """
    Get the chord label for a specific time point using vectorized operations.
    
    Args:
        time (float): Time point to look up
        time_labels_df (pd.DataFrame): DataFrame containing chord labels and intervals
        
    Returns:
        str: Chord label for the given time point, or None if no match found
    """
    if time_labels_df is None or time_labels_df.empty:
        return None
    
    # Use vectorized operations for faster lookup
    mask = (time_labels_df['start_time'] <= time) & (time_labels_df['end_time'] >= time)
    matches = time_labels_df[mask]
    return matches['label'].values[0] if not matches.empty else None

def process_song(args, label_type):
    """
    Process a single song's data with its corresponding labels.
    
    Args:
        args (tuple): Tuple containing (chroma_df, labels_df)
        label_type (str): Type of labels being processed
        
    Returns:
        pd.DataFrame: Processed DataFrame with chord labels
    """
    chroma_df, labels_df = args
    if chroma_df is None or labels_df is None:
        return None
        
    try:
        # Create a copy of the DataFrame
        temp_df = chroma_df.copy()
        
        # Vectorize the chord lookup
        temp_df['chord'] = temp_df['time'].apply(
            lambda time: get_chord_for_time(time, labels_df)
        )
        
        # Add label type for reference
        temp_df['label_type'] = label_type
        
        return temp_df
        
    except Exception as e:
        logging.error(f"Error processing song: {e}")
        return None

def process_dataset(chromas, chroma_paths, labels, label_paths, dataset_name, num_processes):
    """
    Process a complete dataset using parallel processing.
    
    Args:
        chromas (list): List of chroma DataFrames
        chroma_paths (list): List of paths to chroma files
        labels (list): List of label DataFrames
        label_paths (dict): Dictionary containing paths to label files for each label type
        dataset_name (str): Name of the dataset
        num_processes (int): Number of processes to use
        
    Returns:
        list: Processed dataset
    """
    print(f"Consolidating {dataset_name} dataset...")

    # Create a mapping of folder names to their corresponding data
    chroma_map = {os.path.basename(os.path.dirname(chroma_path)): chroma_df 
                 for chroma_path, chroma_df in zip(chroma_paths, chromas)}
    label_map = {os.path.basename(os.path.dirname(label_path)): label_df 
                for label_path, label_df in zip(label_paths, labels)}
    
    # Find the intersection of folders that have both chroma and label data
    common_folders = set(chroma_map.keys()) & set(label_map.keys())
    
    if len(common_folders) < len(chromas) or len(common_folders) < len(labels):
        logging.warning(f"Found {len(common_folders)} matching folders out of {len(chromas)} chromas and {len(labels)} labels")
    
    # Prepare arguments for parallel processing using only matching folders
    process_args = [(chroma_map[folder], label_map[folder]) for folder in common_folders]
    
    # Create partial function with label_type
    process_func = partial(process_song, label_type=dataset_name)
    
    # Process songs in parallel
    with Pool(processes=num_processes) as pool:
        dataset = list(tqdm(
            pool.imap(process_func, process_args),
            total=len(process_args),
            desc=f'Processing {dataset_name} songs'
        ))
    
    # Filter out None results
    dataset = [df for df in dataset if df is not None]
    
    return dataset

def main():
    # Import chord labels
    print("Reading chord labels...")
    chords_directory = 'data/billboard-2.0.1-mirex/McGill-Billboard'
    
    # Dictionary to store all label types
    label_types = {
        'majmin': 'majmin.lab',
        'majmin7': 'majmin7.lab',
        'majmininv': 'majmininv.lab',
        'majmin7inv': 'majmin7inv.lab'
    }
    
    # Track which folders we process for labels
    processed_label_folders = set()
    
    # Process all label types
    labels = {}
    label_paths = {}  # New dictionary to store paths
    for label_type, filename in label_types.items():
        print(f"Reading {filename} files...")
        label_list = []
        label_path_list = []  # New list to store paths
        for folder in tqdm(os.listdir(chords_directory), desc=f'Processing {label_type} labels'):
            folder_path = os.path.join(chords_directory, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    label_df = import_labels(file_path)
                    if label_df is not None:
                        label_list.append(label_df)
                        label_path_list.append(file_path)  # Store the path
                        processed_label_folders.add(folder)
                    else:
                        logging.error(f"Failed to load label file: {file_path}")
                else:
                    logging.error(f"Label file not found: {file_path}")
        labels[label_type] = label_list
        label_paths[label_type] = label_path_list  # Store paths for this label type
        logging.info(f"Loaded {len(label_list)} {label_type} label files")
    
    # Import chroma features
    print("Reading chroma features...")
    chromas_directory = 'data/billboard-2.0-chordino/McGill-Billboard'
    all_chromas = []
    chroma_paths = []  # New list to store paths
    
    # Track which folders we process for chromas
    processed_chroma_folders = set()
    
    for folder in tqdm(os.listdir(chromas_directory), desc='Processing chroma features'):
        folder_path = os.path.join(chromas_directory, folder)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'bothchroma.csv')
            if os.path.exists(file_path):
                chroma_df = import_chromas(file_path)
                if chroma_df is not None:
                    all_chromas.append(chroma_df)
                    chroma_paths.append(file_path)  # Store the path
                    processed_chroma_folders.add(folder)
                else:
                    logging.error(f"Failed to load chroma file: {file_path}")
            else:
                logging.error(f"Chroma file not found: {file_path}")
    
    logging.info(f"Loaded {len(all_chromas)} chroma files")
    
    # Check for mismatches in processed folders
    missing_in_labels = processed_chroma_folders - processed_label_folders
    missing_in_chromas = processed_label_folders - processed_chroma_folders
    
    if missing_in_labels:
        logging.error(f"Folders with chromas but missing labels: {missing_in_labels}")
    if missing_in_chromas:
        logging.error(f"Folders with labels but missing chromas: {missing_in_chromas}")
    
    # Determine number of processes to use
    num_processes = max(1, cpu_count() - 1)
    
    # Process and save all datasets
    for dataset_name, label_list in labels.items():
        if len(all_chromas) != len(label_list):
            logging.error(f"Dataset {dataset_name} has mismatched counts: {len(all_chromas)} chromas vs {len(label_list)} labels")
            
        # Process dataset with paths
        dataset = process_dataset(all_chromas, chroma_paths, label_list, label_paths[dataset_name], dataset_name, num_processes)
        
        if dataset:  # Only save if we have valid data
            # Save the dataset
            print(f"Saving {dataset_name} dataset...")
            os.makedirs(f'data/{dataset_name}', exist_ok=True)
            with open(f'data/{dataset_name}/{dataset_name}_dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
            
            # Clear memory
            del dataset
            gc.collect()
        else:
            logging.error(f"Skipping save for {dataset_name} due to no valid data")
    
    print("Done!")

if __name__ == "__main__":
    main()