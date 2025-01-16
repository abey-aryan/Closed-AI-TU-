import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def normalize_data(X_train, X_test, numerical_columns):
    """
    Normalize numerical columns in the training and testing data using MinMaxScaler.

    Args:
        X_train (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Testing dataset.
        numerical_columns (list): List of columns to normalize.

    Returns:
        pd.DataFrame: Normalized training dataset.
        pd.DataFrame: Normalized testing dataset.
        MinMaxScaler: Scaler used for normalization.
    """
    scaler = MinMaxScaler()
    
    # Normalize the training data
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    
    # Normalize the testing data based on the training data scaling
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    return X_train, X_test, scaler


def save_scaler(scaler, file_path):
    """
    Save the scaler for later use during inference.

    Args:
        scaler (MinMaxScaler): Fitted scaler.
        file_path (str): Path to save the scaler file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(file_path):
    """
    Load the scaler for inference.

    Args:
        file_path (str): Path to the saved scaler file.

    Returns:
        MinMaxScaler: Loaded scaler.
    """
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def create_sequences(data, seq_length, target_col):
    """
    Create sequences of time-series data for training.

    Args:
        data (pd.DataFrame): Dataset with normalized values.
        seq_length (int): Sequence length (e.g., 12 months).
        target_col (str): Column name of the target variable.

    Returns:
        np.array: Sequences of features.
        np.array: Corresponding labels.
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequence = data.iloc[i:i+seq_length].drop(columns=[target_col]).values
        target = data.iloc[i+seq_length][target_col]
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)