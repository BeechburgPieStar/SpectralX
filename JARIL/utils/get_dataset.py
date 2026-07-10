import os
import torch
import numpy as np
import random
import scipy.io as sio
from sklearn.model_selection import train_test_split


def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def TrainDataset(data_path='data/train_data_split_amp.mat', train_ratio=0.9, seed=2023):
    """
    Load training and validation datasets for multi-task learning.
    
    Args:
        data_path: Path to the training data file
        train_ratio: Ratio of training data (rest is validation)
        seed: Random seed for reproducibility
    
    Returns:
        x_train, y_train_act, y_train_loc, x_val, y_val_act, y_val_loc
    """
    setup_seed(seed)
    
    # Load data
    data = sio.loadmat(data_path)
    x = data['train_data']
    y_act = data['train_activity_label'].squeeze()
    y_loc = data['train_location_label'].squeeze()
    
    # Convert to tensors
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y_act = torch.from_numpy(y_act).type(torch.LongTensor)
    y_loc = torch.from_numpy(y_loc).type(torch.LongTensor)
    
    # Split into train and validation
    indices = np.arange(len(x))
    train_idx, val_idx = train_test_split(indices, train_size=train_ratio, random_state=seed)
    
    x_train, x_val = x[train_idx], x[val_idx]
    y_train_act, y_val_act = y_act[train_idx], y_act[val_idx]
    y_train_loc, y_val_loc = y_loc[train_idx], y_loc[val_idx]
    
    return x_train, y_train_act, y_train_loc, x_val, y_val_act, y_val_loc


def TestDataset(data_path='data/test_data_split_amp.mat', seed=2023):
    """
    Load test dataset for multi-task learning.
    
    Args:
        data_path: Path to the test data file
        seed: Random seed for reproducibility
    
    Returns:
        x_test, y_test_act, y_test_loc
    """
    setup_seed(seed)
    
    # Load data
    data = sio.loadmat(data_path)
    x = data['test_data']
    y_act = data['test_activity_label'].squeeze()
    y_loc = data['test_location_label'].squeeze()
    
    # Convert to tensors
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y_act = torch.from_numpy(y_act).type(torch.LongTensor)
    y_loc = torch.from_numpy(y_loc).type(torch.LongTensor)
    
    return x, y_act, y_loc
