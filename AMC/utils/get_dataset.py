import os
import torch
import numpy as np
import random
import h5py
import glob


def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def TrainDataset(seed=2023):
    """
    Load training and validation datasets from h5 files.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        x_train, y_train, x_val, y_val: Training and validation data.
    """
    setup_seed(seed)

    train_files = sorted(glob.glob(os.path.join('./dataset', 'train_snr_*.h5')))
    val_files = sorted(glob.glob(os.path.join('./dataset', 'val_snr_*.h5')))

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for fpath in train_files:
        with h5py.File(fpath, 'r') as f:
            x_train_list.append(f['X'][:])
            y_train_list.append(f['Y'][:])
    
    for fpath in val_files:
        with h5py.File(fpath, 'r') as f:
            x_val_list.append(f['X'][:])
            y_val_list.append(f['Y'][:])

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_val = np.concatenate(x_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)

    idx_train = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx_train], y_train[idx_train]

    idx_val = np.random.permutation(len(x_val))
    x_val, y_val = x_val[idx_val], y_val[idx_val]

    return x_train, y_train, x_val, y_val


def TestDataset(test_snr, seed=2023):
    """
    Load test dataset for a specific SNR value.

    Args:
        test_snr: SNR value for the test set.
        seed: Random seed for reproducibility.

    Returns:
        x_test, y_test: Test data and labels.
    """
    setup_seed(seed)

    if test_snr is None:
        raise ValueError("In test mode, `test_snr` must be specified.")

    test_file = os.path.join('./dataset', f'test_snr_{int(test_snr)}.h5')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    with h5py.File(test_file, 'r') as f:
        x = f['X'][:]
        x = x.transpose(0, 2, 1)
        y = f['Y'][:]

    return x, y
