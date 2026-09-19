import mne
import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Initialize the dataset loader for the eegbci dataset
    
    Parameters:
    - subject_ids: List of subject IDs to load
    - runs: List of run numbers (e. g. [4] for left vs. right hand)
    - preload: Whether to load data into memory
    - baseline: Baseline correction
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # Return total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Access a single sample by index
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype = torch.float32)
    
