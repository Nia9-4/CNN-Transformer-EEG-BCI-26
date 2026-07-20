import mne 
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):

    def __init__(self, files, labels, window_size=1000):
        self.files = files
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.files)
    
    