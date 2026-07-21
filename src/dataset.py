import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne 
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import torch
# necessary?
from torch.utils.data import Dataset
%matplotlib notebook

matplotlib.rcParams['figure.figsize'] = (5, 5)

# working with EEGBCI dataset from PhysioNet 
# 64-channel EEG
subjects = [1] # adjust if needed
runs = [4, 8, 12] # adjust if needed
raw_fnames = eegbci.load_data(subjects, runs)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# concatenate runs from subject
raw = concatenate_raws(raws)
# make channel names follow standard conventions
eegbci.standardize(raw)

# overview over dataset
print(raw)
print(raw.info)

# power spectral density (PSD) for each sensor type
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=30)

# preprocessing
class EEGDataset(Dataset):

    def __init__(self, files, labels, window_size=1000):
        self.files = files
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.files)
    
