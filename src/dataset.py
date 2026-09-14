import numpy as np
import mne 
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from torch.utils.data import Dataset


"""A plotted analysis of the dataset can be found in a separate
Jupyter notebook in the folder 'notebooks'"""

# Dataset documentation: https://www.physionet.org/content/eegmmidb/1.0.0/
# MNE: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html


class PreprocessedDataset(Dataset):
    """
    Initialize the dataset loader for the eegbci dataset
    
    Parameters:
    - subject_ids: List of subject IDs to load
    - runs: List of run numbers (e. g. [4] for left vs. right hand)
    - preload: Whether to load data into memory
    - filter_freqs: Tuple of low and high frequency for bandpass filtering
    - baseline: Baseline correction
    """
    
    def __init__(self, subject_ids=None, runs=None, preload=True, baseline=None):
        BAD = {88, 89, 92, 100}
        self.subject_ids = subject_ids if subject_ids else [s for s in range(1, 110) if s not in BAD] 
                            # all subjects with reliable data
        self.runs = runs if runs else [4, 8, 12] # runs for imagined left/right fist
        self.preload = preload
        self.baseline = baseline
        self.X = None 
        self.y = None

    def load_subject_data(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        # Loading the raw data file for a single subject
        paths = eegbci.load_data(subject_id, runs=self.runs, preload=self.preload, update_path=True)
        raw = concatenate_raws([read_raw_edf(p, preload=True) for p in paths])
        events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))

        eegbci.standardize(raw)    
        # 10-10 system used excluding Nz, F9/F10, ...
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'))

        # Apply notch and high-pass filter (2nd needed for ICA)
        raw.notch_filter(freqs=[60, 120])
        raw_for_ica = raw.copy().filter(l_freq=4.0, h_freq=None)
        # 4 Hz removes drift and blinks

        # Apply ICA to filtered copy 
        ica = ICA(n_components=0.99, random_state=42, method='fastica')
        ica.fit(raw_for_ica)

        # Find and apply components to original raw data
        # using frontal-polar electrodes closest to eyes 
        # -> most sensitive to EOG signals
        eog_ind = ica.find_bands_eog(raw, ch_name=['Fp1', 'Fp2'])
        ica.exclude = eog_ind 
        ica.apply(raw)

        # Define event ID
        event_id = {"left": 2, "right": 3}

        # Epoching into 4 s windows
        epochs = mne.Epochs(raw, events=events, event_id=event_id, 
                                tmin=0.5, tmax=3.5, baseline=self.baseline, 
                                preload=self.preload, reject=dict(eeg=150e-6), 
                                flat=dict(eeg=1e-7))

        # Extract data and labels
        X = epochs.get_data().astype(np.float32) * 1e6 # conversion to µV
         # (n_epochs, n_channels, n_samples)
        
        # Normalize data by z-score normalization 
        # Calculate mean and std across epoch and time dim per channel
        mu = np.mean(X, axis=(0, 2), keepdims=True)
        sd = np.std(X, axis=(0, 2), keepdims=True)
        X = (X - mu) / (sd + 1e-8)

        y = epochs.events[:, 2] # Event IDs from 3rd column (2 for left, 3 for right)

        # Raw event IDs do not start at 0 which PyTorch classification losses expect 
        # -> map to (0, 1) binary scale
        label_map = {2: 0, 3: 1}

        # Target vector y
        # T1 = 0 and T2 = 1
        y = np.array([label_map[label] for label in y])

        return X, y

    def __len__(self):
        # Return total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Access a single sample by index
        return self.X[idx], self.y[idx]

    def load_data(self) -> None:
        # Load and process data for all subjects
        self.X_subjects = []
        self.y_subjects = []

        for subject_id in self.subject_ids:
            X, y = self.load_subject_data(subject_id)
            self.X_subjects.append(X)
            self.y_subjects.append(y)
        
        # Concatenate all data from subjects 
        self.X = np.concatenate(self.X_subjects, axis=0)
        self.y = np.concatenate(self.y_subjects, axis=0)

        return self.X, self.y 
        # (batch, channels, samples)