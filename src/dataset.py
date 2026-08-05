import numpy as np
import matplotlib
import mne 
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from torch.utils.data import Dataset
import warnings

%matplotlib notebook

matplotlib.rcParams['figure.figsize'] = (5, 5)


"""A plotted analysis of the dataset can be found in a separate
Jupyter notebook in the folder 'notebooks'"""



class PreprocessedDataset(Dataset):
    def __init__(self, dataset_path, subject_ids=None, preload=True, filter_freqs=(1.0, 50.0), baseline=None):
        self.dataset_path = dataset_path
        self.subject_ids = subject_ids if subject_ids else range(1, 109) # all subjects
        self.preload = preload
        self.filter_freqs = filter_freqs
        self.baseline = baseline
        self.epochs_list = []

    def load_and_preprocess(self)
        # Empty lists to store the path for each subject
        X_subjects = []
        y_subjects = []

        for subject_id in self.subject_ids:
            # Loading the raw data file for each subject's directory
            raw, events, _ = eegbci.load_data(subject_id, runs=self.runs)
            # Overview of 10-20 montage: https://soft-dynamics.de/pdf/Int1020Syst.pdf 
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

            # Apply bandpass filter with values defined in init fct
            raw.filter(l_freq=self.filter_freqs[0], h_freq=self.filter_freqs[1])

            # Apply ICA to remove artifacts, if necessary
            ica = ICA(num_components=20, random_state=97)
            ica.fit(raw)
            ica.apply(raw)

            # Epoching to get specific time windows of continuous temporal data
            event_id = {"left": 2, "right": 3}
            epochs = mne.Epochs(raw, events=events, event_id=event_id, 
                                tmin=0, tmax=4, baseline=self.baseline, preload=self.preload)
            self.epochs_list.append(epochs)

            # raw EEG data for all epochs as NumPy array
            X_subjects.append(epochs.get_data) # (n_epochs, n_channels, n_samples)
            y_subjects.append(epochs.events[:, 2]) # 2nd column = event labels/IDs (our class we want to predict)
            # third column in MNE is event_id since second column is filled with 0 and first is the sample

        # concatenate all data from subjects for training
        X = np.concatenate(X_subjects, axis=0)
        y = np.concatenate(y_subjects, axis=0)

        # raw event IDs do not start at 0 which PyTorch classification losses expect
        label_map = {
            2: 0,
            3: 1
        }

        # T1 = 0 and T2 = 1
        y = np.array([label_map[label] for label in epochs.events[:, -1]])

        return X, y

    # knowing the number of elements in dataset
    def __len__(self): # useful for splitting dataset into training and validation set or calculating metrics
        return sum(len(epochs) for epochs in self.epochs_list)

    # allows to access elements of dataset by index like in standard array/list
    def __getitem__(self, idx):
        for epochs in self.epochs_list:
            if idx < len(epochs):
                data = epochs.get_data()[idx]
                label = epochs.events[epoch_idx][2] - 1
                return data, label
            idx -= len(epochs)
        raise IndexError(f'Index {idx} is out of bounds')

    # Fct for plotting PSDs quickly etc. (may be delayed later)
    def get_epoch(self, idx):
        """Retrieve specific epoch and its label"""
        total_len = 0
        for i, epochs in enumerate(self.epochs_list):
            if idx >= total_len and idx < total_len + len(epochs):
                epoch_idx = idx - total_len
                return self.epochs_list[i][epoch_idx]
            total_len += len(epochs)
        raise IndexError(f'Index {idx} is out of bounds')

