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
        # Loading the raw data by finding -raw.fif files for each subject's directory
        X_subjects = []
        y_subjects = []

        for subject_id in self.subject_ids:
            raw, events, _ = eegbci.load_data(subject_id, runs=self.runs)
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

            # Apply bandpass filter
            raw.filter(l_freq=self.filter_freqs[0], h_freq=self.filter_freqs[1])

            # Apply ICA to remove artifacts
            ica = ICA(num_components=20, random_state=97)
            ica.fit(raw)
            ica.apply(raw)

            # Epoching
            event_id = {"left": 2, "right": 3}
            epochs = mne.Epochs(raw, events=events, event_id=event_id, 
                                tmin=0, tmax=4, baseline=self.baseline, preload=self.preload)
            self.epochs_list.append(epochs)

            X_subjects.append(epochs.get_data)
            y_subjects.append(epochs.events[:, 2]) 
            # third column in MNE is event_id since second column is filled with 0 and first is the sample

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
    
    def __len__(self):
        return sum(len(epochs) for epochs in self.epochs_list)

    # Provide minimal info for DataLoader
    def __getitem__(self, idx):
        total_len = 0
        for i, epochs in enumerate(self.epochs_list):
            if idx >= total_len and idx < total_len + len(epochs):
                epoch_idx = idx - total_len
                self.data = [epochs.get_data() for epochs in self.epochs_list]
                data = self.data[i][epoch_idx]
                label = epochs.events[epoch_idx][2] - 1
                return data, label
            total_len += len(epochs)
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

