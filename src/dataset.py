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

        # Loading the raw data by finding -raw.fif files for each subject's directory
        for subject_id in self.subject_ids:
            raw_fif, events, _ = eegbci.load_data(subject_id, runs=self.runs)
            raw_fif.set_montage(mne.channels.make_standard_montage('standard_1020'))

            # Apply bandpass filter
            raw_fif.filter(l_freq=self.filter_freqs[0], h_freq=self.filter_freqs[1])

            # Apply ICA to remove artifacts
            ica = ICA(num_components=20, random_state=97)
            ica.fit(raw_fif)
            ica.apply(raw_fif)

            # Epoching
            tmin, tmax = -1.0, 4.0
            baseline = self.baseline
            epochs = mne.Epochs(raw_fif, events=events, event_id=dict(left_hand=1, right_hand=2), 
                                tmin=tmin, tmax=tmax, baseline=baseline, preload=self.preload)
            self.epochs_list.append(epochs)

    def load_and_preprocess(self):
        # Load data for each subject and run
        all_X_train = []
        all_Y_train = []

        for subject in subjects:
            for run in runs:
                X_train, y_train = load_data(subject=subject, runs=[run])

                # Reshape data to fit MNE conventions
                n_samples = X_train.shape[0]
                n_channels = X_train.shape[1]
                X_train = np.reshape(X_train, (n_samples, n_channels, -1))
                all_X_train.append(X_train)
                all_y_train.append(y_train)

        # Concat all data and labels
        X_train_all = np.concatenate(all_X_train, axis=0)
        y_train_all = np.concatenate(all_y_train, axis=0)

        """more needs to be added here"""
    
    def __len__(self):
        return sum(len(epochs) for epochs in self.epochs_list)

    # Provide minimal info for DataLoader
    def __getitem__(self, idx):
        total_len = 0
        for i, epochs in enumerate(self.epochs_list):
            if idx >= total_len and idx < total_len + len(epochs):
                epoch_idx = idx - total_len
                data = epochs.get_data()[epoch_idx]
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

