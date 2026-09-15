import numpy as np
import mne 
import os
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from typing import Tuple

try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.getcwd()

if os.path.basename(PROJECT_ROOT) == 'src':
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

print(f"Project Root identified as: {PROJECT_ROOT}")


"""A plotted analysis of the dataset can be found in a separate
Jupyter notebook in the folder 'notebooks'

Due to the MNE library still depending on some NumPy1. functions but newer
scipy commands, numpy=1.26.4 and scipy=1.12.0 are suggested for running 
this code successfully in an environment."""

# Dataset documentation: https://www.physionet.org/content/eegmmidb/1.0.0/
# MNE: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html


def load_subject_data(subject_id, runs=[4, 8, 12], preload=True, baseline=None) -> Tuple[np.ndarray, np.ndarray]:
    # Loading the raw data file for a single subject
    print('Checkpoint 1: Loading EDF files')
    paths = eegbci.load_data(subject_id, runs=runs, update_path=False)
    raw = concatenate_raws([read_raw_edf(p, preload=True) for p in paths])

    print('Checkpoint 2: Annotations and Montage')
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
    eegbci.standardize(raw)    
    # 10-10 system used excluding Nz, F9/F10, ...
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))

    # Apply notch and high-pass filter (2nd needed for ICA)
    print('Checkpoint 3: Filtering')
    raw.notch_filter(freqs=[60]) # Nyquist freq 80 Hz (160/2)
    raw_for_ica = raw.copy().filter(l_freq=4.0, h_freq=None)
    # 4 Hz removes drift and blinks

    # Apply ICA to filtered copy 
    print('Checkpoint 4: Starting ICA')
    ica = ICA(n_components=0.99, random_state=42, method='fastica')
    ica.fit(raw_for_ica)

    # Find and apply components to original raw data
    # using frontal-polar electrodes closest to eyes 
    # -> most sensitive to EOG signals
    eog_ind = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
    # Debugging case if MNE returns list in list
    if isinstance(eog_ind, list) and len(eog_ind) > 0 and isinstance(eog_ind[0], list):
        eog_ind = eog_ind[0]
    eog_ind = eog_ind[0]

    ica.apply(raw, exclude=set(eog_ind))

    # Define event ID
    print('Checkpoint 5: Epoching...')
    event_id = {"left": 2, "right": 3}

    # Epoching into 4 s windows
    epochs = mne.Epochs(raw, events=events, event_id=event_id, 
                                tmin=0.5, tmax=3.5, baseline=baseline, 
                                preload=preload, reject=dict(eeg=1000e-6), 
                                flat=dict(eeg=1e-7))

    # Extract data and labels
    print('Checkpoint 6: Finalizing data')
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
    # (batch, channels, samples)


def run_preprocessing():
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    x_path = os.path.join(processed_dir, 'eeg_X_processed.npy')
    y_path = os.path.join(processed_dir, 'eeg_y_processed.npy')

    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Processed data already exists. Skipping preprocessing to save time.")
        return
    
    print("Processed data not found. Starting full preprocessing pipeline.")
    os.makedirs(processed_dir, exist_ok=True)

    BAD = {88, 89, 92, 100}
    subject_ids = [s for s in range(1, 110) if s not in BAD] 

    X_all, y_all = [], []
    for s in subject_ids:
        print(f"Processing subject {s}...")
        try:
            X, y = load_subject_data(s)
            X_all.append(X)
            y_all.append(y)
        except Exception as e:
            print(f"Skipping subject: {s}: {e}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    np.save(os.path.join(processed_dir, 'eeg_X_processed.npy'), X)
    np.save(os.path.join(processed_dir, 'eeg_y_processed.npy'), y)

    print(f"Finished! 'eeg_X_processed.npy' and 'eeg_y_processed.npy' saved to {processed_dir}")

if __name__ == "__main__":
    run_preprocessing()