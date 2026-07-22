import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import torch # necessary?
import mne 
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from torch.utils.data import Dataset
%matplotlib notebook

matplotlib.rcParams['figure.figsize'] = (5, 5)

# working with EEGBCI dataset from PhysioNet 
# 64-channel EEG
subjects = [1] # adjust if needed
runs = [4, 8, 12] # adjust if needed
raw_fnames = eegbci.load_data(subjects, runs)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raws.info['chs']

# concatenate runs from subject
raw = concatenate_raws(raws)
# set channel names
eegbci.standardize(raw)

# set montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage) # what is a montage?
raw.plot_sensors(kind='3d')

# overview over dataset
print(raw)
print(raw.info)

# power spectral density (PSD) for each sensor type
raw.compute_psd(fmax=80).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=30)

# Filtering
# remove 60 Hz power line noise using notch filter
raw.notch_filter(60)

# remove very high and very low frequencies
raw.filter(l_freq=1.0, h_freq=50.0) # only keeping 1-50Hz
raw.info['sfreq']

# Downsample data
raw.resample(120, npaud='auto')
raw.plot_psd(tmin=0, tmax=60, fmin=2, fmax=60, average=False, spatial_colors=True, xscale='log')

# look at raw data to look for bad channels
raw.plot(scalings = dict(eeg=200e-6))

# remove channels like C4 which look noisier
raw.info['bads'] = ['C4']
picks = mne.pick_types(raw.info, exclude='bads')
raw.plot(scalings=dict(eeg=200e-6), bad_color='red')

# interpolate data coming from bad channel via spherical spline interpolation
raw.interpolate_bads(reset_bads=True)
raw.plot(scalings=dict(eeg=200e-6))

# set up, fit ICA & localize the signal
# play around with num_components to get the ones that represent actual brain activity
num_components = 20
ica = ICA(n_components=num_components, random_state=97, max_iter=800)
ica.fit(raw)

# plot ICA on scalp
ica.plot_components()

# visualize each component's properties to reject artifactual comp
raw.plot(n_channels=32, scalings=dict(eeg=100e-6))

# heuristic: looking at spectrum of each component
ica.plot_properties(raw, picks=0) # exact comp num will prolly not work
ica.plot_properties(raw, picks=9) # outliers might have increasing intensity for higher freq

# look at data with bad component removed
ica.plot_overlay(raw, exclude=[0])
ica.exclude = [0]
ica.apply(raw)
raw.plot(scalings=dict(eeg=200e-6))



# include here detecting experimental events, epoching



# time-frequency analysis
frequencies = np.arange(7, 30, 3)
power = aud_epochs.compute_tfr(
    "morelt", n_cycles=2, return_itc=False, freqs=frequencies, decim=3, average=True
)
power.plot(["EEG001"]) # check what to input here

# inverse modeling - projecting data into subject's source space
# minimum-norm estimation (MNE)
inverse_operator_file = (?) # get eegbci .fif file inv
inv_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file)
# set signal-to-noise ratio (SNR) ro compute regularization params
snr = 3.0
lambda2 = 1.0 / snr**2
# generate source time course (STC)
stc = mne.minimum_norm.apply_inverse(?, inv_operator, lambda2=lambda2, method="MNE") # apply beamforming?




# rewrite everything into an EEG dataset class for importability?
class PreprocessedEEGDataset(Dataset):

    def __init__(self, files, labels, window_size=1000):
        self.files = files
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.files)
    
