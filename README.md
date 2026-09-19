# Decoding EEG-Based Movement Imagery (MI) using a CNN-Transformer
_by Antonia Reul (2026)_ | Contact: areul@uni-osnabrueck.de

This project attempts to develop a CNN-Transformer, combining scientific techniques which have proven highly useful in research the past couple of years, which can decode movement imagination from EEG data reliably. The original idea of this project was to build on Ma et al. (2022) and Liao et al. (2025), but inspiration from other scientific papers and publications has been drawn as well. Please see the PDF **'decoding_eeg_movement_imagery.pdf'** for a **full report on architectural and parameter choices**. 

## Installation/Setup
The MNE library, as well as other medical libraries, are still getting upgraded and unfortunately sometimes still rely on old code. To ensure that the code works, 'requirements.txt' has been added specifying library versions which should be installed to ensure that the code runs.


## Repository Structure

```text
.
├── notebooks/              # Jupyter notebooks for experimentation, pipeline overview, and dataset descriptions.
├── results/
│   └── plots/              # Visualization of evaluation metrics after training runs.
├── src/
│   ├── datamodule.py/      # Defines the DataLoader and data handling pipeline.
│   ├── dataset.py/         # Initializes the dataset loader (critical for HPC workflow consistency).
│   ├── download_data.py/   # Script to download data from the MNE EEGBCI module.
│   ├── evaluate.py/        # Contains evaluation functions and performance metrics.
│   ├── model.py/           # Defines the neural network architecture.
│   ├── preprocess.py/      # Helper functions and reusable preprocessing utilities.
│   └── train.py/           # Main execution script for model training.
├── .DS_Store               # macOS system file (can be ignored).
├── .gitignore              # Defines which files (e.g., temporary data, virtual envs) should be ignored by Git.
├── README.md               # This file: The primary documentation entry point.
├── check_mne_path.py       # Utility script to verify MNE installation paths on the cluster.
├── job.sh                  # Main SLURM job script for submitting training jobs to the HPC.
├── preprocess.sh           # Shell script used for data preprocessing on the cluster's head node.
└── requirements.txt        # List of all Python libraries and their versions required.
```

#### References
* Liao, W., Liu, H. & Wang, W. (2025). Advancing BCI with a transformer-based model for motor imagery classification. Sci Rep 15, 23380. https://doi.org/10.1038/s41598-025-06364-4. - the code is available here: https://github.com/BlackCattt9/EEGEncoder
* Ma, Y., Song, Y. & Gao, F. (2022). A novel hybrid CNN-Transformer model for EEG Motor Imagery classification. International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, 1-8. https://doi.org/10.1109/IJCNN55064.2022.9892821.
* Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C28G6P - documentation: https://physionet.org/content/eegmmidb/1.0.0/
