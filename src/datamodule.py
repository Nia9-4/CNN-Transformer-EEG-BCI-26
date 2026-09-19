import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import PreprocessedDataset

# Ensure project runs on HPCs and with ipynb test
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.getcwd()
    
if os.path.basename(PROJECT_ROOT) == 'src':
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

print(f"Project Root identified as: {PROJECT_ROOT}")


def create_dataloaders(batch_size):
    """
    Create data loaders to organize the data into small batches
    """

    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    try:
        X = np.load(os.path.join(processed_dir, 'eeg_X_processed.npy'))
        y = np.load(os.path.join(processed_dir, 'eeg_y_processed.npy'))
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed files not found in {processed_dir}. Please run preprocess.py first!")

    full_dataset = PreprocessedDataset(X, y)
    indices = np.arange(len(X))

    # 80 % of subjects used for training, 20 % for testing
    train_val_subjects, test_subjects = train_test_split(indices, test_size=0.2, random_state=42)

    # 70 % of subjects used for training, 10 % for validation during training
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.125, random_state=42)

    print(f"Subjects, train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")

    train_loader = DataLoader(Subset(full_dataset, train_subjects), batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(Subset(full_dataset, val_subjects), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_subjects), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
