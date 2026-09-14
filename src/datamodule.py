import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import PreprocessedDataset


def create_dataloaders(batch_size):
    """
    Create data loaders to organize the data into small batches
    """
    X = np.load('eeg_X_preprocessed.npy')
    y = np.load('eeg_y_preprocessed.npy')

    full_dataset = PreprocessedDataset(X, y)
    BAD = {88, 89, 92, 100}
    all_subjects = list(s for s in range(len(X)) if s not in BAD)

    # 80 % of subjects used for training, 20 % for testing
    train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)

    # 70 % of subjects used for training, 10 % for validation during training
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.125, random_state=42)

    print(f"Subjects, train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")

    # Some other projects use batch size of 16 with the PhysioNet dataset
    train_loader = DataLoader(Subset(full_dataset, train_subjects), batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(Subset(full_dataset, val_subjects), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_subjects), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader