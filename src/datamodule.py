from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import PreprocessedDataset, load_data


def create_dataloaders(X, y):
    """
    Create data loaders to organize the data into small batches
    """

    BAD = {88, 89, 92, 100}
    all_subjects = list(s for s in range(1, 110) if s not in BAD)

    # 80 % of subjects used for training, 20 % for testing
    train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)

    # 70 % of subjects used for training, 10 % for validation during training
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.125, random_state=42)

    print(f"Subjects, train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(train_subjects)}")

    train_dataset = PreprocessedDataset(subject_ids=train_subjects)
    val_dataset = PreprocessedDataset(subject_ids=val_subjects)
    test_dataset = PreprocessedDataset(subject_ids=test_subjects)

    train_dataset.load_data()
    val_dataset.load_data()
    test_dataset.load_data()

    # Some other projects use batch size of 16 with the PhysioNet dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader