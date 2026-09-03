from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import PreprocessedDataset, load_and_preprocess


"""Data loaders organize data into batches (small groups) 
and mixes up the order of items which helps the models learn better"""


def create_dataloaders(X, y):
    subjects = list(range(1, 110)) # 109 participants

    # 80 % of subjects used for training, 20 % for testing
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)

    X_train = []
    y_train = []

    for subject in train_subjects:
        X_subj, y_subj = load_and_preprocess(subject)
        X_train.append(X_subj)
        y_train.append(y_subj)

    X_test = []
    y_test = []

    for subject in test_subjects:
        X_subj, y_subj = load_and_preprocess(subject)
        X_test.append(X_subj)
        y_test.append(y_subj)
    
    train_dataset = PreprocessedDataset(X_train, y_train)
    test_dataset = PreprocessedDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    # some other projects use batch size of 16 with the PhysioNet dataset
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader