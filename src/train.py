import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Dataset, DataLoader

from model import BaselineMLP, EEGClassifier
from dataset import PreprocessedDataset, load_and_preprocess
from datamodule import create_dataloaders


""" 
TO DO:

* ensure all necessary params given
* training loop completeness and functionality
"""


# Device is assigned to cuda (GPU) if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Ensure deterministic behavior crucial for replication
torch.backends.cuddn.deterministic = True

# Preprocess data
preprocessor = PreprocessedDataset()
X, y = preprocessor.load_and_preprocess()

model_baseline = BaselineMLP(input_dim=, num_classes=) 
# hidden_dim and dropout could also be modified if desired
model_main = EEGClassifier()
# add the params for the model: (patch_size=32, in_channels=32, embed_dim=128, 
# num_patches = 10, out_channels_cnn=64, transfomer_layers=2, num_heads=4, 
# ff_dim=256, expansion = ?, dropout_rate=0.1)

optimizer_main = optim.Adam(model_main.parameters(), lr=0.001)
optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.001)

# Create train and test dataloaders
train_loader, test_loader = create_dataloaders(X, y)

# Training loop for main model
num_epochs = 8 # number of epochs

for epoch in range(num_epochs):
    for batch in dataloader:
        train(model_main, train_loader)
        optimizer_main.zero_grad()
        pred = model_main(x)
        loss = nn.CrossEntropyLoss(pred, x)
        loss.backward()
        optimizer_main.step()
    print(loss.item())

# Training loop for baseline model
for epoch in range(num_epochs):
    for batch in dataloader:
        train(model_baseline, train_loader)
        optimizer_baseline.zero_grad()
        pred = model_baseline(x)
        loss = nn.CrossEntropyLoss(pred, x)
        loss.backward()
        optimizer_baseline.step()
    print(loss.item)