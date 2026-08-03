import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Dataset, DataLoader
from model import CNN
from dataset import PreprocessedDataset, load_and_preprocess
from datamodule import create_dataloaders

preprocessor = PreprocessedDataset()
X, y = preprocessor.load_and_preprocess()

# model = CNNTransformer()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = create_dataloaders(X, y)

# Training loop

num_epochs = 8 # number of epochs

for epoch in range(num_epochs):
    for batch in dataloader:
        #train(model, train_loader)
        #optimizer.zero_grad()
        #pred = model(x)
        #loss = nn.CrossEntropyLoss(pred, x)
        #loss.backward()
        optimizer.step()
    print(loss.item())
