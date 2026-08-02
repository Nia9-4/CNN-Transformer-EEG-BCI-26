import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Dataset, DataLoader
from model import CNN
from dataset import PreprocessedDataset

dataset = PreprocessedDataset(#X_train_csp, y_train)

model = CNNTransformer()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop

n = 8 # number of epochs

for epoch in range(n):
    for batch in dataloader:
        inputs, labels = batch['data'], batch['labels']
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.CrossEntropyLoss(pred, x)
        loss.backward()
        optimizer.step()
    print(loss.item())
