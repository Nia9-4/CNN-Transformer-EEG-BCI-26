import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from model import CNN
from dataset import get_dataloader

model = CNN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

n = 8 # number of epochs

for epoch in range(n):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.CrossEntropyLoss(pred, x)
        loss.backward()
        optimizer.step()
    print(loss.item())
