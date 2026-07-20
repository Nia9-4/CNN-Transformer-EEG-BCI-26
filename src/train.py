import torch
from model import CNN
from dataset import get_dataloader

model = CNN()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

n = 8 # number of epochs

for epoch in range(n):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.nn.CrossEntropyLoss(pred, x)
        loss.backward()
        optimizer.step()
    print(loss.item())
