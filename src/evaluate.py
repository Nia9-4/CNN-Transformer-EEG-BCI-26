import warnings
import torch
from model import CNN

model = CNN()

model.load_state_dict(torch.load("checkpoint.pt"))

model.eval()

correct = 0

with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        correct += (pred==y).sum()

acc = correct / len(test_loader.dataset)
print(acc)