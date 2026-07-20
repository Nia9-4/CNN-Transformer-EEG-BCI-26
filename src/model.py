import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network == nn.Sequential(
            nn.Conv2d(),
            nn.ReLu(),
            nn.Flatten(),
            nn.Linear
        )

        def forward(self, x):
            return self.network(x)