import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Dataset, DataLoader
from model import CNN
from dataset import PreprocessedDataset, load_and_preprocess
from datamodule import create_dataloaders


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuddn.deterministic = True

preprocessor = PreprocessedDataset()
X, y = preprocessor.load_and_preprocess()

# model = EEGClassifier(self, patch_size=32, in_channels=32, embed_dim=128, 
# num_patches = 10, out_channels_cnn=64, transfomer_layers=2, num_heads=4, 
# ff_dim=256, expansion = ?, dropout_rate=0.1)

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
