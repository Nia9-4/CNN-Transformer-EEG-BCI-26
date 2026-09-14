import numpy as np
import matplotlib.pyplot as plt
import mne 
import math
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader

from model import BaselineMLP, EEGClassifier
from dataset import PreprocessedDataset, load_data
from datamodule import create_dataloaders
from evaluate import evaluate

seed = 42
torch.manual_seed(s)
np.random.seed(s)


# Device is assigned to cuda (GPU) if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuddn.deterministic = True
torch.backends.cuddn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


# ========================
# Function to train model
# ========================

def train(model, n_epochs, train_loader, val_loader, optimizer, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  

    # LR warm-up
    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch+1) / float(warmup_epochs)
        return 1.0

    # Scheduler for LR warm-up and cosine annealing LR
    base_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Dictionary of metrics for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_kappa': []
    }

    # Early stopping setup
    best_kappa = -1.0
    patience = 50
    patience_counter =0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Ensure float tensor with same shape as preds
            batch_y = batch_y.float().view(-1)

            # Forward pass
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            # Backward pass
            loss.backward()

            # Clip gradients to avoid exploiding gradients for Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted_classes = (torch.sigmoid(pred) > 0.5).int()
            train_correct += (predicted_classes == batch_y).sum().item()
            train_total += batch_y.size(0)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            base_scheduler.step()

        # Training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        metrics = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(metrics['loss'])
        history['val_acc'].append(metrics['acc'])
        history['val_kappa'].append(metrics['kappa'])

        print(  f"Epoch {epoch+1}/{n_epochs} | Train Loss {avg_train_loss:.4f} | Val Kappa: {metrics['kappa']:.4f}")

        # Early stopping
        if metrics['kappa'] > best_kappa:
            best_kappa = metrics['kappa']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Kappa: {best_kappa:.4f}")
            break
        
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history


# =========================
# Function to plot metrics
# =========================

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.sublot(1, 2, 2)
    plt.plot(epochs, history['val_kappa'], label='Val Kappa', color='magenta')
    plt.title('Validation Kappa')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


# =======================
# Training the models
# =======================
# Load data 
dataset = PreprocessedDataset(subject_ids=[1, 2, 3, 4], runs=[4])
X, y = dataset.load_data()

sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}")
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create train and test dataloaders
train_loader, val_loader, test_loader = create_dataloaders(X, y)

# Hyperparameters for poor sanity check MLP
n_channels = 64
n_samples = 4.5 * 160 # time_window * sampling_freq
in_dim = n_channels * n_samples
hidden_dim = 256

# Initialize models and optimizers
model_main = EEGClassifier()
model_baseline = BaselineCNN()
model_poormlp = PoorMLP(in_dim, hidden_dim)

optimizer_main = optim.Adam(model_main.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_poormlp = optim.Adam(model_poormlp.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop for models
trained_eegclassifier, main_history = train(model=model_main, n_epochs=50, train_loader=train_loader, 
                                    val_loader=val_loader, optimizer=optimizer_main, device=device)
trained_baseline, baseline_history = train(model=model_baseline, n_epochs=50, train_loader=train_loader, 
                                    val_loader=val_loader, optimizer=optimizer_baseline, device=device)
trained_poormlp, poor_history = train(model=model_poormlp, n_epochs=50, train_loader=train_loader, 
                                    val_loader=val_loader, optimizer=optimizer_poormlp, device=device)

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plot_metrics(main_history)

plt.subplot(3, 1, 2)
plot_metrics(baseline_history)

plt.subplot(3, 1, 1)
plot_metrics(poor_history)