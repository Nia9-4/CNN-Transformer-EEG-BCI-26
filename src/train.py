import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from model import BaselineCNN, EEGClassifier
from datamodule import create_dataloaders
from evaluate import evaluate

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# Device is assigned to cuda (GPU) if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.getcwd()

print(f"Project Root identified as: {PROJECT_ROOT}")


# ========================
# Function to train model
# ========================

def train(model, n_epochs, train_loader, val_loader, optimizer, device, model_name="model"):

    # Extract parameters for model run name 
    current_lr = optimizer.param_groups[0]['lr']
    current_bs = train_loader.batch_size

    run_name = f"model_name_lr{current_lr}_bs{current_bs}"

    # Path setup
    model_dir = os.path.join(PROJECT_ROOT, 'results', 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"best_{run_name}.pth")

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

            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Kappa: {best_kappa:.4f}")
            break

    model.load_state_dict(torch.load(save_path))
    return model, history


# =========================
# Function to plot metrics
# =========================

def plot_metrics(history):
    plot_dir = os.path.join(PROJECT_ROOT, 'results', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_kappa'], label='Val Kappa', color='magenta')
    plt.title('Validation Kappa')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'loss_curve.png'))


# =======================
# Training the models
# =======================

# Only if explicitly called
if __name__ == "__main__":
    # Create train and test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=32)

    """# Hyperparameters for poor sanity check MLP
    n_channels = 64
    n_samples = int(4.5 * 160) # time_window * sampling_freq
    in_dim = n_channels * n_samples
    hidden_dim = 256"""

    # Initialize models and optimizers
    model_main = EEGClassifier()
    model_baseline = BaselineCNN()

    optimizer_main = optim.Adam(model_main.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop for models
    trained_eegclassifier, main_history = train(
        model=model_main, 
        n_epochs=50, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer_main, 
        device=device,
        model_name="eegclassifier")

    trained_baseline, baseline_history = train(
        model=model_baseline, 
        n_epochs=50, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer_baseline, 
        device=device,
        model_name="baseline_cnn")

    plot_metrics(main_history)
    plot_metrics(baseline_history)
