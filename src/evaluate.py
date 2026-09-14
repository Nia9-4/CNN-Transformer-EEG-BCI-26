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

from model import PoorMLP, BaselineCNN, EEGClassifier

seed = 42
torch.manual_seed(s)
np.random.seed(s)


def evaluate(model, val_loader, criterion, device):
    model.eval() # disable dropout and batchnorm
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().view(-1)
    
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            val_loss += loss.item()
            predicted_classes = (torch.sigmoid(pred) > 0.5).int()
            val_correct += (predicted_classes == batch_y).sum().item()
            val_total += batch_y.size(0)
            val_preds.extend(predicted_classes.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())

    return {
        'loss': val_loss / len(loader),
        'acc': 100 * val_correct / val_total,
        'kappa': cohen_kappa_score(val_targets, val_preds)
    }