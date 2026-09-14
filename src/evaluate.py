import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


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
        'loss': val_loss / len(val_loader),
        'acc': 100 * val_correct / val_total,
        'kappa': cohen_kappa_score(val_targets, val_preds)
    }