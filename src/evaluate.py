import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    PROJECT_ROOT = os.getcwd()

print(f"Project Root identified as: {PROJECT_ROOT}")


def evaluate(model, val_loader, criterion, device):
    model.eval() # disable dropout and batchnorm
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().view(-1)
    
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            val_loss += loss.item()
            predicted_classes = (torch.sigmoid(pred) > 0.5).int()
            val_correct += (predicted_classes == batch_y).sum().item()
            val_total += batch_y.size(0)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    epoch_kappa = cohen_kappa_score(all_targets, all_preds)

    return {
        'loss': val_loss / len(val_loader),
        'acc': 100 * val_correct / val_total,
        'kappa': epoch_kappa
    }


def get_predictions(model, test_loader, device):
    """Collects all true labels and predictions"""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().view(-1)
            pred = model(batch_X)
        
            predicted_classes = (torch.sigmoid(pred) > 0.5).int()
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    return np.array(all_targets), np.array(all_preds)


def visualize_predictions(y_true, y_pred, model_name="model", save=False):
    """Visualize prediction accuracy with confusion matrices"""

    unique, counts = np.unique(y_pred, return_counts=True)
    print("Prediction Distribution:")
    print(dict(zip(unique, counts)))

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')

    if save:
        plot_dir = os.path.join(PROJECT_ROOT, 'results', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{model_name}_confusion_matrix.png"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    else:
        plt.show()