import warnings
import torch
from model import BaselineMLP, EEGClassifier


"""
TO DO:

* check whether functions are still missing or everything works
* do models really have to be loaded separately?
* perform little test with small dataset
* add plots for visualization purpose
"""

# ========================
# Evaluate baseline model
# ========================

model_baseline = BaselineMLP(input_dim=, num_classes=) 
model_baseline.load_state_dict(torch.load("checkpoint.pt"))
model_baseline.eval()

correct = 0

with torch.no_grad():
    for x, y in test_loader:
        output_baseline = model_baseline(x)
        pred_baseline = output_baseline.argmax(dim=1)
        correct_baseline += (pred_baseline==y).sum()

acc_baseline = correct_baseline / len(test_loader.dataset)
print(f"Accuracy of the baseline MLP: {acc_baseline}")


# ====================
# Evaluate main model
# ====================

model_main = EEGClassifier() # add params here
model_main.load_state_dict(torch.load("checkpoint.pt"))
model_main.eval()

correct = 0

with torch.no_grad():
    for x, y in test_loader:
        output_main = model_main(x)
        pred_main = output_main.argmax(dim=1)
        correct_main += (pred_main==y).sum()

acc_main = correct_main / len(test_loader.dataset)
print(f"Accuracy of the main EEG Classifier (CNN-Transformer): {acc_main}")