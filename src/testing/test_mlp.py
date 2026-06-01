"""
MLP testing loop
"""

from os import path
from collections import Counter

import torch
from torch.utils.data import DataLoader

from data.kinematic_loader import load_kinematic_data
from models.mlp import KinematicMLP

BATCH_SIZE = 16
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "mlp.pth")

_, test_loader, input_size = load_kinematic_data(
    path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv"),
    k=30,
    batch_size=BATCH_SIZE
)

model = KinematicMLP(input_size=input_size)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

correct = 0
total = 0
preds, gts = [], []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features).squeeze(1)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        preds.extend(predicted.tolist())
        gts.extend(labels.tolist())

print("pred distribution:", Counter(int(p) for p in preds))
print("label distribution:", Counter(int(g) for g in gts))
print(f"Accuracy of the network on the test samples: {100 * correct // total} %")