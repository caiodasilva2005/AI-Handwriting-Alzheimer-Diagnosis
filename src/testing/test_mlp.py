"""
MLP testing loop
"""

import os
import sys
from os import path
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from data.kinematic_loader import load_kinematic_data
from models.mlp import KinematicMLP
from evaluation.metrics import evaluate

BATCH_SIZE = 16
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "mlp.pth")

_, test_loader, input_size, _selector, _scaler = load_kinematic_data(
    path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv"),
    k=30,
    batch_size=BATCH_SIZE
)

model = KinematicMLP(input_size=input_size)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

preds, gts = [], []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features).squeeze(1)
        predicted = (outputs >= 0.5).float()
        preds.extend(predicted.tolist())
        gts.extend(labels.tolist())

print("pred distribution:", Counter(int(p) for p in preds))
print("label distribution:", Counter(int(g) for g in gts))

evaluate(model, test_loader)