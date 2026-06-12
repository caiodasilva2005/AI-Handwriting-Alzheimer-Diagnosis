"""
Fusion MLP testing loop (MLP trained on the 18 kinematic features from the fusion dataset)
"""

import os
import sys
from os import path
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.mlp import KinematicMLP
from training.train_mlp_fusion import load_fusion_kinematic_data
from evaluation.metrics import evaluate

BATCH_SIZE = 16
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "mlp_fusion.pth")
CSV_PATH = path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv")

# Same seeded split as training, so this is the genuine held-out test set.
_train_loader, test_loader, _scaler = load_fusion_kinematic_data(
    CSV_PATH, batch_size=BATCH_SIZE)

model = KinematicMLP(input_size=18)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

# Prediction / label distributions as a sanity check (mirrors the other test scripts).
preds, gts = [], []
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features).squeeze(1)
        predicted = (outputs >= 0.5).float()
        preds.extend(int(p) for p in predicted.tolist())
        gts.extend(int(g) for g in labels.tolist())

print("pred distribution:", Counter(preds))
print("label distribution:", Counter(gts))

evaluate(model, test_loader)
