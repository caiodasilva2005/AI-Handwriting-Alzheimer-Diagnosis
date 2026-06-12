"""
Fusion Model testing loop

Loads the trained intermediate-fusion model (models/fusion.pth) and evaluates it
on the held-out test split produced by the fusion loader. Reports the prediction /
label distributions plus accuracy, precision, recall and F1.
"""

import os
import sys
from os import path
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from data.fusion_loader import save_fusion_dataset, load_saved_fusion_data
from models.fusion import IntermediateFusionModel

BATCH_SIZE = 16
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
CNN_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth")
MLP_PATH = path.join(PROJECT_ROOT, "models", "mlp_fusion.pth")
CSV_PATH = path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv")
FUSION_PKL = path.join(PROJECT_ROOT, "datasets", "fusion_dataset.pkl")
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "fusion.pth")

MLP_INPUT_SIZE = 18
if not path.exists(FUSION_PKL):
    save_fusion_dataset(CSV_PATH, save_path=FUSION_PKL)

_train_loader, test_loader = load_saved_fusion_data(
    save_path=FUSION_PKL, batch_size=BATCH_SIZE)

model = IntermediateFusionModel(CNN_PATH, MLP_PATH, mlp_input_size=MLP_INPUT_SIZE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

correct = 0
total = 0
preds, gts = [], []

# Confusion-matrix counts for precision / recall / F1.
TP = TN = FP = FN = 0

with torch.no_grad():
    for images, features, labels in test_loader:
        outputs = model(images, features).squeeze(1)  # P(Alzheimer), (B,)
        predicted = (outputs >= 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        preds.extend(int(p) for p in predicted.tolist())
        gts.extend(int(g) for g in labels.tolist())

        TP += ((predicted == 1) & (labels == 1)).sum().item()
        TN += ((predicted == 0) & (labels == 0)).sum().item()
        FP += ((predicted == 1) & (labels == 0)).sum().item()
        FN += ((predicted == 0) & (labels == 1)).sum().item()

accuracy = correct / total if total else 0.0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("pred distribution:", Counter(preds))
print("label distribution:", Counter(gts))
print(f"\n{'='*40}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"{'='*40}")
