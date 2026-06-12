"""
train_mlp_fusion.py
Trains and saves an MLP on 18 raw kinematic features per task.
This MLP is used as the kinematic stream in the fusion model.
Run from project root: python3 src/training/train_mlp_fusion.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from data.kinematic_loader import KinematicDataset
from models.mlp import KinematicMLP
from training.train_mlp import train_mlp
from evaluation.metrics import evaluate, plot_confusion_matrix
import re
import joblib

DROP_PARTICIPANTS = [46, 79, 106, 107, 133, 134, 135, 136, 137]
FUSEABLE_TASKS = list(range(2, 26))
CSV_PATH = "datasets/kinematic_data.csv"
MODEL_SAVE = "models/mlp_fusion.pth"
SCALER_SAVE = "models/fusion_scaler.pkl"

#    Loads kinematic data as 18 features per task per participant.
def load_fusion_kinematic_data(csv_path, test_size=0.2, batch_size=16, random_seed=42):
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path).copy()
    df['pid'] = df['ID'].str.extract(r'(\d+)').astype(int)
    df = df[~df['pid'].isin(DROP_PARTICIPANTS)].reset_index(drop=True)

    X_all = []
    y_all = []

    for _, row in df.iterrows():
        label = 1 if row['class'] == 'P' else 0
        for task_num in FUSEABLE_TASKS:
            cols = [c for c in df.columns if re.search(r'[a-z]' + str(task_num) + r'$', c)]
            features = row[cols].values.astype(np.float32)
            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Shuffle and split
    indices = np.random.permutation(len(X_scaled))
    X_scaled, y_all = X_scaled[indices], y_all[indices]
    split = int(len(X_scaled) * (1 - test_size))

    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    train_loader = DataLoader(KinematicDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(KinematicDataset(X_test, y_test),   batch_size=batch_size, shuffle=False)

    print(f"Fusion kinematic data: {len(X_train)} train | {len(X_test)} test | 18 features per task")
    return train_loader, test_loader, scaler


if __name__ == "__main__":
    train_loader, test_loader, scaler = load_fusion_kinematic_data(CSV_PATH)

    model = KinematicMLP(input_size=18, dropout_rate=0.4)
    model, losses = train_mlp(model, train_loader, test_loader, num_epochs=300, learning_rate=0.0005, patience=25)

    evaluate(model, test_loader)
    plot_confusion_matrix(model, test_loader)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE)
    joblib.dump(scaler, "models/fusion_scaler.pkl")
    
    print(f"Fusion MLP saved to {MODEL_SAVE}")
    print(f"Fusion scaler saved to {SCALER_SAVE}")