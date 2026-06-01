"""
Loads Darwin.csv (Kinematic data)
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


class KinematicDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_kinematic_data(csv_path, k=50, test_size=0.2, batch_size=16, random_seed=42):
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["ID", "class"])
    y = (df["class"] == "P").astype(int).values

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    indices = np.random.permutation(len(X_scaled))
    X_scaled, y = X_scaled[indices], y[indices]

    split = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(KinematicDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(KinematicDataset(X_test, y_test),  batch_size=batch_size, shuffle=False)

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples | Features: {X_train.shape[1]}")

    return train_loader, test_loader, X_train.shape[1]

def load_kinematic_with_ids(csv_path, k=30, random_seed=42):
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["ID", "class"])
    y = (df["class"] == "P").astype(int).values

    participant_ids = df["ID"].str.extract(r'(\d+)').astype(int).values.flatten()

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    kinematic_map = {}
    for i, pid in enumerate(participant_ids):
        kinematic_map[pid] = (X_scaled[i], y[i])

    print(f"Loaded kinematic data for {len(kinematic_map)} participants")
    return kinematic_map

def get_task_features_for_participant(csv_path, pid, task_num):
    """
    Returns the 18 kinematic features for a specific participant and task.
    """
    import re
    df = pd.read_csv(csv_path)
    df['pid'] = df['ID'].str.extract(r'(\d+)').astype(int)
    
    row = df[df['pid'] == pid].iloc[0]
    cols = [c for c in df.columns if re.search(r'[a-z]' + str(task_num) + r'$', c)]
    return row[cols].values.astype(np.float32)