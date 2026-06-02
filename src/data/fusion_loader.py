"""
fusion_loader.py
Loads matched (image, kinematic features, label) triplets
for each participant-task combination.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from sklearn.preprocessing import StandardScaler

from data.kinematic_loader import get_task_features_for_participant
from data.image_loader import get_image_for_participant

# Participants missing from image dataset
DROP_PARTICIPANTS = [46, 79, 106, 107, 133, 134, 135, 136, 137]

# Task 1 only exists in CSV, so fuse on tasks 2-25
FUSEABLE_TASKS = list(range(2, 26))

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_image_base():
    """Get image base path from kagglehub cache."""
    import pathlib
    kaggle_cache = pathlib.Path.home() / ".cache" / "kagglehub" / "datasets" / "tizianadalessandro" / "darwin-i"
    versions = list(kaggle_cache.glob("versions/*/dataset_offline - task_2_25/dataset_offline - task_2_25"))
    if versions:
        return str(versions[0])
    raise FileNotFoundError("Dataset not found. Run: import kagglehub; kagglehub.dataset_download('tizianadalessandro/darwin-i')")


class FusionDataset(Dataset):
    def __init__(self, csv_path):
        image_base = get_image_base()

        df = pd.read_csv(csv_path).copy()
        df['pid'] = df['ID'].str.extract(r'(\d+)').astype(int)
        df = df[~df['pid'].isin(DROP_PARTICIPANTS)].reset_index(drop=True)

        self.samples = []
        all_features = []

        for _, row in df.iterrows():
            pid = int(row['pid'])
            label = 1 if row['class'] == 'P' else 0

            for task_num in FUSEABLE_TASKS:
                img_path = get_image_for_participant(image_base, pid, task_num)
                if img_path is None:
                    continue
                features = get_task_features_for_participant(csv_path, pid, task_num)
                self.samples.append((img_path, features, label))
                all_features.append(features)

        # Scale kinematic features to mean=0, std=1
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.array(all_features))
        self.samples = [(self.samples[i][0], scaled[i], self.samples[i][2])
                        for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, features, label = self.samples[idx]
        image = transform(read_image(img_path, mode=ImageReadMode.RGB))
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return image, features, label


def save_fusion_dataset(csv_path, save_path="datasets/fusion_dataset.pkl"):
    """Build and save fusion dataset to disk."""
    dataset = FusionDataset(csv_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset.samples, f)


def load_saved_fusion_data(save_path="datasets/fusion_dataset.pkl", test_size=0.2, batch_size=16, random_seed=42):
    """Load pre-saved fusion dataset from disk.

    Each sample is a self-contained (image, kinematic, label) triplet already paired
    by participant and task, so neither loader shuffles - order is stable and the two
    streams stay aligned. A one-time seeded permutation only decides the train/test
    split (so both splits contain both classes), not per-epoch ordering.
    """
    np.random.seed(random_seed)

    with open(save_path, 'rb') as f:
        samples = pickle.load(f)

    dataset = FusionDataset.__new__(FusionDataset)
    dataset.samples = samples

    n = len(samples)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))

    train_loader = DataLoader(torch.utils.data.Subset(dataset, indices[:split]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, indices[split:]), batch_size=batch_size, shuffle=False)

    print(f"Train: {split} | Test: {n - split}")
    return train_loader, test_loader


if __name__ == "__main__":
    save_fusion_dataset("datasets/kinematic_data.csv")
