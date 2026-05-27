"""
CNN training loop
"""

import os
from collections import Counter
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data.image_loader import DarwinDownloader, HandwritingAlzheimerDataset, SampleType
from models.cnn import CNN

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth")

downloader = DarwinDownloader()
annotations_path, image_dir = downloader.generateAnnotations(SampleType.TRAIN, foldCount=1)

dataset = HandwritingAlzheimerDataset(annotations_path, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("train label distribution:", Counter(dataset.img_labels.iloc[:, 1].tolist()))

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

model.train()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"epoch {epoch + 1}/{NUM_EPOCHS}  loss={running_loss / len(dataloader):.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
