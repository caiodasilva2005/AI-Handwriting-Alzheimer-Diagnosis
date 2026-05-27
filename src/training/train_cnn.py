"""
CNN training loop
"""

import os
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
    v2.ToDtype(torch.float32, scale=True)])

NUM_ITERATIONS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))                                                                                                                                           
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth") 

downloader = DarwinDownloader()
annotations_path, image_dir = downloader.generateAnnotations(SampleType.TRAIN, foldCount=1)

dataset = HandwritingAlzheimerDataset(annotations_path, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
loss_function = nn.CrossEntropyLoss()

for iteration in range(NUM_ITERATIONS):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
