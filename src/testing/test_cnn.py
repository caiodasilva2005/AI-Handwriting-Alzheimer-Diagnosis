"""
CNN testing loop
"""

from os import path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data.image_loader import DarwinDownloader, HandwritingAlzheimerDataset, SampleType
from models.cnn import CNN

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True)])

BATCH_SIZE = 16
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth")

downloader = DarwinDownloader()
annotations_path, image_dir = downloader.generateAnnotations(SampleType.TEST, foldCount=1)

dataset = HandwritingAlzheimerDataset(annotations_path, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNN()
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the test images: {100 * correct // total} %")
