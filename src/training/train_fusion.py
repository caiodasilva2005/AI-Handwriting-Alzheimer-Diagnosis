"""
Fusion Model training loop
"""

import os
import sys
from os import path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data.image_loader import DarwinDownloader, HandwritingAlzheimerDataset, SampleType
from data.kinematic_loader import load_kinematic_data
from models.fusion import IntermediateFusionModel

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
CNN_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth")
MLP_PATH = path.join(PROJECT_ROOT, "models", "mlp_fusion.pth")
CSV_PATH = path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv")
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "fusion.pth")

def train_fusion(model, image_loader, kinematic_loader,
                 num_epochs=100, learning_rate=0.001, patience=10):

    model = model.to("cpu")
    criterion = nn.BCELoss()

    # Only the combiner has requires_grad=True; CNN and MLP are frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    epoch_losses = []
    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()  # base models stay in eval() via the model's train() override
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Step both loaders in lockstep; entries correspond by order.
        for (images, img_labels), (kin_features, _) in zip(image_loader, kinematic_loader):
            images = images.to("cpu")
            kin_features = kin_features.to("cpu")
            # Image label is the target (kinematic label is assumed identical).
            labels = img_labels.to("cpu").float().unsqueeze(1)

            predictions = model(images, kin_features)  # (B, 1) P(Alzheimer)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((predictions >= 0.5).float() == labels).sum().item()
            total += labels.size(0)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        acc = correct / max(total, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        # Early stopping on training loss.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, epoch_losses


if __name__ == "__main__":

    BATCH_SIZE = 16

    # --- Image loader (shuffle=False so order is stable for pairing) ---
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((299, 299)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    downloader = DarwinDownloader()
    annotations_path, image_dir = downloader.generateAnnotations(SampleType.TRAIN, foldCount=1)
    image_dataset = HandwritingAlzheimerDataset(annotations_path, image_dir, transform=transform)
    image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Kinematic loader (uses the train split; shuffle handled inside) ---
    kinematic_loader, _test_loader, num_features = load_kinematic_data(
        CSV_PATH, k=18, batch_size=BATCH_SIZE)

    model = IntermediateFusionModel(CNN_PATH, MLP_PATH, mlp_input_size=num_features)
    model, _ = train_fusion(model, image_loader, kinematic_loader)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved fusion model to {MODEL_SAVE_PATH}")
