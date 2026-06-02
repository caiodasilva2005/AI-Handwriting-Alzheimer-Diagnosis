"""
Fusion Model training loop
"""

import os
import sys
from os import path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from data.fusion_loader import save_fusion_dataset, load_saved_fusion_data
from models.fusion import IntermediateFusionModel

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
CNN_PATH = path.join(PROJECT_ROOT, "models", "cnn.pth")
MLP_PATH = path.join(PROJECT_ROOT, "models", "mlp_fusion.pth")
CSV_PATH = path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv")
FUSION_PKL = path.join(PROJECT_ROOT, "datasets", "fusion_dataset.pkl")
MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "fusion.pth")

# The fusion MLP (models/mlp_fusion.pth) consumes 18 raw kinematic features per task.
MLP_INPUT_SIZE = 18

def precompute_features(model, fusion_loader):
    """Run the frozen CNN+MLP once over the whole loader and cache the fused
    (N, 192) feature vectors and (N, 1) labels. These never change during
    training, so doing this once avoids re-running the CNN every epoch."""
    model = model.to("cpu")
    model.eval()

    feature_batches = []
    label_batches = []
    with torch.no_grad():
        for images, kin_features, batch_labels in fusion_loader:
            feature_batches.append(model.extract_features(images.to("cpu"), kin_features.to("cpu")))
            label_batches.append(batch_labels.to("cpu").float().unsqueeze(1))

    features = torch.cat(feature_batches, dim=0)  # (N, 192)
    labels = torch.cat(label_batches, dim=0)      # (N, 1)
    print(f"Precomputed features: {tuple(features.shape)} over {labels.size(0)} samples")
    return features, labels


def train_fusion(model, fusion_loader, num_epochs=100, learning_rate=0.001,
                 patience=10, batch_size=16):

    model = model.to("cpu")
    criterion = nn.BCELoss()

    # Only the combiner has requires_grad=True; CNN and MLP are frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Extract the frozen features a single time, then train only the combiner.
    features, labels = precompute_features(model, fusion_loader)
    num_samples = features.size(0)

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

        # Iterate the cached features in fixed-order batches (no shuffling).
        for start in range(0, num_samples, batch_size):
            feat = features[start:start + batch_size]
            target = labels[start:start + batch_size]

            predictions = model.classify(feat)  # (B, 1) P(Alzheimer)
            loss = criterion(predictions, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((predictions >= 0.5).float() == target).sum().item()
            total += target.size(0)
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

    # Build the paired (image, kinematic, label) dataset once and cache it to disk.
    if not path.exists(FUSION_PKL):
        save_fusion_dataset(CSV_PATH, save_path=FUSION_PKL)

    # Loaders do not shuffle; each triplet is self-paired by (participant, task).
    train_loader, _test_loader = load_saved_fusion_data(
        save_path=FUSION_PKL, batch_size=BATCH_SIZE)

    model = IntermediateFusionModel(CNN_PATH, MLP_PATH, mlp_input_size=MLP_INPUT_SIZE)
    model, _ = train_fusion(model, train_loader)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved fusion model to {MODEL_SAVE_PATH}")
