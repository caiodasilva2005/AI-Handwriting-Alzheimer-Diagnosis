"""
MLP Training loop 
"""
import torch
import torch.nn as nn
import os
from os import path


def train_mlp(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001, device="cpu", patience=10):

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    best_test_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((predictions >= 0.5).float() == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        epoch_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                preds = model(X_batch)
                test_loss += criterion(preds, y_batch).item()

        avg_test_loss = test_loss / len(test_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_test_loss:.4f}")

        # --- Early stopping ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, epoch_losses


if __name__ == "__main__":
    from data.kinematic_loader import load_kinematic_data
    from models.mlp import KinematicMLP

    PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
    CSV_PATH = path.join(PROJECT_ROOT, "datasets", "kinematic_data.csv")
    MODEL_SAVE_PATH = path.join(PROJECT_ROOT, "models", "mlp.pth")

    train_loader, test_loader, num_features = load_kinematic_data(CSV_PATH, k=50)

    model = KinematicMLP(input_size=num_features)
    model, _ = train_mlp(model, train_loader, test_loader)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved MLP to {MODEL_SAVE_PATH}")