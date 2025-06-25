"""
Trains a lightweight PyTorch emulator on PROSAIL data using a fast validation loop.
Input: 14 PROSAIL parameters, Output: 2101 reflectance bands
GPU-enabled (CUDA) for fast testing on NVIDIA A30
Includes debugging output and example prediction plots.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Load data ---
data_dir = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
X = np.load(os.path.join(data_dir, "X_inputs.npy"))
Y = np.load(os.path.join(data_dir, "Y_reflectance.npy"))
print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

# --- Screen and clean NaNs/Infs before anything else ---
invalid_mask = np.any(np.isnan(X) | np.isinf(X), axis=1) | np.any(np.isnan(Y) | np.isinf(Y), axis=1)
X = X[~invalid_mask]
Y = Y[~invalid_mask]
print(f"Filtered valid data: {X.shape[0]} samples remain")

# --- Optional: Clip extreme reflectance values to stabilize loss ---
print(f"Y min: {Y.min()}, max: {Y.max()}, mean: {Y.mean()}, std: {Y.std()}")
Y = np.clip(Y, 0.0, 2.0)

# --- Normalize inputs (standardization) ---
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1e-6  # Avoid division by zero
X_norm = (X - X_mean) / X_std

print(f"Mean of X: {X_mean[:5]}...\nStd of X: {X_std[:5]}...")

# --- Convert to PyTorch tensors ---
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

print(f"Converted to tensors: X_tensor {X_tensor.shape}, Y_tensor {Y_tensor.shape}")

# --- Dataset setup ---
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8192, pin_memory=True)

# --- Define model ---
class Emulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2101)
        )

    def forward(self, x):
        return self.net(x)

# --- Train ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Emulator().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print(f"Training on device: {device}")

for epoch in range(5):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if torch.isnan(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if torch.isnan(loss):
                continue
            val_loss += loss.item() * xb.size(0)
            if len(predictions) < 5:
                predictions.append(pred.cpu().numpy())
                targets.append(yb.cpu().numpy())

    val_loss /= len(val_dataset)
    print(f"Epoch {epoch+1}: Train MSE = {train_loss:.6f}, Validation MSE = {val_loss:.6f}")

# --- Save model ---
os.makedirs("model_checkpoints", exist_ok=True)
torch.save(model.state_dict(), "model_checkpoints/emulator_final.pt")
print("✅ Model saved to model_checkpoints/emulator_final.pt")

# --- Plot example predictions ---
preds = np.vstack(predictions)
trues = np.vstack(targets)
wl = np.linspace(400, 2500, 2101)

os.makedirs("figures/emulator", exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
    axs[i].plot(wl, trues[i], label='Measured', color='black')
    axs[i].plot(wl, preds[i], label='Predicted', color='orange')
    axs[i].set_title(f"Example {i+1}")
    axs[i].set_xlabel("Wavelength (nm)")
    if i == 0:
        axs[i].set_ylabel("Reflectance")
    axs[i].legend()
plt.tight_layout()
plt.savefig("figures/emulator/prediction_examples.png", dpi=300)
print("✅ Saved prediction examples to figures/emulator/prediction_examples.png")
