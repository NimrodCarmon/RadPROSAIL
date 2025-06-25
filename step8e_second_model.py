"""
Trains a CNN-enhanced emulator on PROSAIL data.
Uses 1D convolution to enforce spectral smoothness and autocorrelation.
Adds spectral shape sensitivity using first-derivative loss.
Includes learning rate scheduler and debugging output.
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

# --- Clean NaNs/Infs ---
mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.any(np.isnan(Y) | np.isinf(Y), axis=1)
X = X[mask]
Y = Y[mask]
print(f"Filtered valid data: {X.shape[0]} samples remain")

# --- Clip Y for stability ---
Y = np.clip(Y, 0.0, 2.0)
print(f"Y stats after clip: min={Y.min()}, max={Y.max()}, mean={Y.mean()}, std={Y.std()}")

# --- Normalize inputs ---
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1e-6
# --- Save normalization stats for future inference ---
np.save(os.path.join(data_dir, "X_mean_std.npy"), np.stack([X_mean, X_std]))
print("✅ Saved X_mean and X_std to X_mean_std.npy")

X_norm = (X - X_mean) / X_std

# --- Convert to tensors ---
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# --- Dataset ---
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2048, pin_memory=True)

# --- Model ---
class CNNSpectralEmulator(nn.Module):
    def __init__(self, input_dim=14, output_dim=2101):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.smoother = nn.Conv1d(1, 1, kernel_size=15, padding=7, bias=False)
        with torch.no_grad():
            self.smoother.weight.fill_(1/15.0)
            self.smoother.weight.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = self.smoother(x)
        return x.squeeze(1)

# --- Loss Functions ---
def first_derivative(x):
    return x[:, 1:] - x[:, :-1]

def combined_loss(pred, target, alpha=0.2, beta=0.05):
    mse = nn.functional.mse_loss(pred, target)
    d_pred = first_derivative(pred)
    d_true = first_derivative(target)
    d_loss = nn.functional.mse_loss(d_pred, d_true)
    scale_loss = nn.functional.mse_loss(pred.mean(dim=1), target.mean(dim=1))
    return mse + alpha * d_loss + beta * scale_loss

# --- Train ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSpectralEmulator().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"Training on device: {device}")

prev_val = float('inf')
for epoch in range(15):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = combined_loss(pred, yb)
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
            loss = combined_loss(pred, yb)
            if torch.isnan(loss):
                continue
            val_loss += loss.item() * xb.size(0)
            if len(predictions) < 5:
                predictions.append(pred.cpu().numpy())
                targets.append(yb.cpu().numpy())

    val_loss /= len(val_dataset)
    scheduler.step(val_loss)
    improved = "✅" if epoch == 0 or val_loss < prev_val else "✖"
    print(f"Epoch {epoch+1}: Train MSE+Shape = {train_loss:.6f}, Validation MSE+Shape = {val_loss:.6f} {improved}")
    prev_val = val_loss

# --- Save model ---
os.makedirs("model_checkpoints", exist_ok=True)
torch.save(model.state_dict(), "model_checkpoints/emulator_cnn.pt")
print("✅ Model saved to model_checkpoints/emulator_cnn.pt")

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
plt.savefig("figures/emulator/prediction_examples_cnn.png", dpi=300)
print("✅ Saved prediction examples to figures/emulator/prediction_examples_cnn.png")

# --- Inference Speed Test ---
import time
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(10000, 14).to(device)
    torch.cuda.synchronize()
    start = time.time()
    _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Inference speed: {10000 / (end - start):,.0f} spectra/sec")
