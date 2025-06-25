import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load data ---
data_dir = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
X = np.load(os.path.join(data_dir, "X_inputs.npy"))
Y = np.load(os.path.join(data_dir, "Y_reflectance.npy"))

# --- Clean and clip ---
mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.any(np.isnan(Y) | np.isinf(Y), axis=1)
X = X[mask]
Y = Y[mask]  # no clipping here — penalize instead

# --- Normalize X ---
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1e-6
np.save(os.path.join(data_dir, "X_mean_std.npy"), np.stack([X_mean, X_std]))
X_norm = (X - X_mean) / X_std

# --- Convert to tensors ---
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# --- Model ---
class SmoothedLinearEmulator(nn.Module):
    def __init__(self, input_dim=14, output_dim=2101, kernel_size=15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.smoother = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        with torch.no_grad():
            self.smoother.weight.fill_(1 / kernel_size)
            self.smoother.weight.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)       # (B, 2101)
        x = x.unsqueeze(1)        # (B, 1, 2101)
        x = self.smoother(x)      # Smooth in linear space
        return x.squeeze(1)       # (B, 2101)

model = SmoothedLinearEmulator().to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

# --- Loss function ---
def total_loss_fn(pred, target, lambda_d1=1e-2, lambda_d2=1e-2, lambda_bounds=1e-1):
    mse = nn.functional.mse_loss(pred, target)

    # First derivative
    d1 = pred[:, 1:] - pred[:, :-1]
    d1_loss = torch.mean(d1 ** 2)

    # Second derivative
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    d2_loss = torch.mean(d2 ** 2)

    # Penalize out-of-bound values (soft constraint)
    bound_violation = torch.relu(pred - 1) + torch.relu(-pred)
    bound_penalty = torch.mean(bound_violation ** 2)

    return mse + lambda_d1 * d1_loss + lambda_d2 * d2_loss + lambda_bounds * bound_penalty

# --- Training ---
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Training with smoother and penalty terms for 15 epochs...")

for epoch in range(15):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = total_loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1:2d}: Total Loss = {total_loss / len(dataset):.6f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    preds = model(X_tensor.to(device)).cpu().numpy()
    targets = Y_tensor.numpy()

# --- Per-band RMSE plot ---
band_rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))
wl = np.linspace(400, 2500, 2101)

os.makedirs("figures/emulator", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(wl, band_rmse)
plt.xlabel("Wavelength (nm)")
plt.ylabel("RMSE")
plt.title("Per-band RMSE after 15 epochs (linear + smoother + penalties)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/emulator/debug_band_rmse_smoothed_linear.png", dpi=300)
print("✅ Saved per-band RMSE plot to figures/emulator/debug_band_rmse_smoothed_linear.png")

# --- LAI sweep diagnostic ---
import matplotlib.cm as cm

standard_input = np.array([0.0] * 14)
standard_input[[1, 2, 3, 4, 5, 6]] = [40.0, 0.01, 0.009, 1.0, 0.0, 30.0]  # Cab, Cw, Cm, Asoil, view_angle, sza

lai_vals = np.linspace(0.1, 6.0, 10)
inputs = []
for lai in lai_vals:
    x = standard_input.copy()
    x[0] = lai
    inputs.append(x)

inputs = np.stack(inputs)
inputs_norm = (inputs - X_mean) / X_std
inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    preds_lai = model(inputs_tensor).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 5))
colors = cm.viridis((lai_vals - lai_vals.min()) / (lai_vals.max() - lai_vals.min()))

for i, (refl, lai) in enumerate(zip(preds_lai, lai_vals)):
    ax.plot(wl, refl, label=f"LAI={lai:.1f}", color=colors[i])

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("Emulator Output: Varying LAI (Standard Conditions)")
ax.grid(True, linestyle='--', alpha=0.5)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=lai_vals.min(), vmax=lai_vals.max()))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("LAI")
plt.tight_layout()
plt.savefig("figures/emulator/lai_sweep_reflectance_linear_smooth.png", dpi=300)
print("✅ Saved LAI sweep plot to figures/emulator/lai_sweep_reflectance_linear_smooth.png")
