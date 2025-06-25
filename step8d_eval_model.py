import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load data and normalization ---
data_dir = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
X = np.load(os.path.join(data_dir, "X_inputs.npy"))
Y = np.load(os.path.join(data_dir, "Y_reflectance.npy"))

invalid_mask = np.any(np.isnan(X) | np.isinf(X), axis=1) | np.any(np.isnan(Y) | np.isinf(Y), axis=1)
X = X[~invalid_mask]
Y = Y[~invalid_mask]

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1e-6
X_norm = (X - X_mean) / X_std

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# --- Define and load model ---
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Emulator().to(device)
model.load_state_dict(torch.load("model_checkpoints/emulator_final.pt", map_location=device))
model.eval()

# --- Predict on a random subset ---
idx = np.random.choice(len(X_tensor), size=5, replace=False)
xb = X_tensor[idx].to(device)
yb = Y_tensor[idx].numpy()
with torch.no_grad():
    preds = model(xb).cpu().numpy()

# --- Plot ---
wl = np.linspace(400, 2500, 2101)
os.makedirs("figures/emulator", exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
    axs[i].plot(wl, yb[i], label='Measured', color='black')
    axs[i].plot(wl, preds[i], label='Predicted', color='orange')
    axs[i].set_title(f"Sample {i+1}")
    axs[i].set_xlabel("Wavelength (nm)")
    if i == 0:
        axs[i].set_ylabel("Reflectance")
    axs[i].legend()
plt.tight_layout()
plt.savefig("figures/emulator/inference_examples.png", dpi=300)
print("✅ Saved plot to figures/emulator/inference_examples.png")
