import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

# --- Parameters ---
DATA_DIR = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
N_COMPONENTS = 100
N_EPOCHS = 20
PCA_SAMPLE_SIZE = 100000
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and preprocess data ---
X = np.load(f"{DATA_DIR}/X_inputs_clean.npy")
Y = np.load(f"{DATA_DIR}/Y_reflectance_clean.npy")

X = np.delete(X, 13, axis=1)  # Remove constant feature explicitly

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std < 1e-8] = 1.0
X_norm = (X - X_mean) / X_std

# --- Sobol sequence sampling for PCA basis ---
sobol_sampler = qmc.Sobol(d=X.shape[1], scramble=True)
sobol_samples = sobol_sampler.random_base2(m=int(np.ceil(np.log2(PCA_SAMPLE_SIZE))))
X_sobol = qmc.scale(sobol_samples[:PCA_SAMPLE_SIZE], X.min(axis=0), X.max(axis=0))

nn_finder = NearestNeighbors(n_neighbors=1).fit(X)
_, indices = nn_finder.kneighbors(X_sobol)
Y_sample = Y[indices.flatten()]

# PCA fit
pca = PCA(n_components=N_COMPONENTS)
pca.fit(Y_sample)
print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

# Transform full dataset into PCA scores
Y_scores = pca.transform(Y)

# --- Prepare dataset ---
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scores, dtype=torch.float32)
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Neural network model ---
class PCAEmulator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = PCAEmulator(X.shape[1], N_COMPONENTS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- Training ---
print("✅ Starting training...")
for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= val_size

    print(f"Epoch {epoch+1}/{N_EPOCHS} | Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# --- Save trained model ---
os.makedirs("model_checkpoints", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean, 'X_std': X_std,
    'pca_mean': pca.mean_, 'pca_components': pca.components_,
}, "model_checkpoints/pca_emulator_sobol.pt")
print("✅ PCA emulator model saved.")

# --- Inference examples ---
num_examples = 5
val_inputs, val_true_scores = val_ds[:]
random_indices = np.random.choice(len(val_ds), num_examples, replace=False)
example_inputs = val_inputs[random_indices].to(DEVICE)
example_true_scores = val_true_scores[random_indices].cpu().numpy()

with torch.no_grad():
    example_pred_scores = model(example_inputs).cpu().numpy()

true_spectra = pca.inverse_transform(example_true_scores)
pred_spectra = pca.inverse_transform(example_pred_scores)

# Plot actual vs inferred spectra
wl = np.linspace(400, 2500, Y.shape[1])
fig, axs = plt.subplots(1, num_examples, figsize=(5*num_examples, 4), sharey=True)
for i in range(num_examples):
    axs[i].plot(wl, true_spectra[i], color='black', label='Actual')
    axs[i].plot(wl, pred_spectra[i], color='red', linestyle='--', label='Inferred')
    axs[i].set_title(f"Sample {i+1}")
    axs[i].set_xlabel("Wavelength (nm)")
    axs[i].grid(True)
    if i == 0:
        axs[i].set_ylabel("Reflectance")
    axs[i].legend()
plt.tight_layout()
plt.savefig("model_checkpoints/pca_emulator_inference_examples.png", dpi=300)
print("✅ Saved inference examples plot.")

# --- Per-band RMSE on validation set ---
print("✅ Calculating per-band RMSE on validation set...")
with torch.no_grad():
    val_pred_scores = model(val_inputs.to(DEVICE)).cpu().numpy()
val_true_spectra = pca.inverse_transform(val_true_scores.cpu().numpy())
val_pred_spectra = pca.inverse_transform(val_pred_scores)

per_band_rmse = np.sqrt(np.mean((val_true_spectra - val_pred_spectra)**2, axis=0))

plt.figure(figsize=(10, 4))
plt.plot(wl, per_band_rmse, color='blue')
plt.xlabel("Wavelength (nm)")
plt.ylabel("RMSE")
plt.title("Per-band RMSE on Validation Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_checkpoints/pca_emulator_per_band_rmse.png", dpi=300)
print("✅ Saved per-band RMSE plot.")

print("✅ All tasks completed successfully.")
