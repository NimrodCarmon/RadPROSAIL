"""
Evaluation / stress‑test script for the CNN PROSAIL emulator

Goals
------
1.  Load trained model + normalization statistics.
2.  Generate a dense, uniform (or LH‑sampled) coverage of the 14‑D state space.
3.  Predict reflectance and flag
      • any wavelength with ρ < 0  (non‑physical)
      • any wavelength with ρ > 1  (also non‑physical for typical nadir BRDF).
4.  Compute global metrics (RMSE, SAM, % illegal values).
5.  Compute the same metrics inside user‑defined *regions* of the state space
   (here: deciles of each driving variable; easy to change).

The code is pure PyTorch/NumPy, no external deps beyond what the training
script already used.  It streams data in configurable mini‑batches so GPU/CPU
memory never spikes.

---------------------------------------------------------------------------
CONFIGURABLE SECTION
---------------------------------------------------------------------------

• `DATA_DIR`      – folder with **X_inputs.npy**, **Y_reflectance.npy**, **X_mean_std.npy**
• `CKPT_PATH`     – the *.pt* file with model weights.
• `N_STRESS`      – how many synthetic points to sample for the stress test.
                    100 k is usually enough; bump to 1 M if you trust your GPU.
• `BATCH`         – batch size for forward passes.
• `REGION_BINS`   – number of quantile bins per driver for regional metrics.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
DATA_DIR    = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
CKPT_PATH   = "model_checkpoints/emulator_cnn.pt"
N_STRESS    = 100_000
BATCH       = 4096
REGION_BINS = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_SEED  = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

# -----------------------------------------------------------------------------
# Helper: model definition must be byte‑identical to training time
# -----------------------------------------------------------------------------
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
            self.smoother.weight.fill_(1. / 15.)
            self.smoother.weight.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = self.smoother(x)
        return x.squeeze(1)

# -----------------------------------------------------------------------------
# 1.  Load data + stats
# -----------------------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X_inputs.npy"))
Y = np.load(os.path.join(DATA_DIR, "Y_reflectance.npy"))
mask = ~np.any(~np.isfinite(X), axis=1) & ~np.any(~np.isfinite(Y), axis=1)
X, Y = X[mask], Y[mask]

X_mean, X_std = np.load(os.path.join(DATA_DIR, "X_mean_std.npy"))
X_std[X_std == 0] = 1e-6

x_min, x_max = X.min(0), X.max(0)

# -----------------------------------------------------------------------------
# 2.  Build / load model
# -----------------------------------------------------------------------------
model = CNNSpectralEmulator().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# -----------------------------------------------------------------------------
# 3.  Utility metrics
# -----------------------------------------------------------------------------
def rmse(pred, true):          # shape (N, L)
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))

def sam(pred, true):           # spectral angle mapper (radians)
    num = np.sum(pred * true, axis=1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    den[den == 0] = 1e-12
    return np.arccos(np.clip(num / den, -1, 1))

# -----------------------------------------------------------------------------
# 4.  Evaluation on the *original* validation set (accuracy sanity check)
# -----------------------------------------------------------------------------
X_val_t = torch.tensor((X - X_mean) / X_std, dtype=torch.float32)
Y_val_t = torch.tensor(Y, dtype=torch.float32)

val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t),
                        batch_size=BATCH, shuffle=False, pin_memory=True)

preds_true, preds_hat = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        phat = model(xb.to(DEVICE)).cpu()
        preds_hat.append(phat)
        preds_true.append(yb)
preds_hat = torch.cat(preds_hat, 0).numpy()
preds_true = torch.cat(preds_true, 0).numpy()

val_RMSE = rmse(preds_hat, preds_true).mean()
val_SAM  = sam(preds_hat, preds_true).mean()
print(f"[Validation]   RMSE: {val_RMSE:.4f}  | SAM: {val_SAM*180/np.pi:.3f} deg")

# -----------------------------------------------------------------------------
# 5.  Dense stress‑test sampling
# -----------------------------------------------------------------------------
#   Uniform sampling inside empirical [min,max] range for every driver.
u = np.random.rand(N_STRESS, X.shape[1])
X_stress = x_min + u * (x_max - x_min)
X_stress_t = torch.tensor((X_stress - X_mean) / X_std, dtype=torch.float32)

stress_loader = DataLoader(X_stress_t, batch_size=BATCH, shuffle=False, pin_memory=True)

neg_mask_total   = np.zeros((N_STRESS, 2101), dtype=bool)
gt1_mask_total   = np.zeros_like(neg_mask_total)
pred_stress      = np.empty_like(neg_mask_total, dtype=np.float32)

with torch.no_grad():
    idx0 = 0
    for xb in stress_loader:
        bs = xb.size(0)
        phat = model(xb.to(DEVICE)).cpu().numpy()
        pred_stress[idx0:idx0+bs] = phat
        neg_mask_total[idx0:idx0+bs] = phat < 0
        gt1_mask_total[idx0:idx0+bs] = phat > 1
        idx0 += bs

n_spec_neg = (neg_mask_total.any(axis=1)).sum()
n_spec_gt1 = (gt1_mask_total.any(axis=1)).sum()
print(f"[Stress‑test]  spectra with <0    : {n_spec_neg / N_STRESS:.3%}")
print(f"[Stress‑test]  spectra with >1    : {n_spec_gt1 / N_STRESS:.3%}")
print(f"[Stress‑test]  total illegal vals : {np.logical_or(neg_mask_total, gt1_mask_total).sum()/neg_mask_total.size:.3%}")

# -----------------------------------------------------------------------------
# 6.  Regional accuracy / legality statistics
#    – Each driver variable is binned into deciles (REGION_BINS) independently.
# -----------------------------------------------------------------------------
def regional_stats(param_idx, name):
    bins = np.quantile(X[:, param_idx], np.linspace(0, 1, REGION_BINS+1))
    rmse_bin, sam_bin, neg_bin = [], [], []
    for i in range(REGION_BINS):
        lo, hi = bins[i], bins[i+1]
        sel = (X[:, param_idx] >= lo) & (X[:, param_idx] < hi)
        if not np.any(sel):
            rmse_bin.append(np.nan); sam_bin.append(np.nan); neg_bin.append(np.nan); continue
        phat = preds_hat[sel]
        ptrue = preds_true[sel]
        rmse_bin.append(rmse(phat, ptrue).mean())
        sam_bin.append(sam(phat, ptrue).mean())
        neg_bin.append((phat < 0).any(axis=1).mean())
    print(f"\n[Regional] Driver {param_idx} ({name})")
    print(" bin  RMSE  SAM(deg)  %neg")
    for i in range(REGION_BINS):
        print(f"{i:>3d}  {rmse_bin[i]:.4f}  {sam_bin[i]*180/np.pi:.2f}   {neg_bin[i]*100:.2f}")

#  Example for the first three drivers
param_names = ["LAI", "Cab", "Cw"] + [f"P{i}" for i in range(4,14)]
for idx in range(3):
    regional_stats(idx, param_names[idx])

# -----------------------------------------------------------------------------
# 7.  Optional: wavelength‑wise distribution of negatives (quick overview)
# -----------------------------------------------------------------------------
neg_per_wl = neg_mask_total.mean(0) * 100
most_problematic = np.argsort(neg_per_wl)[-10:][::-1]   # top‑10 wavelengths
print("\n[Stress‑test] worst wavelengths (nm | %neg):")
wl = np.linspace(400, 2500, 2101)
for i in most_problematic:
    print(f"{int(wl[i]):4d}  {neg_per_wl[i]:6.2f}")

# -----------------------------------------------------------------------------
# 8.  Wall‑clock performance
# -----------------------------------------------------------------------------
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
t0 = time.time()
with torch.no_grad():
    _ = model(torch.randn(10000, 14, device=DEVICE))
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
print(f"\nInference throughput: {10000 / (time.time() - t0):,.0f} spectra /s on {DEVICE}")
