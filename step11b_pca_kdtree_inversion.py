import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial import cKDTree
from tqdm import tqdm

# === CONFIGURATION ===
ARCHIVE_PATH = "pca_lut_projection.npz"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/lut_pca_kdtree"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD PCA LUT ARCHIVE ===
print("✅ Loading PCA LUT archive...")
data = np.load(ARCHIVE_PATH)
X_inputs = data['X_inputs']
Y_pca = data['Y_pca']
pca_mean = data['pca_mean']
pca_components = data['pca_components']
wavelengths_obs = data['wavelengths_obs']
param_names = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil']

print(f"✅ Loaded PCA LUT with {X_inputs.shape[0]} samples in {Y_pca.shape[1]} PCA dims.")

# === BUILD KD-TREE IN PCA SPACE ===
kdtree = cKDTree(Y_pca)
print("✅ KD-tree built in PCA space.")

# === DEFINE WEIGHT VECTOR ===
weights = np.ones_like(wavelengths_obs)
weights[(wavelengths_obs >= 400) & (wavelengths_obs <= 750)] = 2.0
weights[(wavelengths_obs >= 970) & (wavelengths_obs <= 990)] = 2.0
weights[(wavelengths_obs >= 1120) & (wavelengths_obs <= 1140)] = 2.0
weights /= weights.mean()  # normalize to keep overall scale stable

# === LOAD OBSERVED REFLECTANCE ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs_full = ds['reflectance'].values[:, :10, :10]  # (bands, 10, 10)

# === INVERSION ===
param_maps = np.zeros((8, 10, 10))
var_maps = np.zeros((8, 10, 10))
residual_maps = np.zeros((10, 10))
fit_spectra = {}

print("✅ Starting inversion with weighted cost...")
start_time = time.time()

for i in tqdm(range(10), desc="Pixels"):
    for j in range(10):
        R_obs = R_obs_full[:, i, j]
        R_centered = R_obs - pca_mean
        R_proj = R_centered @ pca_components.T

        # Get 20 nearest in PCA space
        _, idxs = kdtree.query(R_proj, k=20)

        best_cost = np.inf
        best_idx = idxs[0]

        dR = np.gradient(R_obs, wavelengths_obs)

        for idx in idxs:
            Y_rec = Y_pca[idx] @ pca_components + pca_mean
            dY = np.gradient(Y_rec, wavelengths_obs)

            cost_val = np.sqrt(np.sum(weights * (R_obs - Y_rec) ** 2))
            cost_deriv = np.sqrt(np.sum(weights * (dR - dY) ** 2))
            cost_total = 0.5 * cost_val + 0.5 * cost_deriv

            if cost_total < best_cost:
                best_cost = cost_total
                best_idx = idx

        best_params = X_inputs[best_idx]
        param_maps[:, i, j] = best_params
        var_maps[:, i, j] = X_inputs[idxs].var(axis=0)
        residual_maps[i, j] = best_cost

        if (i, j) in [(0, 0), (3, 3), (5, 5), (7, 2)]:
            Y_fit = Y_pca[best_idx] @ pca_components + pca_mean
            fit_spectra[(i, j)] = (R_obs, Y_fit)

elapsed = time.time() - start_time
print(f"✅ Inversion completed in {elapsed:.2f} seconds for 100 pixels.")

# === PARAMETER MAPS ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, name in enumerate(param_names):
    im = axes[idx].imshow(param_maps[idx], cmap='viridis')
    axes[idx].set_title(f'{name} estimate')
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_estimates_weighted.jpg'), dpi=200)
print("✅ Saved parameter estimate map.")

# === VARIANCE MAPS ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, name in enumerate(param_names):
    im = axes[idx].imshow(var_maps[idx], cmap='magma')
    axes[idx].set_title(f'{name} variance')
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_variances_weighted.jpg'), dpi=200)
print("✅ Saved parameter variance map.")

# === RESIDUAL MAP ===
plt.figure(figsize=(6, 5))
plt.imshow(residual_maps, cmap='inferno')
plt.title("Combined Cost (Weighted Value + Derivative)")
plt.colorbar(label="Cost")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_cost_map_weighted.jpg'), dpi=200)
print("✅ Saved weighted cost map.")

# === PLOT SAMPLE FITS ===
for (i, j), (obs, fit) in fit_spectra.items():
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths_obs, obs, 'k--', label='Observed')
    plt.plot(wavelengths_obs, fit, 'r-', label='Best Fit')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Spectral Fit at pixel ({i},{j})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fit_pixel_{i}_{j}_weighted.jpg'), dpi=200)
    plt.close()
print("✅ Saved example spectral fits.")
