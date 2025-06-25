import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
import time

# === CONFIGURATION ===
MODEL_CHECKPOINT = "model_checkpoints/pca_emulator_sobol.pt"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/pca_inverse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda')

# === LOAD OBSERVED DATA ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs = ds['reflectance'].values[:, 50, 50]
wavelengths, fwhm = ds['wavelengths'].values, ds['fwhm'].values
wl_highres = np.linspace(400, 2500, 2101)

# === LOAD EMULATOR ===
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)
X_mean = torch.tensor(checkpoint['X_mean'], device=device, dtype=torch.float32)
X_std = torch.tensor(checkpoint['X_std'], device=device, dtype=torch.float32)
pca_mean = torch.tensor(checkpoint['pca_mean'], device=device, dtype=torch.float32)
pca_components = torch.tensor(checkpoint['pca_components'], device=device, dtype=torch.float32)

class PCAEmulator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

emulator = PCAEmulator(len(X_mean), pca_components.shape[0]).to(device)
emulator.load_state_dict(checkpoint['model_state_dict'])
emulator.eval()

# === GPU CONVOLUTION ===
def convolve_to_sensor_gpu(r_highres_gpu, wl_highres, wl_sensor, fwhm_sensor):
    wl_highres_gpu = torch.tensor(wl_highres, device=device, dtype=torch.float32)
    refl_sensor_gpu = torch.zeros((r_highres_gpu.shape[0], len(wl_sensor)), device=device)
    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
        window_gpu = torch.exp(-0.5 * ((wl_highres_gpu - wl_c) / sigma)**2)
        window_gpu /= window_gpu.sum()
        refl_sensor_gpu[:, i] = (r_highres_gpu * window_gpu).sum(dim=1)
    return refl_sensor_gpu.cpu().numpy()

# === LUT PARAMETERS ===
param_bounds_original = np.array([
    [1.0, 10.0, 2.0, 0.0, 0.005, 0.005, 0.1, 0.0, -2.0],
    [2.5, 80.0, 20.0, 0.5, 0.05, 0.05, 6.0, 1.0, 2.0]
])

GEOMETRY_PARAMS = np.array([30.0, 0.0, 0.0, 0.1])

# === GENERATE LUT ON GPU ===
n_samples = int(1e6)
batch_size = int(1e5)
n_batches = n_samples // batch_size

lut_params, lut_refl = [], []

print("✅ Generating LUT...")
with torch.no_grad():
    for batch in range(n_batches):
        random_samples = param_bounds_original[0] + (
            param_bounds_original[1] - param_bounds_original[0]
        ) * np.random.rand(batch_size, 9)
        params_tensor = torch.tensor(random_samples, dtype=torch.float32, device=device)
        geom_params_gpu = torch.tensor(GEOMETRY_PARAMS, device=device, dtype=torch.float32).repeat(batch_size, 1)
        full_params = torch.cat([params_tensor, geom_params_gpu], dim=1)
        x_norm = (full_params - X_mean) / X_std
        pred_scores = emulator(x_norm)
        pred_refl = pred_scores @ pca_components + pca_mean
        refl_sensor_batch = convolve_to_sensor_gpu(pred_refl, wl_highres, wavelengths, fwhm)
        lut_params.append(random_samples)
        lut_refl.append(refl_sensor_batch)
        print(f"Batch {batch+1}/{n_batches} completed.")

lut_params = np.vstack(lut_params)
lut_refl = np.vstack(lut_refl)

# === KD-TREE ===
kdtree = cKDTree(lut_refl)
print("✅ KD-Tree built.")

# === NEAREST 500 RETRIEVAL ===
_, idxs500 = kdtree.query(R_obs, k=500)
nearest_params500 = lut_params[idxs500]
nearest_refl500 = lut_refl[idxs500]

# === PLOT ALL SAMPLES IN SPECTRAL SPACE ===
plt.figure(figsize=(10, 6))
for refl in nearest_refl500:
    plt.plot(wavelengths, refl, color='blue', alpha=0.05)
plt.plot(wavelengths, R_obs, 'k--', linewidth=2, label='Observed')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('500 Nearest Neighbor Samples in Reflectance Space')
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nearest_samples_reflectance.png'), dpi=200)
print("✅ Saved reflectance samples plot.")

# === PLOT ALL SAMPLES IN PARAMETER SPACE ===
param_names = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil', 'LIDFa']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i in range(9):
    axes[i].scatter(range(len(nearest_params500)), nearest_params500[:, i],
                    color='green', alpha=0.3, s=10)
    axes[i].set_title(f'{param_names[i]}')
    axes[i].set_xlabel('Sample index')
    axes[i].grid(alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nearest_samples_parameters.png'), dpi=200)
print("✅ Saved parameter samples plot.")

# Compute the spectral residual (cost) for coloring
spectral_residuals = np.linalg.norm(nearest_refl500 - R_obs, axis=1)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
sc = None
for i in range(9):
    sc = axes[i].scatter(range(len(nearest_params500)), nearest_params500[:, i],
                         c=spectral_residuals, cmap='viridis_r', s=15, alpha=0.7)
    axes[i].set_title(f'{param_names[i]}')
    axes[i].set_xlabel('Sample index')
    axes[i].grid(alpha=0.5)
    
fig.colorbar(sc, ax=axes, label='Spectral Residual (Cost)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nearest_samples_parameters_colored_cost.png'), dpi=200)
print("✅ Saved parameter samples colored by cost.")
