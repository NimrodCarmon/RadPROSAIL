import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

# === CONFIGURATION ===
MODEL_CHECKPOINT = "model_checkpoints/pca_emulator_sobol.pt"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/pca_inverse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_IDX = slice(0, 9)
device = torch.device('cuda')

# === LOAD DATA ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs = torch.tensor(ds['reflectance'].values[:, 50, 50], device=device, dtype=torch.float32)
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

# === GPU-BASED CONVOLUTION ===
def convolve_to_sensor_gpu(r_highres_gpu, wl_highres, wl_sensor, fwhm_sensor):
    wl_highres_gpu = torch.tensor(wl_highres, device=device, dtype=torch.float32)
    refl_sensor_gpu = torch.zeros((r_highres_gpu.shape[0], len(wl_sensor)), device=device, dtype=torch.float32)

    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
        window_gpu = torch.exp(-0.5 * ((wl_highres_gpu - wl_c) / sigma)**2)
        window_gpu /= window_gpu.sum()
        refl_sensor_gpu[:, i] = (r_highres_gpu * window_gpu).sum(dim=1)

    return refl_sensor_gpu

# === PARAMETERS AND GEOMETRY ===
param_bounds_original = torch.tensor([
    [1.0, 10.0, 2.0, 0.0, 0.005, 0.005, 0.1, 0.0, -2.0],
    [2.5, 80.0, 20.0, 0.5, 0.05, 0.05, 6.0, 1.0, 2.0]
], device=device, dtype=torch.float32)

GEOMETRY_PARAMS = torch.tensor([30.0, 0.0, 0.0, 0.1], device=device, dtype=torch.float32)

# === MINI-BATCHED MONTE CARLO ===
batch_size = int(1e5)  # 100k per batch
total_samples = int(1.5e6)
n_batches = total_samples // batch_size

best_residual_norm = float('inf')
best_params = None
best_refl = None

with torch.no_grad():
    for batch_idx in range(n_batches):
        random_samples = (param_bounds_original[0] + 
                          (param_bounds_original[1] - param_bounds_original[0]) * 
                          torch.rand(batch_size, 9, device=device, dtype=torch.float32))

        geom_params = GEOMETRY_PARAMS.repeat(batch_size, 1)
        full_params = torch.cat([random_samples, geom_params], dim=1)
        x_norm = (full_params - X_mean) / X_std

        pred_scores = emulator(x_norm)
        pred_refl = pred_scores @ pca_components + pca_mean

        refl_sensor = convolve_to_sensor_gpu(pred_refl, wl_highres, wavelengths, fwhm)
        residuals = refl_sensor - R_obs
        residual_norm = torch.norm(residuals, dim=1)

        batch_best_idx = torch.argmin(residual_norm)
        batch_best_norm = residual_norm[batch_best_idx]

        if batch_best_norm < best_residual_norm:
            best_residual_norm = batch_best_norm.item()
            best_params = random_samples[batch_best_idx].cpu().numpy()
            best_refl = refl_sensor[batch_best_idx].cpu().numpy()

        print(f"✅ Batch {batch_idx+1}/{n_batches} done. Best residual norm so far: {best_residual_norm:.4e}")

# === FINAL RESULTS ===
print("✅ Final Best Parameters (MC):", best_params)
print("✅ Final Minimum Residual Norm:", best_residual_norm)

# === PLOT RESULTS ===
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, R_obs.cpu().numpy(), 'k--', label='Observed')
plt.plot(wavelengths, best_refl, 'r-', label='MC Best Fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.grid(alpha=0.5)
plt.title('Monte Carlo Reflectance Optimization (Mini-Batched)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mc_optimized_reflectance.png'), dpi=200)
print("✅ Saved Monte Carlo reflectance comparison plot.")
