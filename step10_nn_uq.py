import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

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

# === GPU CONVOLUTION FUNCTION ===
def convolve_to_sensor_gpu(r_highres_gpu, wl_highres, wl_sensor, fwhm_sensor):
    wl_highres_gpu = torch.tensor(wl_highres, device=device, dtype=torch.float32)
    refl_sensor_gpu = torch.zeros((r_highres_gpu.shape[0], len(wl_sensor)), device=device)
    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
        window_gpu = torch.exp(-0.5 * ((wl_highres_gpu - wl_c) / sigma)**2)
        window_gpu /= window_gpu.sum()
        refl_sensor_gpu[:, i] = (r_highres_gpu * window_gpu).sum(dim=1)
    return refl_sensor_gpu.cpu().numpy()

# === LUT SETUP ===
param_bounds_original = np.array([
    [1.0, 10.0, 2.0, 0.0, 0.005, 0.005, 0.1, 0.0, -2.0],
    [2.5, 80.0, 20.0, 0.5, 0.05, 0.05, 6.0, 1.0, 2.0]
])

GEOMETRY_PARAMS = np.array([30.0, 0.0, 0.0, 0.1])

# === GENERATE LUT (once) ===
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
kdtree = cKDTree(lut_refl)
print("✅ LUT and KD-Tree ready.")

# === Nearest 20 Retrieval ===
distances, indices = kdtree.query(R_obs, k=20)
best_params = lut_params[indices[0]]
closest_20_params = lut_params[indices]

# Uncertainty estimates from closest 20
mean_params = np.mean(closest_20_params, axis=0)
std_params = np.std(closest_20_params, axis=0)

# === OUTPUT PARAMETER ESTIMATES ===
param_names = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil', 'LIDFa']
print("✅ Best fit parameters and uncertainties:")
for i, name in enumerate(param_names):
    print(f"{name}: Best={best_params[i]:.3f}, Mean(20)={mean_params[i]:.3f}, Std(20)={std_params[i]:.3f}")

# === PLOT REFLECTANCE ===
best_refl = lut_refl[indices[0]]

plt.figure(figsize=(10,5))
plt.plot(wavelengths, R_obs, 'k--', label='Observed')
plt.plot(wavelengths, best_refl, 'r-', label='Best Fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.title('Observed vs Best Fit Reflectance')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'best_fit_reflectance.png'), dpi=200)
print("✅ Saved best fit reflectance plot.")

# === PARAMETER UNCERTAINTY PLOTS ===
fig, axes = plt.subplots(3,3,figsize=(15,12))
axes = axes.flatten()
for i in range(9):
    axes[i].hist(closest_20_params[:, i], bins=10, color='skyblue', alpha=0.7)
    axes[i].axvline(best_params[i], color='red', linestyle='--', label='Best Fit')
    axes[i].set_title(f'{param_names[i]}: {best_params[i]:.3f}')
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'input_params_best_fit_uncertainty.png'), dpi=200)
print("✅ Saved parameter uncertainty plot.")
